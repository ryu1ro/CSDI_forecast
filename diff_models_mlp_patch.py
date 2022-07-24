import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlp_mixer import MixerBlock


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI_mlp_patch(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(config)
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.side_dim = config['side_dim']
        self.channels = config['channels']
        self.diffusion_embedding_dim = config['diffusion_embedding_dim']

        self.diffusion_projection = nn.Linear(self.diffusion_embedding_dim, self.channels)
        self.cond_projection = Conv1d_with_init(self.side_dim, 2 * self.channels, 1)
        self.mid_projection = Conv1d_with_init(self.channels, 2 * self.channels, 1)
        self.output_projection = Conv1d_with_init(self.channels, 2 * self.channels, 1)

        # self.config_tf = config['transformer']
        self.mixer_block_1 = MixerBlock(
            tokens_mlp_dim=137*48,
            channels_mlp_dim=512,
            tokens_hidden_dim=512,
            channels_hidden_dim=512,
        )
        self.mixer_block_2 = MixerBlock(
            tokens_mlp_dim=137*48,
            channels_mlp_dim=512,
            tokens_hidden_dim=512,
            channels_hidden_dim=512,
        )
        # self.mixer_block_3 = MixerBlock(
        #     tokens_mlp_dim=17*24,
        #     channels_mlp_dim=1024,
        #     tokens_hidden_dim=1024,
        #     channels_hidden_dim=1024,
        # )
        # self.mixer_block_4 = MixerBlock(
        #     tokens_mlp_dim=17*24,
        #     channels_mlp_dim=1024,
        #     tokens_hidden_dim=1024,
        #     channels_hidden_dim=1024,
        # )
        self.embed = nn.Conv2d(
            self.channels,
            512,
            kernel_size=(1,4),
            stride=(1,4)
            )
        self.embed_transposed = nn.ConvTranspose2d(
            512,
            self.channels,
            kernel_size=(1,4),
            stride=(1,4),
            # output_padding=(1,0)
        )

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb
        y = y.reshape(base_shape)
        y = self.embed(y)
        bs,c,h,w = y.shape
        y = y.view(bs,c,-1).transpose(1,2)
        y = self.mixer_block_1(y)   #.transpose(1,2).reshape(bs,c,h,w)
        y = self.mixer_block_2(y).transpose(1,2).reshape(bs,c,h,w)
        # y = self.mixer_block_3(y).transpose(1,2).reshape(bs,c,h,w)
        # y = self.mixer_block_4(y).transpose(1,2).reshape(bs,c,h,w)
        y = self.embed_transposed(y).reshape(B,channel,-1) # (B,channel,K*L)

        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip