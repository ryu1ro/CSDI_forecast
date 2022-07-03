import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_sde(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.feature_len = config['diffusion']['feature_len'] #number of features K
        self.seq_len = config['diffusion']['seq_len']
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.emb_dow_dim = config["model"]["dowemb"]
        self.emb_hour_dim = config["model"]["houremb"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + self.emb_dow_dim + self.emb_hour_dim
        self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.feature_len, embedding_dim=self.emb_feature_dim
        )
        #time covariates embedding
        self.embed_layer_dow = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.emb_dow_dim
        )
        self.embed_layer_hour = nn.Embedding(
            num_embeddings=self.seq_len, embedding_dim=self.emb_hour_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, observed_tc, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.feature_len).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        dow_embed = self.embed_layer_dow(observed_tc[:,:,0])    #(B,L,emb_dim)
        hour_embed = self.embed_layer_hour(observed_tc[:,:,1])  #(B,L,emb_dim)
        dow_embed = dow_embed.unsqueeze(2).expand(-1,-1,K,-1)
        hour_embed = hour_embed.unsqueeze(2).expand(-1,-1,K,-1)

        side_info = torch.cat([time_embed, feature_embed, dow_embed, hour_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)


        side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def forward(self, x, t, batch):
        (
            observed_data,
            observed_mask,
            cond_mask,
            observed_tp,
            observed_tc,
        ) = batch
        side_info = self.get_side_info(observed_tp, observed_tc, cond_mask)
        total_input = self.set_input_to_diffmodel(x, observed_data, cond_mask)
        score = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        return score
