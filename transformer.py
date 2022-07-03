import torch
from torch import nn
from torch.nn import functional as f
import math


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv1d(in_channels, key_channels, 1)
        self.queries = nn.Conv1d(in_channels, key_channels, 1)
        self.values = nn.Conv1d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv1d(value_channels, in_channels, 1)

    def forward(self, input_):
        B, channel_dim, L = input_.size()
        keys = self.keys(input_)
        queries = self.queries(input_)
        values = self.values(input_)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = f.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(B, head_value_channels, L)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention #(B, channel, L)

class LinearTranformerEncodeLayer(nn.Module):
    def __init__(self, channels, head_count, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.head_count = head_count

        self.linear1 = nn.Linear(channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, channels)

        # self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.sa_block = EfficientAttention(
            in_channels=channels,
            key_channels=channels,
            head_count=head_count,
            value_channels=channels
        )

    def forward(self, input_):
        x = input_
        x = self.norm1(self.sa_block(x).permute(0,2,1)) #(B,channel,L)->(B,L,channel)
        x = self.norm2(x + self._ff_block(x)).permute(0,2,1) #(B,L,channel)->(B,channel,L)
        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class NystromBlock(nn.Module):
    def __init__(
        self,
        head_dim,
        num_head,
        num_landmarks,
        seq_len,
        kernel_size=5,
        is_padding=False
        ):
        super().__init__()

        self.head_dim = head_dim
        self.num_head = num_head
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len
        self.is_padding = is_padding
        self.padding = (self.seq_len//self.num_landmarks + 1)*self.num_landmarks - self.seq_len

        self.conv = nn.Conv2d(
            in_channels = self.num_head, out_channels = self.num_head,
            kernel_size = (kernel_size, 1), padding = (kernel_size // 2, 0),
            bias = False,
            groups = self.num_head)

    def forward(self, Q, K, V):

        Q = Q / math.sqrt(math.sqrt(self.head_dim))
        K = K / math.sqrt(math.sqrt(self.head_dim))

        # if self.num_landmarks == self.seq_len:
        #     attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
        #     X = torch.matmul(attn, V)
        # else:
        if self.is_padding:
            Q_pad = f.pad(Q, (0,0,0,self.padding), mode='reflect')
            K_pad = f.pad(K, (0,0,0,self.padding), mode='reflect')
            Q_landmarks = Q_pad.reshape(-1, self.num_head, self.num_landmarks, (self.seq_len + self.padding)// self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K_pad.reshape(-1, self.num_head, self.num_landmarks, (self.seq_len + self.padding)// self.num_landmarks, self.head_dim).mean(dim = -2)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

        kernel_1 =f.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel_2 =f.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel_3 =f.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) , dim = -1)
        X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        # if self.use_conv:
        X += self.conv(V)

        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        # if self.init_option == "original":
        #     # This original implementation is more conservative to compute coefficient of Z_0.
        #     V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        # else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def extra_repr(self):
        return f'num_landmarks={self.num_landmarks}, seq_len={self.seq_len}'

class NystromAttention(nn.Module):

    def __init__(
        self,
        in_channels,
        key_channels,
        head_count,
        value_channels,
        num_landmarks,
        seq_len,
        is_padding
        ):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.num_landmarks = num_landmarks
        self.seq_len = seq_len

        self.keys = nn.Conv1d(in_channels, key_channels, 1)
        self.queries = nn.Conv1d(in_channels, key_channels, 1)
        self.values = nn.Conv1d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv1d(value_channels, in_channels, 1)

        self.nystrom_block = NystromBlock(
            head_dim=self.key_channels // self.head_count,
            num_head=self.head_count,
            num_landmarks=self.num_landmarks,
            seq_len=self.seq_len,
            is_padding=is_padding
        )

    def forward(self, input_):
        B, channel_dim, L = input_.size()
        K = self.keys(input_).reshape(B,self.head_count,-1,L).permute(0,1,3,2)
        Q = self.queries(input_).reshape(B,self.head_count,-1,L).permute(0,1,3,2)
        V = self.values(input_).reshape(B,self.head_count,-1,L).permute(0,1,3,2)

        attention = self.nystrom_block(Q,K,V).permute(0,1,3,2).reshape(B,-1,L)
        attention = self.reprojection(attention)
        return attention #(B, channel, L)

class NystromTranformerEncodeLayer(nn.Module):
    def __init__(
        self,
        channels,
        head_count,
        seq_len,
        num_landmarks=64,
        dim_feedforward=64,
        dropout=0.1,
        is_padding=False
        ):
        super().__init__()
        self.head_count = head_count

        self.linear1 = nn.Linear(channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, channels)

        # self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.sa_block = NystromAttention(
            in_channels=channels,
            key_channels=channels,
            head_count=head_count,
            value_channels=channels,
            num_landmarks=num_landmarks,
            seq_len=seq_len,
            is_padding=is_padding
        )

    def forward(self, input_):
        x = input_
        x = self.norm1(self.sa_block(x).permute(0,2,1)) #(B,channel,L)->(B,L,channel)
        x = self.norm2(x + self._ff_block(x)).permute(0,2,1) #(B,L,channel)->(B,channel,L)
        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
