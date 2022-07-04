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


class NysAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        in_dim=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        num_landmarks=64,
        kernel_size=0,
        init_option = "exact"
        ):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        head_dim = dim // num_heads
        self.scale = head_dim ** 0.5
        self.landmarks = num_landmarks
        self.kernel_size = kernel_size
        self.init_option = init_option

        self.qkv = nn.Linear(dim, in_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_dim, in_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels = self.num_heads, out_channels = self.num_heads,
                kernel_size = (self.kernel_size, 1), padding = (self.kernel_size // 2, 0),
                bias = False,
                groups = self.num_heads)

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim).permute(2, 0, 3, 1, 4)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.in_dim//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q /= self.scale

        keys_head_dim = k.size(-1)
        segs = N // self.landmarks
        if (N % self.landmarks == 0):
            keys_landmarks = k.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
            queries_landmarks = q.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
        else:
            num_k = (segs + 1) * self.landmarks - N
            keys_landmarks_f = k[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            keys_landmarks_l = k[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

            queries_landmarks_f = q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            queries_landmarks_l = q[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)

        kernel_1 = torch.nn.functional.softmax(torch.matmul(q, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel_2 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel_3 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, k.transpose(-1, -2)), dim = -1)
        x = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, v))

        if self.kernel_size > 0:
            x += self.conv(v)

        x = x.transpose(1, 2).reshape(B, N, self.in_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        # x = v.squeeze(1) + x
        return x

class NystromformerEncodeLayer(nn.Module):
    def __init__(self, channels, head_count, num_landmarks, dim_feedforward=64, dropout=0.1):
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

        self.sa_block = NysAttention(
            dim=channels,
            num_heads=head_count,
            in_dim=channels,
            num_landmarks=num_landmarks,
            kernel_size=3
        )

    def forward(self, input_):
        x = input_
        x = self.norm1(self.sa_block(x.permute(0,2,1))) #(B,channel,L)->(B,L,channel)
        x = self.norm2(x + self._ff_block(x)).permute(0,2,1) #(B,L,channel)->(B,channel,L)
        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)