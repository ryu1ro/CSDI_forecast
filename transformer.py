import torch
from torch import nn
from torch.nn import functional as f


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
