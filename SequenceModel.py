import torch.nn as nn
import torch

# 序列模型
class SequenceModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        self.time_attention = nn.MultiheadAttention(
            embed_dim=512,  # 256*2
            num_heads=4,
            dropout=0.1
        )
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(features)  # [B, T, 1024]
        # 时间注意力
        attn_out, _ = self.time_attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1]  # 取最后一个时间步
