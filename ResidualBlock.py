import torch.nn as nn
import torch
import torch.nn.functional as F

# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.block = nn.Sequential(  
            nn.Linear(features, features),
            nn.LayerNorm(features),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(features, features),
            nn.LayerNorm(features),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.block(x), 0.1)
