import torch.nn as nn
import torch
from typing import Tuple
from ResidualBlock import ResidualBlock

# PPO模型
class PPO(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.base_fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.1),
        )
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(1024) for _ in range(3)
        ])
        # 双critic设计
        self.actor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.base_fc(state)
        for block in self.res_blocks:
            x = block(x)
        return self.actor(x), self.critic(x)
