import torch.nn as nn
import numpy as np
import torch
from typing import Tuple
from PPO import PPO
from FeatureExtractor import FeatureExtractor
from SequenceModel import SequenceModel

# 游戏AI
class Agent(nn.Module):
    def __init__(self, action_dim: int, custom_feature_size: int = 11):
        super().__init__()
        self.feature_extractor = FeatureExtractor(custom_feature_size)
        self.sequence_model = SequenceModel(input_size=256)
        self.ppo = PPO(state_dim=512 + 32, action_dim=action_dim)   # LSTM输出512 + 自定义特征32

    def process_custom_features(self, player_hp: float, player_gunshi: float, player_tili: float,
                              boss_hp: float, distance_to_boss: float, player_state,
                              player_x, player_y, boss_x, boss_y, boss_z) -> torch.Tensor:
        features = [
            np.clip(player_hp / 815, 0.0, 1.0),
            np.clip(player_gunshi / 480, 0.0, 1.0),
            np.clip(player_tili / 470, 0.0, 1.0),
            np.clip(boss_hp / 18628, 0.0, 1.0),
            np.clip(distance_to_boss / 0.1, 0.0, 1.0),
            np.clip(player_state/ 7, 0.0, 1.0),
            np.clip((player_x - 6.9) / (7.1 - 6.9), 0.0, 1.0),
            np.clip((player_y - -6.7)/ (-6.4 - -6.7), 0.0, 1.0),
            np.clip((boss_x - 6.9)   / ( 7.1 - 6.9), 0.0, 1.0),
            np.clip((boss_y - -6.7)  / (-6.4 - -6.7), 0.0, 1.0),
            np.clip((boss_z - 5.5)   / (5.7 - 5.5), 0.0, 1.0),
        ]
        return torch.tensor(features, dtype=torch.float32)
        
    def forward(self, frames: torch.Tensor, custom_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 特征提取
        frame_features, processed_custom = self.feature_extractor(frames, custom_features)
        # 序列建模
        sequence_features = self.sequence_model(frame_features)
        # 合并特征
        combined_features = torch.cat([sequence_features, processed_custom], dim=-1)
        # PPO决策
        return self.ppo(combined_features)