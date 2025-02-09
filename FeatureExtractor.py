import torch.nn as nn
import torch
import torchvision.models as models
from ChannelAttention import ChannelAttention
import torch.nn.functional as F

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, custom_feature_size: int):
        super().__init__()
        self.frame_norm = nn.InstanceNorm2d(256)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(
            *list(resnet.children())[:7],
            ChannelAttention(256)  # 新增通道注意力
        )
        self.custom_feature_processor = nn.Sequential(
            nn.Linear(custom_feature_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Sigmoid()
        )
        
    def forward(self, frames: torch.Tensor, custom_features: torch.Tensor) -> torch.Tensor:
        batch_size = frames.size(0)
        # 处理图像特征 (B, 4, 3, 224, 224) -> (B, 4, 512)
        frame_features = []
        for i in range(4):
            frame = frames[:, i]
            features = self.cnn(frame)
            features = self.frame_norm(features)
            avg_pool = F.adaptive_avg_pool2d(features, (1, 1))
            max_pool = F.adaptive_max_pool2d(features, (1, 1))
            features = 0.6 * avg_pool + 0.4 * max_pool  # 加权组合
            frame_features.append(features.view(batch_size, -1))
        frame_features = torch.stack(frame_features, dim=1)
        # if self.training:  # 只在训练时检查
        #     print(f"\n--- Feature Health Check ---")
        #     print(f"Frame Features - Mean: {frame_features.mean().item():.4f} | Std: {frame_features.std().item():.4f}")
        # 处理自定义特征
        processed_custom = self.custom_feature_processor(custom_features)
        return frame_features, processed_custom
    