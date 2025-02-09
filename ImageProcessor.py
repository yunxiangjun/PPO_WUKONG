
import tools.screen as screen
from collections import deque
import time
import torch

# 图像处理器
class ImageProcessor:
    def __init__(self, queue_size: int = 4):
        self.frame_queue = deque(maxlen=queue_size)
        self.capture_interval = 0.05  # 50ms
        screen.init()
        time.sleep(3)

    def capture_frame(self) -> bool:
        self.frame_queue.clear()
        while self.frame_queue.__len__() < 4:
            frame = screen.get_screen()  # 假设返回的形状是 [height, width, channels]
            # 将图像数据转换为 torch.Tensor，并调整形状
            frame_tensor = torch.tensor(frame, dtype=torch.float32)  # 转换为张量
            frame_tensor = frame_tensor.permute(2, 0, 1)  # 将形状从 [H, W, C] 调整为 [C, H, W]
             # 添加ImageNet标准化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
            frame_tensor = (frame_tensor/255 - mean) / std
            self.frame_queue.append(frame_tensor)
        return torch.stack(list(self.frame_queue))