from torch.distributions import Categorical
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

# PPO训练器
class PPOTrainer:
    def __init__(self, model, optimizer, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), clip_epsilon=0.05, entropy_coef=0.1, gamma = 0.9):
        self.model = model
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.device = device
        self.gamma = gamma
        
    def train(self, frames, custom_features, actions, rewards, next_frames, next_custom_features, dones, old_log_probs):
        # 计算优势函数
        
        with torch.no_grad():
            _, values = self.model(frames, custom_features)
            _, next_values = self.model(next_frames, next_custom_features)
            td_target, advantages = compute_gae(rewards, values.squeeze(-1), next_values.squeeze(-1), dones)
            # returns = rewards + (1 - dones) * self.gamma * next_values.detach()
            # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # 多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_loss = 0
        train_step = 3
        # print(f"Advantages - Max: {advantages.max()}, Min: {advantages.min()}, Mean: {advantages.mean()}")
        for _ in range(train_step):
            action_logits, current_values = self.model(frames, custom_features)
            dist = Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            # print(f"Ratio - Max: {ratio.max()}, Min: {ratio.min()}, Mean: {ratio.mean()}")
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(current_values.squeeze(), td_target.detach().squeeze())
            loss = policy_loss + value_loss - self.entropy_coef * entropy 
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=5)
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm()}")
            self.optimizer.step()
            # 累积损失
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # 返回训练指标
        return policy_loss.item() / train_step, value_loss.item() / train_step, entropy.item() / train_step, total_norm, total_loss / train_step

def compute_gae(rewards: torch.Tensor,  
                values: torch.Tensor,   
                next_values: torch.Tensor,  
                dones: torch.Tensor,    
                gamma: float = 0.9,
                lambda_: float = 0.95) -> torch.Tensor:
    T = len(rewards)
    td_target = rewards + gamma * next_values * (1 - dones)
    delta = td_target - values
    advantage = torch.zeros_like(delta)
    advantage[-1] = delta[-1]
    for t in reversed(range(T - 1)):
        advantage[t] = delta[t] + gamma * lambda_ * advantage[t + 1] * (1 - dones[t])
    return td_target, advantage
