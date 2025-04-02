import time
import numpy as np
# 奖励计算器
class RewardCalculator:
    def __init__(self, player_hp, boss_hp):
        self.prev_player_hp = player_hp
        self.prev_boss_hp = boss_hp
        self.max_boss_hp = boss_hp
        self.max_player_hp = player_hp
        self.combo_count = 0
        self.step_count = 0
        self.attack_interval = 3
        self.last_attack_time = time.time()
        
        # 攻击激励增强系数
        self.DAMAGE_REWARD = 7
        self.COMBO_MULTIPLIER = 0.8  # 连击倍率
        self.HEALTH_PENALTY = 20  # 惩罚强度
        self.HEALTH_PENALTY_ON_ATTACK = 5  # 惩罚强度
        self.HEAL_REWARD_RATIO = 20.0  # 治疗奖励
        self.DISTANCE_BONUS = 1  # 距离奖励
        self.PENALTY_REWARD = 10
        self.WIN_REWARD = 50.0
        self.LOSE_PENALTY = -50.0  # 失败惩罚
        self.TIME_FACTOR = 0.01  # 时间衰减

    def calculate_reward(self, player_hp: float, boss_hp: float, player_gunshi, player_tili,
                        distance_to_boss: float, action_idx: int, action_diversity, combo_step, HEALTH_COUNT) -> float:
        reward = 0
        self.step_count += 1
        self.combo_count = combo_step
        # === 攻击激励模块 ===
        boss_damage = self.prev_boss_hp - boss_hp
        if boss_damage > 0:
            time_since_last_attack = time.time() - self.last_attack_time
            # 连击系数：使用tanh平滑处理
            combo_bonus = (2.5 * (1 + np.tanh(self.combo_count / 3)))
            # 时间惩罚：鼓励保持攻击频率
            time_penalty = 0.8 ** (time_since_last_attack / self.attack_interval)
            reward += self.DAMAGE_REWARD * combo_bonus * time_penalty
            self.last_attack_time = time.time()

        # === 治疗奖励 ===
        if action_idx == 4:  # 治疗动作
            current_ratio = player_hp / self.max_player_hp
            if HEALTH_COUNT <= 0: reward -= 20
            if current_ratio <= 0.5:
                heal_bonus = (1 - current_ratio) * self.HEAL_REWARD_RATIO
                reward += heal_bonus
            else:
                heal_bonus = current_ratio * self.HEAL_REWARD_RATIO
                reward -= heal_bonus
        # === 技能奖励 ===
        if action_idx == 3:
            reward += 30 if player_gunshi == 480 else -30

        # === 距离奖励优化 ===
        print(f'distance_to_boss: {distance_to_boss}')
        if action_idx == 2 and distance_to_boss < 0.025:
            reward += self.DISTANCE_BONUS * (1 + self.combo_count*0.1)  # 连击加成距离奖励
        # === 受伤惩罚优化 ===
        player_damage = self.prev_player_hp - player_hp
        # player_damage > 20 防止dot伤害
        if player_damage > 20:
            reward -= self.HEALTH_PENALTY
        # === 终局奖励 ===
        if boss_hp <= 0:
            reward += self.WIN_REWARD 
        elif player_hp <= 0:
            reward += self.LOSE_PENALTY

        reward -= self.TIME_FACTOR * self.step_count
        print(f'prev_player_hp: {self.prev_player_hp}, player_hp: {player_hp}, prev_boss_hp: {self.prev_boss_hp}, boss_hp: {boss_hp}')
        self.prev_player_hp = player_hp
        self.prev_boss_hp = boss_hp
        return reward
