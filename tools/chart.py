import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 读取并解析日志文件
def parse_training_log(file_path):
    pattern = re.compile(
        r"train_count: (\d+),\s+"
        r"policy_loss: ([\d\-.e]+),\s+"
        r"value_loss: ([\d\-.e]+),\s+"
        r"total_loss: ([\d\-.e]+),\s+"
        r"entropy: ([\d\-.e]+),\s+"
        r"total_gradient_norm: ([\d\-.e]+),\s+"
        r"learning_rate: ([\d\-.e]+)"
    )
    
    data = {"train_count": [], "policy_loss": [], "value_loss": [], "total_loss": [], "entropy": [], "total_gradient_norm": [], "learning_rate": []}
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                data["train_count"].append(int(match.group(1)))
                data["policy_loss"].append(float(match.group(2)))
                data["value_loss"].append(float(match.group(3)))
                data["total_loss"].append(float(match.group(4)))
                data["entropy"].append(float(match.group(5)))
                data["total_gradient_norm"].append(float(match.group(6)))
                data["learning_rate"].append(float(match.group(7)))
    return data

# 平滑曲线（滑动平均）
def smooth_curve(values, window_size=1000):
    if len(values) < window_size:
        return values  # 数据量不足时不平滑
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode='valid')

# 绘制训练曲线
def plot_training_metrics(data, dir):
    smoothed_total_loss = smooth_curve(data["total_loss"])
    smoothed_policy_loss = smooth_curve(data["policy_loss"])
    smoothed_value_loss = smooth_curve(data["value_loss"])
    smoothed_entropy = smooth_curve(data["entropy"])
    smoothed_learning_rate = smooth_curve(data["learning_rate"])
    
    plt.figure(figsize=(12, 8))
    
    # 总损失
    plt.subplot(2, 2, 1)
    plt.plot(data["train_count"][:len(smoothed_total_loss)], smoothed_total_loss, label="Total Loss", color="red")
    plt.xlabel("Training Step")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.grid()
    
    # Policy Loss & Value Loss
    plt.subplot(2, 2, 2)
    plt.plot(data["train_count"][:len(smoothed_policy_loss)], smoothed_policy_loss, label="Policy Loss", color="blue")
    plt.plot(data["train_count"][:len(smoothed_value_loss)], smoothed_value_loss, label="Value Loss", color="green")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    
    # Entropy
    plt.subplot(2, 2, 3)
    plt.plot(data["train_count"][:len(smoothed_entropy)], smoothed_entropy, label="Entropy", color="purple")
    plt.xlabel("Training Step")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid()
    
    # 学习率变化
    plt.subplot(2, 2, 4)
    plt.plot(data["train_count"][:len(smoothed_learning_rate)], smoothed_learning_rate, label="Learning Rate", color="orange")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter("%.6f"))
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(f'{dir}/smoothed_learning_rate.png', dpi=300, bbox_inches='tight')
    # plt.show()


def parse_step_log(file_path):
    rewards = []
    player_hp = []
    boss_hp = []
    episode_rewards = {}  # 记录每个 episode 的总 reward

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            try:
                episode = int(parts[0].split(":")[1])
                reward = float(parts[1].split(":")[1])
                player_hp.append(float(parts[3].split(":")[1]))
                boss_hp.append(float(parts[4].split(":")[1]))
                rewards.append(reward)

                # if episode not in episode_rewards:
                #     episode_rewards[episode] = 0
                # episode_rewards[episode] += reward  # 累加 episode 内所有 reward
            except ValueError:
                continue  # 遇到格式错误时跳过该行

    return rewards, player_hp, boss_hp, episode_rewards

def parse_episode_log(file_path):
    rewards = []
    wins = []
    episodes = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(",")
            episodes.append(int(parts[0].split(":")[1]))
            rewards.append(float(parts[1].split(":")[1]))
            wins.append(bool(parts[2].split(":")[1].strip() == "True"))
    return episodes, rewards, wins

def plot_episode_log(episode, rewards, wins):
    if not rewards:
        print("日志文件为空或解析失败")
        return
    # plt.figure(figsize=(12, 8))
    
    # plt.subplot(2, 1, 1)
    # # plt.plot(episode, rewards, label="Reward", color="red")
    # plt.plot(episode, rewards, label="Reward", color="blue")
    # # plt.plot(episode, wins, label="wins", color="green")
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.title("Episode Reward Trend")
    # plt.legend()
    # plt.grid()



    # plt.subplot(2, 1, 2)
    # plt.plot(episode, wins, label="Reward", color="red")
    # plt.xlabel("Episode")
    # plt.ylabel("Win")
    # plt.title("Episode Win Trend")
    # plt.legend()
    # plt.grid()

    fig, ax1 = plt.subplots(figsize=(12, 8))
# 第一个y轴：奖励趋势
    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward', color=color, fontsize=14, weight='bold')
    ax1.plot(episode, rewards, label="Reward", color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    # 创建第二个y轴：胜利趋势
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Win', color=color)
    ax2.plot(episode, wins, label="Win", color=color)
    ax2.set_ylim(0, 3)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    # 设置标题
    plt.title("Episode Reward and Win Trends")
    plt.tight_layout()
    plt.savefig(f'{dir}/episode.png', dpi=300, bbox_inches='tight')

def plot_step_log(rewards, player_hp, boss_hp, episode_rewards, dir):
    if not rewards:
        print("日志文件为空或解析失败")
        return

    steps = np.arange(len(rewards))
    episodes = list(episode_rewards.keys())
    episode_rewards_values = list(episode_rewards.values())

    # 创建绘图窗口
    plt.figure(figsize=(12, 8))
    
    # Reward 变化曲线
    plt.subplot(2, 2, 1)
    plt.plot(steps, rewards, label="Reward", color="red")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Step-wise Reward Trend")
    plt.legend()
    plt.grid()

    # Player HP 变化曲线
    plt.subplot(2, 2, 2)
    plt.plot(steps, player_hp, label="Player HP", color="blue")
    plt.xlabel("Step")
    plt.ylabel("Player HP")
    plt.title("Player HP Trend")
    plt.legend()
    plt.grid()

    # Boss HP 变化曲线
    plt.subplot(2, 2, 3)
    plt.plot(steps, boss_hp, label="Boss HP", color="green")
    plt.xlabel("Step")
    plt.ylabel("Boss HP")
    plt.title("Boss HP Trend")
    plt.legend()
    plt.grid()

    # 每个 Episode 的总 Reward 变化曲线
    plt.subplot(2, 2, 4)
    plt.plot(episodes, episode_rewards_values, label="Total Episode Reward", color="orange", marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode-wise Total Reward Trend")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'{dir}/reward.png', dpi=300, bbox_inches='tight')
    # plt.show()


dir = './20250209_093623'
log_file = f"{dir}/training_log.txt"  
data = parse_training_log(log_file)
plot_training_metrics(data, dir)

log_file = f"{dir}/step_log.txt"  
rewards, player_hp, boss_hp, episode_rewards = parse_step_log(log_file)
plot_step_log(rewards, player_hp, boss_hp, episode_rewards, dir)

log_file = f"{dir}/episode_log.txt"  
episodes, rewards, wins = parse_episode_log(log_file)
plot_episode_log(episodes, rewards, wins)