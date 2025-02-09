import torch
from torch.distributions import Categorical
import numpy as np
import time
from tools.state import State
import threading
import os
import signal
import traceback
import datetime
import tools.change_window as change_window
from ImageProcessor import ImageProcessor
from RewardCalculator import RewardCalculator
from ActionExecutor import ActionExecutor
from ReplayBuffer import ReplayBuffer
from PPOTrainer import PPOTrainer
from Agent import Agent

def main():
    change_window.correction_window()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    action_executor = ActionExecutor()
    model_main = Agent(action_dim=len(action_executor.actions)).to(device)
    model_train = Agent(action_dim=len(action_executor.actions)).to(device)
    model_path = ''
    if os.path.exists(model_path):
        model_train.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pre-trained model from", model_path)
    model_main.load_state_dict(model_train.state_dict())
    # 初始学习率
    initial_lr = 0.0001
    # 目标学习率 
    final_lr = 0.0001
    optimizer = torch.optim.AdamW(model_train.parameters(), lr=initial_lr, weight_decay=1e-5, eps=1e-7)
    replay_buffer = ReplayBuffer(capacity=300)
    image_processor = ImageProcessor()
    state = State('guangzhi')
    max_episode = 50
    episode = 0
    save_interval = 50
    exit_event = threading.Event()
    batch_size = 12
    sync_interval = 1
    gamma = 0.9
    max_score = -10000

    # 创建当前时间的文件夹
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('output', current_time)
    training_log_path = os.path.join(save_dir, 'training_log.txt')
    step_log_path = os.path.join(save_dir, 'step_log.txt')
    episode_path = os.path.join(save_dir, 'episode_log.txt')
    replay_buffer_path = os.path.join(save_dir, 'replay_buffer.pkl')
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练线程函数
    def train_thread_fn(replay_buffer, model_train, optimizer, device, save_dir, save_interval, exit_event):
        ppo_trainer = PPOTrainer(model_train, optimizer, gamma=gamma)
        train_count = 0
        while not exit_event.is_set():
            # ppo_trainer.entropy_coef = max(0.08 - (train_count/1000)*0.2, 0.05)
            # 采样数据
            batch = replay_buffer.sample(batch_size)
            if not batch:
                time.sleep(1)
                continue
            # 解析数据
            frames, custom_features, actions, rewards, next_frames, next_custom_features, dones, old_log_probs = zip(*batch)
            # 转换为 tensors 并移动到 device
            with torch.no_grad():
                frames = torch.stack(frames).to(device)
                custom_features = torch.stack(custom_features).to(device)
                actions = torch.tensor(actions, dtype=torch.int64).to(device)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
                next_frames = torch.stack(next_frames).to(device)
                next_custom_features = torch.stack(next_custom_features).to(device)
                dones = torch.tensor(dones, dtype=torch.float32).to(device)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
            # 训练模型
            policy_loss, value_loss, entropy, total_norm, total_loss = ppo_trainer.train(frames, custom_features, actions, rewards, next_frames, next_custom_features, dones, old_log_probs)
            # 保存训练日志
            training_log = {
                'train_count': train_count,
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'total_loss': total_loss,
                'entropy': entropy,
                'total_gradient_norm': total_norm,
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            with open(training_log_path, 'a') as txtfile:
                for key, value in training_log.items():
                    txtfile.write(f"{key}: {value},\t")
                txtfile.write("\n")
            train_count += 1
            # scheduler.step()
            # 保存模型
            if train_count % save_interval == 0:
                model_save_path = os.path.join(save_dir, f'model_{train_count}.pth')
                torch.save(model_train.state_dict(), model_save_path)
            
    # 启动训练线程
    # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=final_lr/initial_lr, total_iters=max_episode)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    train_thread = threading.Thread(target=train_thread_fn,
        args=(replay_buffer, model_train, optimizer, device, save_dir, save_interval, exit_event), daemon=True)
    train_thread.start()
    try:
        while episode < max_episode:
            episode_reward = 0
            HEALTH_COUNT = 10
            print(f'episode: {episode}')
            if episode % sync_interval == 0:
                # replay_buffer.save(replay_buffer_path)
                model_main.load_state_dict(model_train.state_dict())
            
            # 检测环境是否可用
            game_state = 0
            while game_state != 1:
                game_state = state.get_game_state()
                # 处于加载场景 但尚未进入战斗场景 短时间等待
                if state.game_loading_scene and game_state == 2:
                    time.sleep(1)
                    print('等待进入战斗场景')
                    continue
                # 执行复战
                if not state.game_loading_scene and game_state == 2:
                    print('执行复战')
                    state.handle_game_restart()
                    continue
                # 读取内存异常 等待70后 重新进入
                if game_state == 3:
                    print('读取内存异常,执行重新加载')
                    time.sleep(70)
                    state.handle_game_reload()
                    continue
            
            # 战斗场景 开始训练
            # 获取图像帧
            state.step_list = []
            state.game_start_opt()
            player_hp, player_gunshi, player_tili, boss_hp, player_pos, boss_pos = state.get_custom_features()
            
            frames = image_processor.capture_frame()
            frames = frames.unsqueeze(0).to(device)  # 添加batch维度
            distance_to_boss = np.sqrt((player_pos[0] - boss_pos[0])**2 + (player_pos[1] - boss_pos[1])**2)
            reward_calculator = RewardCalculator(player_hp, boss_hp)
            max_player_hp = player_hp
            attack_step = state.read_attack_step()
            print(f'获得状态: player_hp: {player_hp}, player_gunshi:{player_gunshi}, player_tili:{player_tili}, boss_hp: {boss_hp}')
            # 处理特征
            custom_features = model_main.process_custom_features(
                player_hp, player_gunshi, player_tili, boss_hp, distance_to_boss, attack_step,
                player_pos[0], player_pos[1], boss_pos[0], boss_pos[1], boss_pos[2]
            ).unsqueeze(0).to(device)
            done = False
            break_to_episode = False
            while done == False:
                if (player_hp == -1 or player_gunshi == -1 or player_tili == -1 or boss_hp == -1 or attack_step == -1):
                    print('读取内存异常，等待70秒后重新载入')
                    time.sleep(70)
                    state.handle_game_reload()
                    break_to_episode = True
                    break
                game_state = 1 if player_hp > 0 and boss_hp > 0 else 2
                if game_state == 2:
                    break
                # 计算模型延迟
                model_start_time = time.time()
                # 获取动作概率和价值
                with torch.no_grad():
                    action_logits, value = model_main(frames, custom_features)
                    dist = Categorical(logits=action_logits)
                # 动作掩码 调整凤穿花概率，只在棍势480时使用
                action_mask = torch.ones_like(action_logits)
                if player_gunshi >= 480 and distance_to_boss <= 0.05:
                    action_mask[:, [0,1,2,4]] = False
                    action_mask[:, 3] = 5.0
                else:
                    action_mask[:, 3] = False
                # if episode < 10:
                if player_hp >= (max_player_hp / 2) or HEALTH_COUNT <= 0:
                    action_mask[:, 4] = False
                else:
                    action_mask[:, 4] = action_mask[:, 4] * 5
                masked_logits = action_logits + torch.log(action_mask) 
                dist = Categorical(logits=masked_logits)
                print(f'模型决策延迟：{time.time() - model_start_time}')
                action = dist.sample()
                action = action.to(device)
                executed_action = action_executor.execute(action.item(), attack_step)
                # time.sleep(0.2)
                # 获取下一状态的frames和custom_features
                next_frames = image_processor.capture_frame()
                next_player_hp, next_player_gunshi, next_player_tili, next_boss_hp, next_player_pos, next_boss_pos = state.get_custom_features()
                distance_to_boss = np.sqrt((next_player_pos[0] - next_boss_pos[0])**2 + (next_player_pos[1] - next_boss_pos[1])**2)
                next_custom_features = model_main.process_custom_features(next_player_hp, next_player_gunshi, next_player_tili, next_boss_hp, distance_to_boss, attack_step,
                    next_player_pos[0], next_player_pos[1], next_boss_pos[0], next_boss_pos[1], next_boss_pos[2]).to(device)
                attack_step = state.read_attack_step()
                if action.item() == 4: HEALTH_COUNT -= 1
                # 计算奖励
                reward = reward_calculator.calculate_reward(next_player_hp, next_boss_hp,
                    next_player_gunshi, next_player_tili, distance_to_boss, action.item(),
                    action_diversity=1.0, combo_step=attack_step, HEALTH_COUNT=HEALTH_COUNT)
                # 判断done
                done = next_player_hp == 0 or next_boss_hp == 0
                episode_reward += reward
                old_log_probs = dist.log_prob(action).detach()
                # 存储数据到回放缓冲区
                replay_buffer.push(
                    frames.squeeze(0).cpu(), 
                    custom_features.squeeze(0).cpu(), 
                    action.item(), 
                    reward, 
                    next_frames.cpu(), 
                    next_custom_features.cpu(),
                    done,
                    old_log_probs.cpu()
                )
                step_log = {
                    'episode': episode,
                    'reward': reward,
                    'action': old_log_probs.cpu().numpy(),
                    'player_hp': player_hp,
                    'boss_hp': boss_hp,
                    'next_player_hp': next_player_hp,
                    'next_boss_hp': next_boss_hp
                }
                with open(step_log_path, 'a') as txtfile:
                    for key, value in step_log.items():
                        txtfile.write(f"{key}: {value},\t")
                    txtfile.write("\n")
                player_hp = next_player_hp
                player_gunshi = next_player_gunshi
                player_tili = next_player_tili
                boss_hp = next_boss_hp
                player_pos = next_player_pos
                boss_pos = next_boss_pos
                frames = next_frames.unsqueeze(0).to(device)
                custom_features = next_custom_features.unsqueeze(0).to(device)
                print(f"Executed action: {executed_action}")
                print(f"Reward: {reward}")
 
            if (break_to_episode): break
            episode += 1           
            print(f'total reward: {episode_reward}, best_reward: {max_score}')
            episode_log = {
                'episode': episode,
                'reward': episode_reward,
                'win': player_hp > 0
            }
            with open(episode_path, 'a') as txtfile:
                for key, value in episode_log.items():
                    txtfile.write(f"{key}: {value},\t")
                txtfile.write("\n")
            if episode_reward > max_score:
                max_score = episode_reward
                # 保存最佳模型
                torch.save(model_main.state_dict(), best_model_path)
                print(f"New best model saved with score: {max_score} at episode {episode}")
            
    except KeyboardInterrupt:
        traceback.print_exc()
        print("Interrupt received, shutting down...")
        exit_event.set()
        # 等待训练线程退出
        train_thread.join()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda signum, frame: os.kill(os.getpid(), signal.SIGINT))
    main()