import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import gym
from gym import spaces
import pickle
import logging
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 经验回放缓冲区
Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones'])


class ReplayBuffer:
    """多智能体经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
        self.buffer.append(Experience(states, actions, rewards, next_states, dones))

    def sample(self, batch_size: int):
        experiences = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in experiences]

        states = torch.stack([torch.tensor(e.states, dtype=torch.float32) for e in batch])
        actions = torch.stack([torch.tensor(e.actions, dtype=torch.float32) for e in batch])
        rewards = torch.stack([torch.tensor(e.rewards, dtype=torch.float32) for e in batch])
        next_states = torch.stack([torch.tensor(e.next_states, dtype=torch.float32) for e in batch])
        dones = torch.stack([torch.tensor(e.dones, dtype=torch.bool) for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# 图神经网络编码器
class GNNEncoder(nn.Module):
    """基于图注意力网络的空间关系编码器"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GNNEncoder, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)
        return x


# 时间序列编码器
class TemporalEncoder(nn.Module):
    """基于LSTM的时间序列特征编码器"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, cell) = self.lstm(x)
        return hidden[-1]  # 返回最后一层的隐藏状态


# Actor网络
class Actor(nn.Module):
    """MADDPG的Actor网络"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.state_dim = state_dim

        # 简化的特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 决策网络
        self.policy_net = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

    def forward(self, state, edge_index=None, temporal_seq=None):
        # 确保输入维度正确
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # 特征提取
        features = self.feature_extractor(state)

        # 策略输出
        action = self.policy_net(features)

        return self.max_action * action


# Critic网络
class Critic(nn.Module):
    """MADDPG的Critic网络"""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        super(Critic, self).__init__()

        # 全局状态和动作维度
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents

        self.fc1 = nn.Linear(total_state_dim + total_action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, states, actions):
        # states: (batch_size, num_agents, state_dim)
        # actions: (batch_size, num_agents, action_dim)

        # 展平为全局状态和动作
        states_flat = states.view(states.size(0), -1)
        actions_flat = actions.view(actions.size(0), -1)

        x = torch.cat([states_flat, actions_flat], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.fc4(x)

        return q_value


# MADDPG Agent
class MADDPGAgent:
    """多智能体深度确定性策略梯度智能体"""

    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 num_agents: int, max_action: float = 1.0,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 tau: float = 0.005, gamma: float = 0.99):

        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma

        # Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic网络
        self.critic = Critic(state_dim, action_dim, num_agents).to(device)
        self.critic_target = Critic(state_dim, action_dim, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # 初始化目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # 噪声
        self.noise_scale = 0.1

    def select_action(self, state, add_noise=True):
        """选择动作"""
        # 确保state是正确的形状
        if isinstance(state, list):
            state = np.array(state)

        state = torch.FloatTensor(state).to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()
            if len(action.shape) > 1:
                action = action[0]

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def update(self, replay_buffer: ReplayBuffer, batch_size: int,
               other_agents: List['MADDPGAgent']):
        """更新网络参数"""
        if len(replay_buffer) < batch_size:
            return

        # 采样经验
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # 获取当前智能体的奖励
        agent_rewards = rewards[:, self.agent_id].unsqueeze(1)
        agent_dones = dones[:, self.agent_id].unsqueeze(1)

        # 计算目标Q值
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate([self] + other_agents):
                if i == self.agent_id:
                    continue
                next_action = agent.actor_target(next_states[:, i])
                next_actions.append(next_action)
            next_actions.insert(self.agent_id,
                                self.actor_target(next_states[:, self.agent_id]))
            next_actions = torch.stack(next_actions, dim=1)

            target_q = self.critic_target(next_states, next_actions)
            target_q = agent_rewards + (self.gamma * target_q * (~agent_dones).float())

        # 更新Critic
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor
        actions_pred = actions.clone()
        agent_state = states[:, self.agent_id]
        predicted_action = self.actor(agent_state)
        actions_pred[:, self.agent_id] = predicted_action

        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def hard_update(self, target, source):
        """硬更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


# PyGNOME环境封装
class OilSpillEnvironment:
    """基于PyGNOME的溢油扩散环境"""

    def __init__(self, config: dict):
        self.config = config
        self.num_ships = config.get('num_ships', 5)
        self.map_size = config.get('map_size', (100, 100))
        self.max_steps = config.get('max_steps', 200)

        # 状态和动作空间定义
        self.state_dim = 128  # 包含油污分布、船只状态、环境参数
        self.action_dim = 4  # 船只移动方向和速度

        # 环境状态
        self.current_step = 0
        self.oil_spill_locations = []
        self.ship_positions = []
        self.wind_data = None
        self.current_data = None

        self._initialize_environment()

    def _initialize_environment(self):
        """初始化环境状态"""
        # 初始化船只位置
        self.ship_positions = [
            np.random.uniform(0, self.map_size[0], 2)
            for _ in range(self.num_ships)
        ]

        # 初始化溢油位置（模拟）
        spill_center = (self.map_size[0] // 2, self.map_size[1] // 2)
        self.oil_spill_locations = [spill_center]

        # 模拟风流场数据
        self.wind_data = np.random.normal(0, 5, (self.map_size[0], self.map_size[1], 2))
        self.current_data = np.random.normal(0, 2, (self.map_size[0], self.map_size[1], 2))

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self._initialize_environment()
        return self._get_observations()

    def step(self, actions):
        """环境步进"""
        self.current_step += 1

        # 更新船只位置
        for i, action in enumerate(actions):
            # action: [dx, dy, speed, operation_type]
            dx, dy, speed, _ = action
            self.ship_positions[i][0] += dx * speed
            self.ship_positions[i][1] += dy * speed

            # 边界约束
            self.ship_positions[i] = np.clip(
                self.ship_positions[i],
                [0, 0],
                [self.map_size[0], self.map_size[1]]
            )

        # 模拟油污扩散（简化版，实际应调用PyGNOME）
        self._simulate_oil_spread()

        # 计算奖励
        rewards = self._calculate_rewards(actions)

        # 检查终止条件
        done = self.current_step >= self.max_steps or self._check_cleanup_complete()

        observations = self._get_observations()

        return observations, rewards, [done] * self.num_ships, {}

    def _simulate_oil_spread(self):
        """模拟油污扩散过程"""
        # 这里应该集成PyGNOME进行真实的油污扩散模拟
        # 当前为简化模拟
        new_locations = []
        for loc in self.oil_spill_locations:
            # 基于风流场数据模拟扩散
            wind_effect = self.wind_data[int(loc[0]), int(loc[1])]
            current_effect = self.current_data[int(loc[0]), int(loc[1])]

            new_x = loc[0] + 0.1 * (wind_effect[0] + current_effect[0])
            new_y = loc[1] + 0.1 * (wind_effect[1] + current_effect[1])

            new_locations.append((new_x, new_y))

        self.oil_spill_locations.extend(new_locations)

    def _get_observations(self):
        """获取观测状态"""
        observations = []

        for i in range(self.num_ships):
            obs = []

            # 船只自身状态 (x, y)
            ship_state = list(self.ship_positions[i])
            obs.extend(ship_state)

            # 相对于其他船只的位置
            for j in range(self.num_ships):
                if i != j:
                    rel_pos = np.array(self.ship_positions[j]) - np.array(self.ship_positions[i])
                    obs.extend(rel_pos.tolist())
                else:
                    obs.extend([0.0, 0.0])  # 自己相对于自己的位置为0

            # 最近的油污点距离和方向
            if self.oil_spill_locations:
                nearest_oil_distances = []
                nearest_oil_directions = []

                for oil_loc in self.oil_spill_locations[:5]:  # 考虑最近的5个油污点
                    diff = np.array(oil_loc) - np.array(self.ship_positions[i])
                    dist = np.linalg.norm(diff)
                    direction = diff / (dist + 1e-8)  # 避免除零

                    nearest_oil_distances.append(dist)
                    nearest_oil_directions.extend(direction.tolist())

                # 填充至固定长度
                while len(nearest_oil_distances) < 5:
                    nearest_oil_distances.append(100.0)  # 大的距离值
                    nearest_oil_directions.extend([0.0, 0.0])

                obs.extend(nearest_oil_distances)
                obs.extend(nearest_oil_directions)
            else:
                # 没有油污时的默认值
                obs.extend([100.0] * 5)  # 5个距离
                obs.extend([0.0] * 10)  # 5个方向向量

            # 环境信息
            obs.append(self.current_step / self.max_steps)  # 时间进度
            obs.append(len(self.oil_spill_locations))  # 油污点数量

            # 确保观测维度正确
            expected_dim = 2 + (self.num_ships * 2) + 5 + 10 + 2  # 27 for 5 ships
            current_dim = len(obs)

            if current_dim < self.state_dim:
                obs.extend([0.0] * (self.state_dim - current_dim))
            elif current_dim > self.state_dim:
                obs = obs[:self.state_dim]

            observations.append(obs)

        return np.array(observations)

    def _calculate_rewards(self, actions):
        """计算奖励函数"""
        rewards = []

        for i in range(self.num_ships):
            reward = 0.0

            # 接近油污的奖励
            min_oil_dist = min([
                np.linalg.norm(np.array(oil_loc) - np.array(self.ship_positions[i]))
                for oil_loc in self.oil_spill_locations
            ])
            reward += max(0, 20 - min_oil_dist) * 0.1

            # 协同奖励（避免船只聚集）
            cooperation_reward = 0
            for j in range(self.num_ships):
                if i != j:
                    dist = np.linalg.norm(
                        np.array(self.ship_positions[i]) - np.array(self.ship_positions[j])
                    )
                    if 5 < dist < 15:  # 理想协作距离
                        cooperation_reward += 0.1
                    elif dist < 5:  # 过于接近，惩罚
                        cooperation_reward -= 0.2

            reward += cooperation_reward

            # 清理效率奖励
            cleanup_reward = actions[i][3] * 0.5  # 基于清理操作强度
            reward += cleanup_reward

            rewards.append(reward)

        return rewards

    def _check_cleanup_complete(self):
        """检查清理是否完成"""
        # 简化的完成条件
        return len(self.oil_spill_locations) < 5


# 训练主函数
def train_maddpg(config: dict):
    """MADDPG训练主函数"""

    # 创建环境
    env = OilSpillEnvironment(config)

    # 创建智能体
    agents = []
    for i in range(env.num_ships):
        agent = MADDPGAgent(
            agent_id=i,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            num_agents=env.num_ships,
            max_action=1.0
        )
        agents.append(agent)

    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(config.get('buffer_size', 100000))

    # 训练参数
    total_episodes = config.get('total_episodes', 1000)
    batch_size = config.get('batch_size', 64)
    update_freq = config.get('update_freq', 100)

    # 记录训练过程
    episode_rewards = []
    critic_losses = []
    actor_losses = []

    print("开始MADDPG训练...")

    for episode in range(total_episodes):
        states = env.reset()
        episode_reward = 0
        step_count = 0

        while True:
            # 选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[i])
                actions.append(action)

            # 执行动作
            next_states, rewards, dones, _ = env.step(actions)

            # 存储经验
            replay_buffer.push(states, actions, rewards, next_states, dones)

            states = next_states
            episode_reward += np.sum(rewards)
            step_count += 1

            # 更新网络
            if len(replay_buffer) > batch_size and step_count % update_freq == 0:
                for i, agent in enumerate(agents):
                    other_agents = [agents[j] for j in range(len(agents)) if j != i]
                    c_loss, a_loss = agent.update(replay_buffer, batch_size, other_agents)

                    if c_loss is not None:
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)

            # 检查终止条件
            if all(dones):
                break

        episode_rewards.append(episode_reward)

        # 打印训练进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")

            # 保存模型
            if episode % 200 == 0:
                save_models(agents, episode)

    # 绘制训练曲线
    plot_training_curves(episode_rewards, critic_losses, actor_losses)

    return agents


def save_models(agents: List[MADDPGAgent], episode: int):
    """保存模型"""
    save_dir = f"models/episode_{episode}"
    os.makedirs(save_dir, exist_ok=True)

    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(),
                   f"{save_dir}/agent_{i}_actor.pth")
        torch.save(agent.critic.state_dict(),
                   f"{save_dir}/agent_{i}_critic.pth")

    print(f"模型已保存至 {save_dir}")


def plot_training_curves(episode_rewards, critic_losses, actor_losses):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 回合奖励
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')

    # Critic损失
    if critic_losses:
        axes[1].plot(critic_losses)
        axes[1].set_title('Critic Loss')
        axes[1].set_xlabel('Update Step')
        axes[1].set_ylabel('Loss')

    # Actor损失
    if actor_losses:
        axes[2].plot(actor_losses)
        axes[2].set_title('Actor Loss')
        axes[2].set_xlabel('Update Step')
        axes[2].set_ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


# 主训练配置和执行
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 训练配置
    config = {
        'num_ships': 5,
        'map_size': (100, 100),
        'max_steps': 200,
        'total_episodes': 2000,
        'batch_size': 64,
        'buffer_size': 100000,
        'update_freq': 100,
        'learning_rate_actor': 1e-4,
        'learning_rate_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005
    }

    print("海上溢油应急调度MADDPG训练系统")
    print("=" * 50)
    print(f"配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    # 开始训练
    trained_agents = train_maddpg(config)

    print("训练完成！")
    print("模型已保存，训练曲线已生成。")