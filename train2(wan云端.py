import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import StepLR
import gym
from gym import spaces
import pickle
import logging
import wandb
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import os
from multiprocessing import Process, Pipe
import imageio
import numpy as np

# 假设 PyGNOME 库可用
try:
    from gnome.model import Model
    from gnome.spill import Spill
    from gnome.movers import WindMover, RandomMover
    from gnome.environment import Wind
except ImportError:
    print("PyGNOME 未安装，将使用简化模拟。")

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

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        experiences = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in experiences]

        states = torch.stack([torch.tensor(e.states, dtype=torch.float32) for e in batch])
        actions = torch.stack([torch.tensor(e.actions, dtype=torch.float32) for e in batch])
        rewards = torch.stack([torch.tensor(e.rewards, dtype=torch.float32) for e in batch])
        next_states = torch.stack([torch.tensor(e.next_states, dtype=torch.float32) for e in batch])
        dones = torch.stack([torch.tensor(e.dones, dtype=torch.bool) for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

# 图神经网络编码器（含注意力机制）
class GNNEncoder(nn.Module):
    """基于图注意力网络的空间关系编码器"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(GNNEncoder, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, dropout=0.1)
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=hidden_dim * 4, nhead=4, dropout=0.1),
            num_layers=1
        )
        self.gat2 = GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.gat1(x, edge_index))
        x = x.view(-1, x.size(-1)).unsqueeze(1)  # (num_nodes, 1, hidden_dim * 4)
        x = self.transformer(x).squeeze(1)
        x = self.gat2(x, edge_index)
        return x

# 时间序列编码器
class TemporalEncoder(nn.Module):
    """基于LSTM的时间序列特征编码器"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super(TemporalEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)
        return hidden[-1]  # 返回最后一层的隐藏状态

# Actor 网络
class Actor(nn.Module):
    """MADDPG的Actor网络，集成GNN和时间编码器"""
    def __init__(self, state_dim: int, action_dim: int, max_action: float, num_agents: int):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.gnn = GNNEncoder(input_dim=state_dim // num_agents, hidden_dim=64, output_dim=64)
        self.temporal = TemporalEncoder(input_dim=64, hidden_dim=64, num_layers=2)
        self.policy_net = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor, edge_index: torch.Tensor, temporal_seq: torch.Tensor) -> torch.Tensor:
        # state: (batch_size, num_agents, state_dim_per_agent)
        # edge_index: 图连接关系
        # temporal_seq: (batch_size, seq_len, feature_dim)
        gnn_out = self.gnn(state.view(-1, state.size(-1)), edge_index)
        gnn_out = gnn_out.view(state.size(0), state.size(1), -1)  # (batch, num_agents, gnn_out_dim)
        temporal_out = self.temporal(gnn_out)  # (batch, num_agents, hidden_dim)
        action = self.policy_net(temporal_out)
        return self.max_action * action

# Critic 网络
class Critic(nn.Module):
    """MADDPG的Critic网络，优化为分解结构"""
    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        super(Critic, self).__init__()
        self.local_critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.global_critic = nn.Sequential(
            nn.Linear(64 * num_agents, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        batch_size = states.size(0)
        local_qs = []
        for i in range(states.size(1)):
            x = torch.cat([states[:, i], actions[:, i]], dim=-1)
            local_qs.append(self.local_critic(x))
        global_input = torch.cat(local_qs, dim=-1)
        q_value = self.global_critic(global_input)
        return q_value

# MADDPG Agent
class MADDPGAgent:
    """多智能体深度确定性策略梯度智能体"""
    def __init__(self, agent_id: int, state_dim: int, action_dim: Dict[str, int],
                 num_agents: int, agent_type: str = 'default', max_action: float = 1.0,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3, tau: float = 0.005,
                 gamma: float = 0.99):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.action_dim = action_dim[agent_type]
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma

        # Actor 网络
        self.actor = Actor(state_dim, self.action_dim, max_action, num_agents).to(device)
        self.actor_target = Actor(state_dim, self.action_dim, max_action, num_agents).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.actor_scheduler = StepLR(self.actor_optimizer, step_size=500, gamma=0.5)

        # Critic 网络
        self.critic = Critic(state_dim, self.action_dim, num_agents).to(device)
        self.critic_target = Critic(state_dim, self.action_dim, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.critic_scheduler = StepLR(self.critic_optimizer, step_size=500, gamma=0.5)

        # 初始化目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # 噪声
        self.noise_scale = 0.1
        self.noise_decay = 0.995
        self.noise_min = 0.01

    def select_action(self, state: np.ndarray, edge_index: torch.Tensor,
                      temporal_seq: torch.Tensor, add_noise: bool = True) -> np.ndarray:
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        temporal_seq = torch.FloatTensor(temporal_seq).to(device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state, edge_index, temporal_seq).cpu().squeeze(0).numpy()
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        self.noise_scale = max(self.noise_min, self.noise_scale * self.noise_decay)
        return action

    def update(self, replay_buffer: ReplayBuffer, batch_size: int,
               other_agents: List['MADDPGAgent'], edge_index: torch.Tensor) -> Tuple[float, float]:
        if len(replay_buffer) < batch_size:
            return None, None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = [x.to(device) for x in [states, actions, rewards, next_states, dones]]

        agent_rewards = rewards[:, self.agent_id].unsqueeze(1)
        agent_dones = dones[:, self.agent_id].unsqueeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate([self] + other_agents):
                if i == self.agent_id:
                    continue
                next_action = agent.actor_target(next_states[:, i].unsqueeze(1), edge_index, next_states[:, i])
                next_actions.append(next_action)
            next_actions.insert(self.agent_id, self.actor_target(next_states[:, self.agent_id].unsqueeze(1), edge_index, next_states[:, self.agent_id]))
            next_actions = torch.stack(next_actions, dim=1)
            target_q = self.critic_target(next_states, next_actions)
            target_q = agent_rewards + (self.gamma * target_q * (~agent_dones).float())

        # 更新 Critic
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        self.critic_scheduler.step()

        # 更新 Actor
        actions_pred = actions.clone()
        agent_state = states[:, self.agent_id].unsqueeze(1)
        predicted_action = self.actor(agent_state, edge_index, states[:, self.agent_id])
        actions_pred[:, self.agent_id] = predicted_action
        actor_loss = -self.critic(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        self.actor_scheduler.step()

        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        return critic_loss.item(), actor_loss.item()

    def soft_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target: nn.Module, source: nn.Module):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# 并行环境
def worker(conn, env_config):
    env = OilSpillEnvironment(env_config)
    while True:
        cmd, data = conn.recv()
        if cmd == 'step':
            obs, reward, done, info = env.step(data)
            conn.send((obs, reward, done, info))
        elif cmd == 'reset':
            obs = env.reset()
            conn.send(obs)
        elif cmd == 'render':
            env.visualize_environment(data)
            conn.send(None)
        elif cmd == 'close':
            conn.close()
            break

class ParallelEnv:
    """并行环境实现"""
    def __init__(self, num_envs: int, config: Dict):
        self.num_envs = num_envs
        self.conns = []
        self.processes = []
        for _ in range(num_envs):
            parent_conn, child_conn = Pipe()
            p = Process(target=worker, args=(child_conn, config))
            p.start()
            self.conns.append(parent_conn)
            self.processes.append(p)

    def reset(self) -> np.ndarray:
        for conn in self.conns:
            conn.send(('reset', None))
        return np.array([conn.recv() for conn in self.conns])

    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        for conn, action in zip(self.conns, actions):
            conn.send(('step', action))
        resultsactions = [conn.recv() for conn in self.conns]
        return tuple(np.array(x) for x in zip(*results))

    def render(self, episode: int):
        for conn in self.conns:
            conn.send(('render', episode))
        for conn in self.conns:
            conn.recv()  # Wait for rendering to complete

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for p in self.processes:
            p.join()

# PyGNOME 环境封装
class OilSpillEnvironment:
    """基于 PyGNOME 的溢油扩散环境"""
    def __init__(self, config: Dict):
        self.config = config
        self.num_ships = config.get('num_ships', 5)
        self.map_size = config.get('map_size', (100, 100))
        self.max_steps = config.get('max_steps', 200)
        self.state_dim = 128
        self.action_dims = config.get('action_dims', {'skimmer': 4, 'boom': 3})  # 异构动作空间
        self.ship_types = config.get('ship_types', ['skimmer'] * self.num_ships)
        self.ship_fuel = [100.0] * self.num_ships  # 燃油限制
        self.prev_min_dists = [100.0] * self.num_ships  # 奖励塑造

        # 环境状态
        self.current_step = 0
        self.oil_spill_locations = []
        self.ship_positions = []
        self.wind_data = None
        self.current_data = None
        self.gnome_model = None

        self._initialize_environment()

    def _initialize_environment(self):
        """初始化环境状态"""
        self.ship_positions = [np.random.uniform(0, self.map_size[0], 2) for _ in range(self.num_ships)]
        spill_center = (self.map_size[0] // 2, self.map_size[1] // 2)
        self.oil_spill_locations = [spill_center]
        self.wind_data = np.random.normal(0, 5, (self.map_size[0], self.map_size[1], 2))
        self.current_data = np.random.normal(0, 2, (self.map_size[0], self.map_size[1], 2))
        self.ship_fuel = [100.0] * self.num_ships
        self.prev_min_dists = [100.0] * self.num_ships

        # 初始化 PyGNOME
        try:
            self.gnome_model = Model(duration=self.max_steps, time_step=3600)
            wind = Wind(speed=self.wind_data, units='m/s')
            self.gnome_model.movers += WindMover(wind)
            self.gnome_model.movers += RandomMover(diffusion_coef=10000)
            spill = Spill(num_elements=1000, release_time=0, position=spill_center)
            self.gnome_model.spills += spill
        except NameError:
            print("PyGNOME 未安装，使用简化模拟。")

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self._initialize_environment()
        return self._get_observations()

    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, List[float], List[bool], Dict]:
        assert len(actions) == self.num_ships, f"预期 {self.num_ships} 个动作，实际得到 {len(actions)}"
        self.current_step += 1

        # 更新船只位置
        for i, (action, ship_type) in enumerate(zip(actions, self.ship_types)):
            if ship_type == 'skimmer':
                dx, dy, speed, op_type = action
            else:  # boom
                dx, dy, op_type = action
                speed = 1.0  # 固定速度
            fuel_cost = speed * 0.1 + op_type * 0.5
            self.ship_fuel[i] = max(0, self.ship_fuel[i] - fuel_cost)
            if self.ship_fuel[i] == 0:
                action = [0] * len(action)  # 无燃油时停止
                dx, dy, speed, op_type = 0, 0, 0, 0
            self.ship_positions[i][0] += dx * speed
            self.ship_positions[i][1] += dy * speed
            self.ship_positions[i] = np.clip(self.ship_positions[i], [0, 0], [self.map_size[0], self.map_size[1]])

        # 随机更新风场
        if np.random.random() < 0.1:
            self.wind_data = np.random.normal(0, 5, (self.map_size[0], self.map_size[1], 2))
            if self.gnome_model:
                self.gnome_model.movers[0].wind.speed = self.wind_data

        # 模拟油污扩散
        self._simulate_oil_spread()

        # 计算奖励
        rewards = self._calculate_rewards(actions)

        # 检查终止条件
        done = self.current_step >= self.max_steps or self._check_cleanup_complete()

        observations = self._get_observations()

        return observations, rewards, [done] * self.num_ships, {}

    def _simulate_oil_spread(self):
        """使用 PyGNOME 模拟油污扩散"""
        if self.gnome_model:
            self.gnome_model.step()
            self.oil_spill_locations = self.gnome_model.spills[0].get_positions()
        else:
            new_locations = []
            for loc in self.oil_spill_locations:
                wind_effect = self.wind_data[int(loc[0]), int(loc[1])]
                current_effect = self.current_data[int(loc[0]), int(loc[1])]
                new_x = loc[0] + 0.1 * (wind_effect[0] + current_effect[0])
                new_y = loc[1] + 0.1 * (wind_effect[1] + current_effect[1])
                new_locations.append((new_x, new_y))
            self.oil_spill_locations.extend(new_locations)

    def _get_observations(self) -> np.ndarray:
        observations = []
        max_oil_points = 10
        for i in range(self.num_ships):
            obs = []
            obs.extend(self.ship_positions[i])
            for j in range(self.num_ships):
                rel_pos = (np.array(self.ship_positions[j]) - np.array(self.ship_positions[i])).tolist() if i != j else [0.0, 0.0]
                obs.extend(rel_pos)
            oil_dists, oil_dirs = [], []
            oil_points = sorted(self.oil_spill_locations, key=lambda x: np.linalg.norm(np.array(x) - np.array(self.ship_positions[i])))
            for oil_loc in oil_points[:max_oil_points]:
                diff = np.array(oil_loc) - np.array(self.ship_positions[i])
                dist = np.linalg.norm(diff)
                oil_dists.append(dist)
                oil_dirs.extend(diff / (dist + 1e-8))
            oil_dists += [100.0] * (max_oil_points - len(oil_dists))
            oil_dirs += [0.0] * (2 * (max_oil_points - len(oil_dirs) // 2))
            obs.extend(oil_dists + oil_dirs)
            obs.extend([self.current_step / self.max_steps, len(self.oil_spill_locations)])
            obs = obs[:self.state_dim] + [0.0] * (self.state_dim - len(obs))
            observations.append(obs)
        return np.array(observations)

    def _calculate_rewards(self, actions: List[np.ndarray]) -> List[float]:
        rewards = []
        for i, (action, ship_type) in enumerate(zip(actions, self.ship_types)):
            reward = 0.0
            min_oil_dist = min([np.linalg.norm(np.array(oil_loc) - np.array(self.ship_positions[i]))
                                for oil_loc in self.oil_spill_locations], default=100.0)
            reward += max(0, 20 - min_oil_dist) * 0.1
            reward += 0.05 * (self.prev_min_dists[i] - min_oil_dist)
            self.prev_min_dists[i] = min_oil_dist
            for j in range(self.num_ships):
                if i != j:
                    dist = np.linalg.norm(np.array(self.ship_positions[i]) - np.array(self.ship_positions[j]))
                    if 5 < dist < 15:
                        reward += 0.1
                    elif dist < 5:
                        reward -= 0.2
            if min_oil_dist < 5:
                op_type = action[3] if ship_type == 'skimmer' else action[2]
                reward += op_type * 0.5
            if self.gnome_model:
                oil_mass = self.gnome_model.spills[0].get_mass()
                reward += 0.1 * (self.prev_oil_mass - oil_mass) if hasattr(self, 'prev_oil_mass') else 0.0
                self.prev_oil_mass = oil_mass
            rewards.append(reward)
        rewards = [(r - np.mean(rewards)) / (np.std(rewards) + 1e-8) for r in rewards]
        return rewards

    def _check_cleanup_complete(self) -> bool:
        return len(self.oil_spill_locations) < 5

    def visualize_environment(self, episode: int):
        """可视化环境状态"""
        plt.figure(figsize=(8, 8))
        plt.scatter([p[0] for p in self.oil_spill_locations], [p[1] for p in self.oil_spill_locations],
                    c='black', label='油污', s=10, alpha=0.5)
        for i, pos in enumerate(self.ship_positions):
            color = 'blue' if self.ship_types[i] == 'skimmer' else 'green'
            plt.scatter(pos[0], pos[1], c=color, label=f'{self.ship_types[i]}' if i == 0 else "", s=50)
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        plt.legend()
        plt.title(f"Episode {episode}, Step {self.current_step}")
        os.makedirs('viz', exist_ok=True)
        plt.savefig(f'viz/episode_{episode}_step_{self.current_step}.png')
        plt.close()

# 训练主函数
def train_maddpg(config: Dict) -> List[MADDPGAgent]:
    """MADDPG 训练主函数"""
    wandb.init(project="oil_spill_maddpg", config=config)
    env = ParallelEnv(num_envs=config.get('num_envs', 4), config=config)
    agents = []
    for i in range(config['num_ships']):
        agent = MADDPGAgent(
            agent_id=i,
            state_dim=config['state_dim'],
            action_dim=config['action_dims'],
            num_agents=config['num_ships'],
            agent_type=config['ship_types'][i],
            max_action=1.0
        )
        agents.append(agent)

    replay_buffer = ReplayBuffer(config.get('buffer_size', 100000))
    total_episodes = config.get('total_episodes', 1000)
    batch_size = config.get('batch_size', 64)
    update_freq = config.get('update_freq', 10)

    episode_rewards = []
    critic_losses = []
    actor_losses = []
    trajectories = []  # 存储最后一个 episode 的轨迹

    print("开始 MADDPG 训练...")
    for episode in range(total_episodes):
        states = env.reset()
        episode_reward = 0
        step_count = 0
        trajectory = {'ships': [], 'oil': []}

        while True:
            # 构造 edge_index（示例：全连接图）
            edge_index = torch.tensor([[i, j] for i in range(config['num_ships']) for j in range(config['num_ships']) if i != j],
                                    dtype=torch.long).t().contiguous().to(device)
            temporal_seq = np.array([states[i] for i in range(states.shape[1])])  # 示例时间序列

            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[0, i], edge_index, temporal_seq)
                actions.append(action)

            next_states, rewards, dones, _ = env.step(actions)
            replay_buffer.push(states[0], actions, rewards, next_states[0], dones)

            # 记录轨迹
            if episode == total_episodes - 1:
                trajectory['ships'].append([p.copy() for p in env.conns[0].recv()[3]['ship_positions']])
                trajectory['oil'].append([p.copy() for p in env.conns[0].recv()[3]['oil_spill_locations']])
                env.render(episode)

            states = next_states
            episode_reward += np.sum(rewards)
            step_count += 1

            if len(replay_buffer) > batch_size and step_count % update_freq == 0:
                for i, agent in enumerate(agents):
                    other_agents = [agents[j] for j in range(len(agents)) if j != i]
                    c_loss, a_loss = agent.update(replay_buffer, batch_size, other_agents, edge_index)
                    if c_loss is not None:
                        critic_losses.append(c_loss)
                        actor_losses.append(a_loss)
                        wandb.log({'critic_loss': c_loss, 'actor_loss': a_loss})

            if all(dones):
                break

        episode_rewards.append(episode_reward)
        wandb.log({'episode': episode, 'reward': episode_reward})
        if episode == total_episodes - 1:
            trajectories.append(trajectory)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
            if episode % 200 == 0:
                save_models(agents, episode)

    env.close()
    plot_training_curves(episode_rewards, critic_losses, actor_losses)
    return agents, trajectories

def save_models(agents: List[MADDPGAgent], episode: int):
    """保存模型"""
    try:
        save_dir = f"models/episode_{episode}"
        os.makedirs(save_dir, exist_ok=True)
        for i, agent in enumerate(agents):
            torch.save(agent.actor.state_dict(), f"{save_dir}/agent_{i}_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/agent_{i}_critic.pth")
        print(f"模型已保存至 {save_dir}")
    except OSError as e:
        logging.error(f"模型保存失败: {e}")

def plot_training_curves(episode_rewards: List[float], critic_losses: List[float], actor_losses: List[float]):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[1].plot(critic_losses)
    axes[1].set_title('Critic Loss')
    axes[2].plot(actor_losses)
    axes[2].set_title('Actor Loss')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()
    wandb.log({"training_curves": wandb.Image('training_curves.png')})

def generate_gif(trajectory: Dict, episode: int):
    """生成最后一个 episode 的 GIF"""
    images = []
    for step in range(len(trajectory['ships'])):
        plt.figure(figsize=(8, 8))
        plt.scatter([p[0] for p in trajectory['oil'][step]], [p[1] for p in trajectory['oil'][step]],
                    c='black', label='油污', s=10, alpha=0.5)
        for i, pos in enumerate(trajectory['ships'][step]):
            color = 'blue' if config['ship_types'][i] == 'skimmer' else 'green'
            plt.scatter(pos[0], pos[1], c=color, label=f'{config["ship_types"][i]}' if i == 0 else "", s=50)
        plt.xlim(0, config['map_size'][0])
        plt.ylim(0, config['map_size'][1])
        plt.legend()
        plt.title(f"Episode {episode}, Step {step}")
        plt.savefig(f'viz/episode_{episode}_step_{step}.png')
        images.append(imageio.imread(f'viz/episode_{episode}_step_{step}.png'))
        plt.close()
    imageio.mimsave(f'viz/episode_{episode}_trajectory.gif', images, fps=5)
    print(f"GIF 已保存至 viz/episode_{episode}_trajectory.gif")
    wandb.log({"trajectory_gif": wandb.Image(f'viz/episode_{episode}_trajectory.gif')})

# 主训练配置和执行
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = {
        'num_ships': 5,
        'map_size': (100, 100),
        'max_steps': 200,
        'num_envs': 4,
        'total_episodes': 2000,
        'batch_size': 128,
        'buffer_size': 100000,
        'update_freq': 10,
        'state_dim': 128,
        'action_dims': {'skimmer': 4, 'boom': 3},
        'ship_types': ['skimmer', 'skimmer', 'boom', 'skimmer', 'boom'],
        'learning_rate_actor': 1e-4,
        'learning_rate_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005
    }

    print("海上溢油应急调度 MADDPG 训练系统")
    print("=" * 50)
    print(f"配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    agents, trajectories = train_maddpg(config)
    if trajectories:
        generate_gif(trajectories[-1], config['total_episodes'] - 1)

    print("训练完成！")
    print("模型已保存，训练曲线和 GIF 已生成。")