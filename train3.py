import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import gym
from gym import spaces
import pickle
import logging
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from datetime import datetime
import os
import random
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import json
import warnings

warnings.filterwarnings('ignore')

# PyGNOME相关导入（需要单独安装）
try:
    from gnome import scripting
    from gnome.model import Model
    from gnome.maps import GnomeMap
    from gnome.environment import Wind, Tide
    from gnome.spill import point_line_release_spill
    from gnome.movers import RandomMover, WindMover, CatsMover
    from gnome.outputters import Renderer

    GNOME_AVAILABLE = True
except ImportError:
    GNOME_AVAILABLE = False
    print("PyGNOME not available, using simplified oil spill simulation")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 经验回放缓冲区
Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones', 'graphs'])


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, states, actions, rewards, next_states, dones, graphs=None):
        experience = Experience(states, actions, rewards, next_states, dones, graphs)
        self.buffer.append(experience)
        self.priorities.append(float(self.max_priority))

    def sample(self, batch_size: int, beta: float = 0.4):
        if len(self.buffer) < batch_size:
            return None

        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]

        states = torch.stack([torch.tensor(e.states, dtype=torch.float32) for e in batch])
        actions = torch.stack([torch.tensor(e.actions, dtype=torch.float32) for e in batch])
        rewards = torch.stack([torch.tensor(e.rewards, dtype=torch.float32) for e in batch])
        next_states = torch.stack([torch.tensor(e.next_states, dtype=torch.float32) for e in batch])
        dones = torch.stack([torch.tensor(e.dones, dtype=torch.bool) for e in batch])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
            self.max_priority = max(self.max_priority, float(priority))

    def __len__(self):
        return len(self.buffer)


class AttentionMechanism(nn.Module):
    """多头注意力机制"""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super(AttentionMechanism, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]

        # 分离为多个头
        values = self.values(values).view(N, -1, self.num_heads, self.head_dim)
        keys = self.keys(keys).view(N, -1, self.num_heads, self.head_dim)
        queries = self.queries(query).view(N, -1, self.num_heads, self.head_dim)

        # 计算注意力分数
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_dim ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, -1, self.head_dim * self.num_heads
        )

        out = self.fc_out(out)
        return out


class EnhancedGNNEncoder(nn.Module):
    """增强的图神经网络编码器，集成空间关系和注意力机制"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super(EnhancedGNNEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 图注意力层
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.1)
        self.gat3 = GATConv(hidden_dim * num_heads, output_dim, heads=1, dropout=0.1)

        # 批量归一化
        self.bn1 = nn.LayerNorm(hidden_dim * num_heads)
        self.bn2 = nn.LayerNorm(hidden_dim * num_heads)

        # 残差连接
        self.residual = nn.Linear(input_dim, output_dim)

        # 自注意力机制
        self.self_attention = AttentionMechanism(output_dim, num_heads=4)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index, batch=None):
        # 保存输入用于残差连接
        residual = x

        # 第一层GAT
        x = F.elu(self.gat1(x, edge_index))
        if batch is not None:
            x = self.bn1(x)
            x = self.dropout(x)

        # 第二层GAT
        x = F.elu(self.gat2(x, edge_index))
        if batch is not None:
            x = self.bn2(x)
        x = self.dropout(x)

        # 第三层GAT
        x = self.gat3(x, edge_index)

        # 残差连接
        if residual.size(-1) != x.size(-1):
            residual = self.residual(residual)
        x = x + residual

        # 自注意力机制
        if batch is not None:
            # 对每个图应用自注意力
            unique_batches = torch.unique(batch)
            attended_x = []
            for b in unique_batches:
                mask = (batch == b)
                graph_x = x[mask].unsqueeze(0)
                attended_graph_x = self.self_attention(graph_x, graph_x, graph_x)
                attended_x.append(attended_graph_x.squeeze(0))
            x = torch.cat(attended_x, dim=0)
        else:
            x = self.self_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)

        return x


class EnhancedTemporalEncoder(nn.Module):
    """增强的时间序列编码器，集成LSTM和注意力机制"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 4):
        super(EnhancedTemporalEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 双向LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.1)

        # 时间注意力机制
        self.temporal_attention = AttentionMechanism(hidden_dim * 2, num_heads=num_heads)

        # 输出层
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x, lengths=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()

        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(x)

        # 时间注意力
        attended_out = self.temporal_attention(lstm_out, lstm_out, lstm_out)

        # 池化操作
        if lengths is not None:
            # 掩码池化
            mask = torch.arange(seq_len).expand(batch_size, seq_len).to(x.device)
            mask = mask < lengths.unsqueeze(1)
            attended_out = attended_out * mask.unsqueeze(-1).float()
            pooled = attended_out.sum(1) / lengths.unsqueeze(-1).float()
        else:
            pooled = attended_out.mean(1)

        output = self.fc(pooled)
        return output


class HeterogeneousActor(nn.Module):
    """异构智能体的Actor网络"""

    def __init__(self, agent_type: str, state_dim: int, action_dim: int, max_action: float,
                 gnn_hidden_dim: int = 128, temporal_hidden_dim: int = 64):
        super(HeterogeneousActor, self).__init__()
        self.agent_type = agent_type
        self.max_action = max_action
        self.state_dim = state_dim

        # 基础特征提取
        self.base_features = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # GNN编码器
        self.gnn_encoder = EnhancedGNNEncoder(128, gnn_hidden_dim, 64)

        # 时间序列编码器
        self.temporal_encoder = EnhancedTemporalEncoder(128, temporal_hidden_dim)

        # 特定于智能体类型的网络
        if agent_type == "cleanup_ship":
            feature_dim = 128 + 64 + 64  # base + gnn + temporal
            self.type_specific = nn.Sequential(
                nn.Linear(feature_dim, 128),
            nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        elif agent_type == "containment_ship":
            feature_dim = 128 + 64 + 64
            self.type_specific = nn.Sequential(
                nn.Linear(feature_dim, 128),
            nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
        elif agent_type == "dispersant_ship":
            feature_dim = 128 + 64 + 64
            self.type_specific = nn.Sequential(
                nn.Linear(feature_dim, 128),
            nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
        )
        else:
            feature_dim = 128 + 64 + 64
            self.type_specific = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

        # 最终策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )

    def forward(self, state, edge_index=None, temporal_seq=None, batch=None):
        # 确保输入维度正确
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # 基础特征提取
        base_features = self.base_features(state)

        # GNN特征（如果提供了图结构）
        if edge_index is not None:
            gnn_features = self.gnn_encoder(base_features, edge_index, batch)
            if batch is not None:
                # 聚合图特征
                gnn_features = global_mean_pool(gnn_features, batch)
        else:
            gnn_features = torch.zeros(base_features.size(0), 64).to(state.device)

        # 时间序列特征（如果提供了时间序列）
        if temporal_seq is not None:
            temporal_features = self.temporal_encoder(temporal_seq)
        else:
            temporal_features = torch.zeros(base_features.size(0), 64).to(state.device)

        # 特征融合
        combined_features = torch.cat([base_features, gnn_features, temporal_features], dim=1)

        # 类型特定处理
        type_features = self.type_specific(combined_features)

        # 策略输出
        action = self.policy_net(type_features)

        return self.max_action * action


class EnhancedCritic(nn.Module):
    """增强的Critic网络"""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int,
                 gnn_hidden_dim: int = 128):
        super(EnhancedCritic, self).__init__()

        # 全局状态和动作维度
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(total_state_dim, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 全局图编码器
        self.global_gnn = EnhancedGNNEncoder(256, gnn_hidden_dim, 64)

        # 价值函数网络
        self.value_net = nn.Sequential(
            nn.Linear(256 + 128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, states, actions, global_edge_index=None, batch=None):
        # 展平为全局状态和动作
        states_flat = states.view(states.size(0), -1)
        actions_flat = actions.view(actions.size(0), -1)

        # 编码状态和动作
        state_features = self.state_encoder(states_flat)
        action_features = self.action_encoder(actions_flat)

        # 全局图特征
        if global_edge_index is not None:
            global_features = self.global_gnn(state_features, global_edge_index, batch)
            if batch is not None:
                global_features = global_mean_pool(global_features, batch)
        else:
            global_features = torch.zeros(state_features.size(0), 64).to(states.device)

        # 特征融合
        combined_features = torch.cat([state_features, action_features, global_features], dim=1)

        # 价值函数
        q_value = self.value_net(combined_features)

        return q_value


class EnhancedMADDPGAgent:
    """增强的MADDPG智能体"""

    def __init__(self, agent_id: int, agent_type: str, state_dim: int, action_dim: int,
                 num_agents: int, max_action: float = 1.0,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 tau: float = 0.005, gamma: float = 0.99):

        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma

        # Actor网络
        self.actor = HeterogeneousActor(agent_type, state_dim, action_dim, max_action).to(device)
        self.actor_target = HeterogeneousActor(agent_type, state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5)

        # Critic网络
        self.critic = EnhancedCritic(state_dim, action_dim, num_agents).to(device)
        self.critic_target = EnhancedCritic(state_dim, action_dim, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5)

        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=500, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=500, gamma=0.95)

        # 初始化目标网络
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        # 噪声参数
        self.noise_scale = 0.1
        self.noise_decay = 0.995
        self.min_noise = 0.01

    def select_action(self, state, edge_index=None, temporal_seq=None, batch=None, add_noise=True):
        """选择动作"""
        if isinstance(state, list):
            state = np.array(state)

        state = torch.FloatTensor(state).to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state, edge_index, temporal_seq, batch).cpu().numpy()
            if len(action.shape) > 1:
                action = action[0]

        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)

        return action

    def update(self, replay_buffer: PrioritizedReplayBuffer, batch_size: int,
               other_agents: List['EnhancedMADDPGAgent'], beta: float = 0.4):
        """更新网络参数"""
        if len(replay_buffer) < batch_size:
            return None, None

        # 采样经验
        sample_result = replay_buffer.sample(batch_size, beta)
        if sample_result is None:
            return None, None

        states, actions, rewards, next_states, dones, indices, weights = sample_result
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        weights = torch.FloatTensor(weights).to(device)

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
        td_errors = target_q - current_q
        critic_loss = (weights * td_errors.pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # 更新Actor
        actions_pred = actions.clone()
        agent_state = states[:, self.agent_id]
        predicted_action = self.actor(agent_state)
        actions_pred[:, self.agent_id] = predicted_action

        actor_loss = -(weights * self.critic(states, actions_pred)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # 更新优先级
        priorities = torch.abs(td_errors).detach().cpu().numpy() + 1e-6
        replay_buffer.update_priorities(indices, priorities)

        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()

        # 衰减噪声
        self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)

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


class EnhancedOilSpillEnvironment:
    """增强的溢油扩散环境，集成PyGNOME"""

    def __init__(self, config: dict):
        self.config = config
        self.num_ships = config.get('num_ships', 5)
        self.map_size = config.get('map_size', (100, 100))
        self.max_steps = config.get('max_steps', 200)
        self.use_gnome = config.get('use_gnome', GNOME_AVAILABLE)

        # 异构智能体配置
        self.agent_types = config.get('agent_types', ['cleanup_ship'] * self.num_ships)

        # 状态和动作空间定义
        self.state_dim = 150  # 增加状态维度
        self.action_dim = 6  # 增加动作维度

        # 环境状态
        self.current_step = 0
        self.oil_spill_locations = []
        self.oil_concentrations = []
        self.ship_positions = []
        self.ship_capabilities = []
        self.wind_data = None
        self.current_data = None
        self.weather_conditions = {}

        # 环境变化参数
        self.weather_change_prob = config.get('weather_change_prob', 0.1)
        self.equipment_failure_prob = config.get('equipment_failure_prob', 0.05)

        # PyGNOME模型
        self.gnome_model = None
        if self.use_gnome:
            self._initialize_gnome()

        self._initialize_environment()

        # 奖励塑造参数
        self.reward_config = {
            'oil_proximity': 0.2,
            'cooperation': 0.3,
            'efficiency': 0.4,
            'safety': 0.1,
            'fuel_consumption': -0.1,
            'equipment_usage': -0.05
        }

    def _initialize_gnome(self):
        """初始化PyGNOME模型"""
        if not GNOME_AVAILABLE:
            return

        try:
            # 创建模型
            self.gnome_model = Model(start_time=datetime.now(),
                                     duration=timedelta(hours=24),
                                     time_step=600)  # 10分钟时间步

            # 添加地图
            # 这里需要根据实际区域配置地图文件
            # map_file = self.config.get('map_file', 'default_map.bna')
            # self.gnome_model.map = GnomeMap(map_file)

            # 添加风场
            wind = Wind(filename=self.config.get('wind_file', None))
            self.gnome_model.environment += wind

            # 添加洋流
            if self.config.get('current_file'):
                current = Tide(filename=self.config['current_file'])
                self.gnome_model.environment += current

            # 添加移动器
            self.gnome_model.movers += RandomMover(diffusion_coeff=50000)
            self.gnome_model.movers += WindMover(wind)

            # 添加溢油源
            spill = point_line_release_spill(num_elements=1000,
                                             start_position=self.config.get('spill_position', (0, 0, 0)),
                                             release_time=datetime.now())
            self.gnome_model.spills += spill

            print("PyGNOME模型初始化成功")

        except Exception as e:
            print(f"PyGNOME初始化失败: {e}, 使用简化模拟")
            self.use_gnome = False

    def _initialize_environment(self):
        """初始化环境状态"""
        # 初始化船只位置和能力
        self.ship_positions = []
        self.ship_capabilities = []

        for i in range(self.num_ships):
            position = np.random.uniform(0, self.map_size[0], 2)
            self.ship_positions.append(position)

            # 不同类型船只的能力
            if self.agent_types[i] == 'cleanup_ship':
                capability = {
                    'cleanup_rate': np.random.uniform(0.8, 1.2),
                    'fuel_capacity': np.random.uniform(0.9, 1.1),
                    'equipment_durability': np.random.uniform(0.8, 1.2)
                }
            elif self.agent_types[i] == 'containment_ship':
                capability = {
                    'containment_rate': np.random.uniform(0.8, 1.2),
                    'fuel_capacity': np.random.uniform(0.9, 1.1),
                    'equipment_durability': np.random.uniform(0.8, 1.2)
                }
            else:  # dispersant_ship
                capability = {
                    'dispersant_rate': np.random.uniform(0.8, 1.2),
                    'fuel_capacity': np.random.uniform(0.9, 1.1),
                    'equipment_durability': np.random.uniform(0.8, 1.2)
                }

            self.ship_capabilities.append(capability)

        # 初始化溢油位置和浓度
            spill_center = (self.map_size[0] // 2, self.map_size[1] // 2)
        self.oil_spill_locations = [spill_center]
        self.oil_concentrations = [1.0]  # 初始浓度

        # 初始化环境数据
        self._update_environmental_data()

    def _update_environmental_data(self):
        """更新环境数据"""
        # 模拟风流场数据
        self.wind_data = np.random.normal(0, 5, (self.map_size[0], self.map_size[1], 2))
        self.current_data = np.random.normal(0, 2, (self.map_size[0], self.map_size[1], 2))

        # 天气条件
        self.weather_conditions = {
            'wind_speed': np.random.uniform(2, 15),
            'wave_height': np.random.uniform(0.5, 3.0),
            'visibility': np.random.uniform(0.5, 10.0),
            'temperature': np.random.uniform(10, 25)
        }

        # 随机天气变化
        if np.random.random() < self.weather_change_prob:
            self.weather_conditions['wind_speed'] *= np.random.uniform(0.8, 1.2)
            self.weather_conditions['wave_height'] *= np.random.uniform(0.8, 1.2)

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self._initialize_environment()
        return self._get_observations(), self._get_graph_data()

    def step(self, actions):
        """环境步进"""
        self.current_step += 1

        # 随机设备故障
        equipment_failures = np.random.random(self.num_ships) < self.equipment_failure_prob

        # 更新船只位置和执行动作
        for i, action in enumerate(actions):
            if not equipment_failures[i]:
                # action: [dx, dy, speed, operation_intensity, fuel_usage, equipment_usage]
                dx, dy, speed, op_intensity, fuel_usage, equip_usage = action

                # 移动船只
                self.ship_positions[i][0] += dx * speed
                self.ship_positions[i][1] += dy * speed

                # 边界约束
                self.ship_positions[i] = np.clip(
                    self.ship_positions[i],
                    [0, 0],
                    [self.map_size[0], self.map_size[1]]
                )

                # 更新船只能力（设备磨损）
                for key in self.ship_capabilities[i]:
                    if 'durability' in key:
                        self.ship_capabilities[i][key] *= (1 - equip_usage * 0.001)
                        self.ship_capabilities[i][key] = max(0.1, self.ship_capabilities[i][key])

        # 更新溢油扩散
        self._simulate_oil_spread()

        # 执行清理操作
        self._execute_cleanup_operations(actions, equipment_failures)

        # 更新环境数据
        self._update_environmental_data()

        # 计算奖励
        rewards = self._calculate_enhanced_rewards(actions, equipment_failures)

        # 检查终止条件
        done = self._check_termination()

        observations = self._get_observations()
        graph_data = self._get_graph_data()

        return observations, rewards, [done] * self.num_ships, {
            'oil_remaining': len(self.oil_spill_locations),
            'weather': self.weather_conditions,
            'equipment_failures': equipment_failures
        }, graph_data

    def _simulate_oil_spread(self):
        """模拟油污扩散过程"""
        if self.use_gnome and self.gnome_model:
            try:
                # 使用PyGNOME进行真实模拟
                self.gnome_model.step()

                # 获取粒子位置
                particles = self.gnome_model.get_spill_container_by_name('spill')
                if particles:
                    positions = particles.get_positions()
                    self.oil_spill_locations = [(pos[0], pos[1]) for pos in positions]
                    # 根据粒子密度计算浓度
                    self.oil_concentrations = [1.0] * len(self.oil_spill_locations)

            except Exception as e:
                print(f"PyGNOME模拟出错: {e}, 使用简化模拟")
                self._simulate_oil_spread_simple()
        else:
            self._simulate_oil_spread_simple()

    def _simulate_oil_spread_simple(self):
        """简化的油污扩散模拟"""
        new_locations = []
        new_concentrations = []

        for i, (loc, conc) in enumerate(zip(self.oil_spill_locations, self.oil_concentrations)):
            # 基于风流场数据模拟扩散
            x, y = int(np.clip(loc[0], 0, self.map_size[0] - 1)), int(np.clip(loc[1], 0, self.map_size[1] - 1))

            wind_effect = self.wind_data[x, y]
            current_effect = self.current_data[x, y]

            # 主要扩散
            spread_factor = 0.1 * (1 + self.weather_conditions['wind_speed'] / 10)
            new_x = loc[0] + spread_factor * (wind_effect[0] + current_effect[0])
            new_y = loc[1] + spread_factor * (wind_effect[1] + current_effect[1])

            # 边界约束
            new_x = np.clip(new_x, 0, self.map_size[0])
            new_y = np.clip(new_y, 0, self.map_size[1])

            new_locations.append((new_x, new_y))

            # 浓度自然衰减
            new_conc = conc * 0.995  # 自然降解
            new_concentrations.append(new_conc)

            # 随机扩散产生新油污点
            if np.random.random() < 0.1 and conc > 0.5:
                spread_distance = np.random.uniform(1, 5)
                spread_angle = np.random.uniform(0, 2 * np.pi)

                spread_x = new_x + spread_distance * np.cos(spread_angle)
                spread_y = new_y + spread_distance * np.sin(spread_angle)

                spread_x = np.clip(spread_x, 0, self.map_size[0])
                spread_y = np.clip(spread_y, 0, self.map_size[1])

                new_locations.append((spread_x, spread_y))
                new_concentrations.append(conc * 0.3)

        self.oil_spill_locations = new_locations
        self.oil_concentrations = new_concentrations

        # 移除低浓度油污
        filtered_locations = []
        filtered_concentrations = []
        for loc, conc in zip(self.oil_spill_locations, self.oil_concentrations):
            if conc > 0.01:  # 最小浓度阈值
                filtered_locations.append(loc)
                filtered_concentrations.append(conc)

        self.oil_spill_locations = filtered_locations
        self.oil_concentrations = filtered_concentrations

    def _execute_cleanup_operations(self, actions, equipment_failures):
        """执行清理操作"""
        for i, action in enumerate(actions):
            if equipment_failures[i]:
                continue

            ship_pos = self.ship_positions[i]
            agent_type = self.agent_types[i]
            op_intensity = action[3]

            # 寻找附近的油污
            nearby_oil_indices = []
            for j, oil_loc in enumerate(self.oil_spill_locations):
                distance = np.linalg.norm(np.array(ship_pos) - np.array(oil_loc))
                if distance < 5:  # 有效作业距离
                    nearby_oil_indices.append(j)

            if nearby_oil_indices:
                # 根据智能体类型执行不同的清理操作
                if agent_type == 'cleanup_ship':
                    cleanup_rate = self.ship_capabilities[i]['cleanup_rate']
                    for idx in nearby_oil_indices:
                        cleanup_amount = op_intensity * cleanup_rate * 0.1
                        self.oil_concentrations[idx] = max(0, self.oil_concentrations[idx] - cleanup_amount)

                elif agent_type == 'containment_ship':
                    containment_rate = self.ship_capabilities[i]['containment_rate']
                    for idx in nearby_oil_indices:
                        # 围堵操作降低扩散速度
                        containment_effect = op_intensity * containment_rate * 0.05
                        self.oil_concentrations[idx] *= (1 - containment_effect)

                elif agent_type == 'dispersant_ship':
                    dispersant_rate = self.ship_capabilities[i]['dispersant_rate']
                    for idx in nearby_oil_indices:
                        # 分散剂操作
                        dispersant_effect = op_intensity * dispersant_rate * 0.08
                        self.oil_concentrations[idx] *= (1 - dispersant_effect)

    def _get_observations(self):
        """获取观测状态"""
        observations = []

        for i in range(self.num_ships):
            obs = []

            # 船只自身状态
            ship_state = list(self.ship_positions[i])
            obs.extend(ship_state)

            # 船只能力状态
            for key, value in self.ship_capabilities[i].items():
                obs.append(value)

            # 智能体类型编码
            type_encoding = [0, 0, 0]
            if self.agent_types[i] == 'cleanup_ship':
                type_encoding[0] = 1
            elif self.agent_types[i] == 'containment_ship':
                type_encoding[1] = 1
            else:  # dispersant_ship
                type_encoding[2] = 1
            obs.extend(type_encoding)

            # 相对于其他船只的位置
            for j in range(self.num_ships):
                if i != j:
                    rel_pos = np.array(self.ship_positions[j]) - np.array(self.ship_positions[i])
                    obs.extend(rel_pos.tolist())
                    # 其他船只类型
                    other_type = [0, 0, 0]
                    if self.agent_types[j] == 'cleanup_ship':
                        other_type[0] = 1
                    elif self.agent_types[j] == 'containment_ship':
                        other_type[1] = 1
                    else:
                        other_type[2] = 1
                    obs.extend(other_type)
                else:
                    obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            # 油污信息
            if self.oil_spill_locations:
                # 最近的油污点
                distances = []
                directions = []
                concentrations = []

                for oil_loc, oil_conc in zip(self.oil_spill_locations[:10], self.oil_concentrations[:10]):
                    diff = np.array(oil_loc) - np.array(self.ship_positions[i])
                    dist = np.linalg.norm(diff)
                    direction = diff / (dist + 1e-8)

                    distances.append(dist)
                    directions.extend(direction.tolist())
                    concentrations.append(oil_conc)

                # 填充至固定长度
                while len(distances) < 10:
                    distances.append(100.0)
                    directions.extend([0.0, 0.0])
                    concentrations.append(0.0)

                obs.extend(distances)
                obs.extend(directions)
                obs.extend(concentrations)
            else:
                obs.extend([100.0] * 10)  # 距离
                obs.extend([0.0] * 20)  # 方向
                obs.extend([0.0] * 10)  # 浓度

            # 环境信息
            obs.append(self.current_step / self.max_steps)
            obs.extend([
                self.weather_conditions['wind_speed'] / 20,
                self.weather_conditions['wave_height'] / 5,
                self.weather_conditions['visibility'] / 10,
                self.weather_conditions['temperature'] / 30
            ])

            # 全局油污统计
            total_oil = sum(self.oil_concentrations) if self.oil_concentrations else 0
            obs.append(total_oil / 100)  # 归一化
            obs.append(len(self.oil_spill_locations) / 100)  # 归一化

            # 确保观测维度正确
            if len(obs) < self.state_dim:
                obs.extend([0.0] * (self.state_dim - len(obs)))
            elif len(obs) > self.state_dim:
                obs = obs[:self.state_dim]

            observations.append(obs)

        return np.array(observations)

    def _get_graph_data(self):
        """获取图结构数据"""
        # 构建智能体之间的图结构
        edge_index = []
        edge_attr = []

        for i in range(self.num_ships):
            for j in range(self.num_ships):
                if i != j:
                    distance = np.linalg.norm(
                        np.array(self.ship_positions[i]) - np.array(self.ship_positions[j])
                    )

                    # 只连接距离较近的智能体
                    if distance < 20:
                        edge_index.append([i, j])
                        edge_attr.append([distance, 1.0])  # 距离和连接强度

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)

        return edge_index, edge_attr

    def _calculate_enhanced_rewards(self, actions, equipment_failures):
        """计算增强奖励函数"""
        rewards = []

        for i in range(self.num_ships):
            reward = 0.0

            # 基础奖励：接近油污
            if self.oil_spill_locations:
                min_oil_dist = float('inf')
                max_oil_conc = 0

                for oil_loc, oil_conc in zip(self.oil_spill_locations, self.oil_concentrations):
                    dist = np.linalg.norm(np.array(oil_loc) - np.array(self.ship_positions[i]))
                    if dist < min_oil_dist:
                        min_oil_dist = dist
                        max_oil_conc = oil_conc

                # 距离奖励（考虑浓度）
                proximity_reward = self.reward_config['oil_proximity'] * max_oil_conc * max(0, 20 - min_oil_dist) / 20
                reward += proximity_reward

            # 协作奖励
            cooperation_reward = 0
            for j in range(self.num_ships):
                if i != j:
                    dist = np.linalg.norm(
                        np.array(self.ship_positions[i]) - np.array(self.ship_positions[j])
                    )

                    # 不同类型智能体的协作奖励
                    if self.agent_types[i] != self.agent_types[j]:
                        if 8 < dist < 15:  # 理想协作距离
                            cooperation_reward += 0.2
                        elif dist < 5:  # 过于接近
                            cooperation_reward -= 0.1
                    else:
                        if 10 < dist < 20:  # 同类型智能体保持距离
                            cooperation_reward += 0.1
                        elif dist < 8:
                            cooperation_reward -= 0.15

            reward += self.reward_config['cooperation'] * cooperation_reward

            # 效率奖励
            if not equipment_failures[i]:
                action = actions[i]
                op_intensity = action[3]

                # 根据智能体类型给予不同的效率奖励
                if self.agent_types[i] == 'cleanup_ship':
                    efficiency_reward = op_intensity * self.ship_capabilities[i]['cleanup_rate']
                elif self.agent_types[i] == 'containment_ship':
                    efficiency_reward = op_intensity * self.ship_capabilities[i]['containment_rate']
                else:  # dispersant_ship
                    efficiency_reward = op_intensity * self.ship_capabilities[i]['dispersant_rate']

                reward += self.reward_config['efficiency'] * efficiency_reward

            # 安全奖励（考虑天气条件）
            safety_penalty = 0
            if self.weather_conditions['wind_speed'] > 12:
                safety_penalty += 0.1
            if self.weather_conditions['wave_height'] > 2:
                safety_penalty += 0.1
            if self.weather_conditions['visibility'] < 2:
                safety_penalty += 0.1

            reward += self.reward_config['safety'] * (1 - safety_penalty)

            # 燃料消耗惩罚
            fuel_usage = actions[i][4] if len(actions[i]) > 4 else 0.1
            reward += self.reward_config['fuel_consumption'] * fuel_usage

            # 设备使用惩罚
            equipment_usage = actions[i][5] if len(actions[i]) > 5 else 0.1
            reward += self.reward_config['equipment_usage'] * equipment_usage

            # 设备故障惩罚
            if equipment_failures[i]:
                reward -= 1.0

            # 塑造奖励：清理进度
            if hasattr(self, 'previous_total_oil'):
                current_total_oil = sum(self.oil_concentrations) if self.oil_concentrations else 0
                cleanup_progress = self.previous_total_oil - current_total_oil
                reward += cleanup_progress * 2.0  # 清理进度奖励

            rewards.append(reward)

        # 记录当前总油污量
        self.previous_total_oil = sum(self.oil_concentrations) if self.oil_concentrations else 0

        return rewards

    def _check_termination(self):
        """检查终止条件"""
        # 时间限制
        if self.current_step >= self.max_steps:
            return True

        # 清理完成
        if not self.oil_spill_locations or sum(self.oil_concentrations) < 0.1:
            return True

        # 严重天气条件
        if (self.weather_conditions['wind_speed'] > 20 or
                self.weather_conditions['wave_height'] > 4 or
                self.weather_conditions['visibility'] < 1):
            return True

        return False

    def render(self):
        """环境可视化"""
        plt.figure(figsize=(12, 10))

        # 绘制地图
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])

        # 绘制油污
        if self.oil_spill_locations:
            oil_x = [loc[0] for loc in self.oil_spill_locations]
            oil_y = [loc[1] for loc in self.oil_spill_locations]
            oil_c = self.oil_concentrations

            scatter = plt.scatter(oil_x, oil_y, c=oil_c, s=100, cmap='Reds', alpha=0.7)
            plt.colorbar(scatter, label='Oil Concentration')

        # 绘制船只
        colors = {'cleanup_ship': 'blue', 'containment_ship': 'green', 'dispersant_ship': 'orange'}
        for i, (pos, agent_type) in enumerate(zip(self.ship_positions, self.agent_types)):
            plt.scatter(pos[0], pos[1], c=colors[agent_type], s=200, marker='^')
            plt.text(pos[0], pos[1] + 2, f'Ship{i}', ha='center', fontsize=8)

        # 绘制风向
        wind_scale = 2
        for i in range(0, self.map_size[0], 10):
            for j in range(0, self.map_size[1], 10):
                wind = self.wind_data[i, j]
                plt.arrow(i, j, wind[0] * wind_scale, wind[1] * wind_scale,
                          head_width=1, head_length=1, fc='gray', ec='gray', alpha=0.5)

        plt.title(f'Oil Spill Response - Step {self.current_step}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(['Oil Spill', 'Cleanup Ship', 'Containment Ship', 'Dispersant Ship'])
        plt.grid(True, alpha=0.3)
        plt.show()


def parallel_environment_worker(env_id, config, shared_buffer, episodes_per_worker):
    """并行环境工作进程"""
    env = EnhancedOilSpillEnvironment(config)

    # 创建智能体
    agents = []
    for i in range(env.num_ships):
        agent = EnhancedMADDPGAgent(
            agent_id=i,
            agent_type=env.agent_types[i],
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            num_agents=env.num_ships,
            max_action=1.0
        )
        agents.append(agent)

    experiences = []

    for episode in range(episodes_per_worker):
        states, graph_data = env.reset()
        episode_reward = 0

        while True:
            # 选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[i])
                actions.append(action)

            # 执行动作
            next_states, rewards, dones, info, next_graph_data = env.step(actions)

            # 存储经验
            experience = {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'next_states': next_states,
                'dones': dones,
                'graph_data': graph_data
            }
            experiences.append(experience)

            states = next_states
            graph_data = next_graph_data
            episode_reward += np.sum(rewards)

            if all(dones):
                break

        # 将经验添加到共享缓冲区
        shared_buffer.extend(experiences)
        experiences = []

    return len(shared_buffer)


def train_enhanced_maddpg(config: dict):
    """增强的MADDPG训练主函数"""

    # 创建环境
    env = EnhancedOilSpillEnvironment(config)

    # 创建智能体
    agents = []
    for i in range(env.num_ships):
        agent = EnhancedMADDPGAgent(
            agent_id=i,
            agent_type=env.agent_types[i],
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            num_agents=env.num_ships,
            max_action=1.0,
            lr_actor=config.get('lr_actor', 1e-4),
            lr_critic=config.get('lr_critic', 1e-3)
        )
        agents.append(agent)

    # 优先级经验回放缓冲区
    replay_buffer = PrioritizedReplayBuffer(config.get('buffer_size', 200000))

    # TensorBoard记录
    writer = SummaryWriter(f"runs/enhanced_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 训练参数
    total_episodes = config.get('total_episodes', 2000)
    batch_size = config.get('batch_size', 128)
    update_freq = config.get('update_freq', 100)
    save_freq = config.get('save_freq', 200)

    # 记录训练过程
    episode_rewards = []
    critic_losses = []
    actor_losses = []

    print("开始增强MADDPG训练...")
    print(f"智能体类型: {env.agent_types}")

    for episode in range(total_episodes):
        states, graph_data = env.reset()
        edge_index, edge_attr = graph_data

        episode_reward = 0
        episode_steps = 0
        episode_critic_loss = 0
        episode_actor_loss = 0

        while True:
            # 选择动作
            actions = []
            for i, agent in enumerate(agents):
                action = agent.select_action(states[i])
                actions.append(action)

            # 执行动作
            next_states, rewards, dones, info, next_graph_data = env.step(actions)

            # 存储经验
            replay_buffer.push(states, actions, rewards, next_states, dones, graph_data)

            states = next_states
            graph_data = next_graph_data
            episode_reward += np.sum(rewards)
            episode_steps += 1

            # 更新网络
            if len(replay_buffer) > batch_size and episode_steps % update_freq == 0:
                beta = min(1.0, 0.4 + (episode / total_episodes) * 0.6)  # 线性增加beta

                for i, agent in enumerate(agents):
                    other_agents = [agents[j] for j in range(len(agents)) if j != i]
                    losses = agent.update(replay_buffer, batch_size, other_agents, beta)

                    if losses[0] is not None:
                        episode_critic_loss += losses[0]
                        episode_actor_loss += losses[1]

            if all(dones):
                break

            episode_rewards.append(episode_reward)
        if episode_critic_loss > 0:
            critic_losses.append(episode_critic_loss / episode_steps)
            actor_losses.append(episode_actor_loss / episode_steps)

        # 记录到TensorBoard
        writer.add_scalar('Episode/Reward', episode_reward, episode)
        writer.add_scalar('Episode/Steps', episode_steps, episode)
        writer.add_scalar('Episode/Oil_Remaining', info.get('oil_remaining', 0), episode)
        writer.add_scalar('Environment/Wind_Speed', info.get('weather', {}).get('wind_speed', 0), episode)
        writer.add_scalar('Environment/Wave_Height', info.get('weather', {}).get('wave_height', 0), episode)

        if critic_losses:
            writer.add_scalar('Loss/Critic', critic_losses[-1], episode)
            writer.add_scalar('Loss/Actor', actor_losses[-1], episode)

        # 记录每个智能体的噪声水平
        for i, agent in enumerate(agents):
            writer.add_scalar(f'Agent_{i}/Noise_Scale', agent.noise_scale, episode)

        # 打印训练进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Oil Remaining: {info.get('oil_remaining', 0)}")
            print(f"  Weather: Wind {info.get('weather', {}).get('wind_speed', 0):.1f} m/s")
            if critic_losses:
                print(f"  Critic Loss: {critic_losses[-1]:.4f}")
                print(f"  Actor Loss: {actor_losses[-1]:.4f}")
            print("-" * 50)

        # 保存模型
        if episode % save_freq == 0 and episode > 0:
            save_enhanced_models(agents, episode, config)

        # 环境可视化（每100个episode）
        if episode % 100 == 0:
            env.render()

    # 保存最终模型
    save_enhanced_models(agents, total_episodes, config)

    # 绘制训练曲线
    plot_enhanced_training_curves(episode_rewards, critic_losses, actor_losses)


def plot_enhanced_training_curves(rewards, critic_losses, actor_losses, save_path='training_curves.png'):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 3, 2)
    plt.plot(critic_losses)
    plt.title('Critic Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 3)
    plt.plot(actor_losses)
    plt.title('Actor Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_enhanced_models(agents, episode, config):
    save_dir = config.get('save_dir', 'models')
    os.makedirs(save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), os.path.join(save_dir, f'actor_agent_{i}_ep{episode}.pth'))
        torch.save(agent.critic.state_dict(), os.path.join(save_dir, f'critic_agent_{i}_ep{episode}.pth'))


if __name__ == "__main__":
    config = {
        'num_ships': 5,
        'map_size': (100, 100),
        'max_steps': 200,
        'agent_types': ['cleanup_ship', 'containment_ship', 'dispersant_ship', 'cleanup_ship', 'containment_ship'],
        'buffer_size': 200000,
        'total_episodes': 200,
        'batch_size': 128,
        'update_freq': 10,
        'save_freq': 50,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,

    }
    train_enhanced_maddpg(config)