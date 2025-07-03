import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Union
import pickle
import logging
import os
import json
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
import imageio

warnings.filterwarnings('ignore')

# TensorBoard 相关
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available, using basic logging")

# PyGNOME 集成
try:
    import gnome
    from gnome import scripting
    from gnome.model import Model
    from gnome.maps import MapFromBNA
    from gnome.environment import Wind, Tide, WaterTemp
    from gnome.spill import point_line_spill
    from gnome.movers import RandomMover, WindMover, CatsMover
    from gnome.weatherers import Evaporation, Dispersion

    GNOME_AVAILABLE = True
    print("PyGNOME successfully imported")
except ImportError:
    GNOME_AVAILABLE = False
    print("PyGNOME not available, using simplified oil spill simulation")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 全局配置
GLOBAL_CONFIG = {
    'use_attention': True,
    'use_gnn': True,
    'use_temporal': True,
    'noise_decay': True,
    'lr_scheduling': True,
    'tensorboard_logging': TENSORBOARD_AVAILABLE,
    'parallel_envs': True,
    'shaped_rewards': True,
    'heterogeneous_agents': True,
    'random_weather': True
}

# 经验回放缓冲区
Experience = namedtuple('Experience', ['states', 'actions', 'rewards', 'next_states', 'dones', 'info'])


class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0

    def push(self, states, actions, rewards, next_states, dones, info=None):
        self.buffer.append(Experience(states, actions, rewards, next_states, dones, info))
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            return None

        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states = torch.stack([torch.tensor(e.states, dtype=torch.float32) for e in batch])
        actions = torch.stack([torch.tensor(e.actions, dtype=torch.float32) for e in batch])
        rewards = torch.stack([torch.tensor(e.rewards, dtype=torch.float32) for e in batch])
        next_states = torch.stack([torch.tensor(e.next_states, dtype=torch.float32) for e in batch])
        dones = torch.stack([torch.tensor(e.dones, dtype=torch.bool) for e in batch])

        return states, actions, rewards, next_states, dones, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)


# 注意力机制
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(output)


# 增强的图神经网络编码器
class EnhancedGNNEncoder(nn.Module):
    """增强的图神经网络编码器，集成注意力机制"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=0.1))
        for _ in range(num_layers - 2):
            self.gat_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.1))
        self.gat_layers.append(GATConv(hidden_dim * 4, output_dim, heads=1, dropout=0.1))

        self.attention = MultiHeadAttention(output_dim, num_heads=8) if GLOBAL_CONFIG['use_attention'] else None
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim * 4) for _ in range(num_layers - 1)])
        self.layer_norms.append(nn.LayerNorm(output_dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, edge_index):
        for i, gat_layer in enumerate(self.gat_layers[:-1]):
            residual = x if x.size(-1) == self.gat_layers[i + 1].in_channels else None
            x = F.elu(gat_layer(x, edge_index))
            x = self.layer_norms[i](x)
            x = self.dropout(x)
            if residual is not None and residual.size() == x.size():
                x = x + residual

        x = self.gat_layers[-1](x, edge_index)
        x = self.layer_norms[-1](x)

        if self.attention is not None:
            x = x.unsqueeze(0)
            x = self.attention(x, x, x)
            x = x.squeeze(0)

        return x


# 增强的时间序列编码器
class EnhancedTemporalEncoder(nn.Module):
    """增强的时间序列编码器，使用 Transformer"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1, bidirectional=True)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim * 2,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(2)
        ])
        self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        transformer_out = lstm_out
        for transformer_layer in self.transformer_layers:
            transformer_out = transformer_layer(transformer_out)
        output = self.output_projection(transformer_out[:, -1, :])
        return output


# 异构智能体类型
class AgentType:
    CLEANUP_VESSEL = "cleanup_vessel"
    BARRIER_VESSEL = "barrier_vessel"
    MONITORING_VESSEL = "monitoring_vessel"
    SKIMMER_VESSEL = "skimmer_vessel"
    DISPERSANT_VESSEL = "dispersant_vessel"


# 增强的 Actor 网络
class EnhancedActor(nn.Module):
    """增强的 Actor 网络，支持异构智能体和多种编码器"""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, agent_type: str,
                 use_gnn: bool = True, use_temporal: bool = True):
        super().__init__()
        self.max_action = max_action
        self.state_dim = state_dim
        self.agent_type = agent_type
        self.use_gnn = use_gnn
        self.use_temporal = use_temporal

        feature_dim = 256
        if use_gnn and GLOBAL_CONFIG['use_gnn']:
            self.gnn_encoder = EnhancedGNNEncoder(state_dim, 128, 256)
            feature_dim += 256
        else:
            self.gnn_encoder = None

        if use_temporal and GLOBAL_CONFIG['use_temporal']:
            self.temporal_encoder = EnhancedTemporalEncoder(state_dim, 128)
            feature_dim += 128
        else:
            self.temporal_encoder = None

        self.base_feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        agent_specific_dim = self._get_agent_specific_dim()
        self.agent_specific_layer = nn.Linear(feature_dim, agent_specific_dim)

        self.policy_net = nn.Sequential(
            nn.Linear(agent_specific_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        if GLOBAL_CONFIG['use_attention']:
            self.feature_attention = MultiHeadAttention(feature_dim, num_heads=8)
        else:
            self.feature_attention = None

    def _get_agent_specific_dim(self):
        type_dims = {
            AgentType.CLEANUP_VESSEL: 128,
            AgentType.BARRIER_VESSEL: 96,
            AgentType.MONITORING_VESSEL: 64,
            AgentType.SKIMMER_VESSEL: 112,
            AgentType.DISPERSANT_VESSEL: 80
        }
        return type_dims.get(self.agent_type, 128)

    def forward(self, state, edge_index=None, temporal_seq=None):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        features = []
        base_features = self.base_feature_extractor(state)
        features.append(base_features)

        if self.gnn_encoder is not None and edge_index is not None:
            gnn_features = self.gnn_encoder(state, edge_index)
            if len(gnn_features.shape) == 2 and len(base_features.shape) == 2:
                features.append(gnn_features)

        if self.temporal_encoder is not None and temporal_seq is not None:
            temporal_features = self.temporal_encoder(temporal_seq)
            features.append(temporal_features)

        if len(features) > 1:
            combined_features = torch.cat(features, dim=-1)
        else:
            combined_features = features[0]

        if self.feature_attention is not None:
            combined_features = self.feature_attention(combined_features, combined_features, combined_features)

        agent_features = self.agent_specific_layer(combined_features)
        action = self.policy_net(agent_features)
        return self.max_action * action


# 增强的 Critic 网络
class EnhancedCritic(nn.Module):
    """增强的 Critic 网络"""

    def __init__(self, state_dim: int, action_dim: int, num_agents: int):
        super().__init__()
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents

        self.q1_net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, states, actions):
        states_flat = states.view(states.size(0), -1)
        actions_flat = actions.view(actions.size(0), -1)
        x = torch.cat([states_flat, actions_flat], dim=-1)
        q1 = self.q1_net(x)
        q2 = self.q2_net(x)
        return q1, q2


# PyGNOME 模拟器
class PyGNOMESimulator:
    """PyGNOME 油污扩散模拟器"""

    def __init__(self, config: dict):
        self.config = config
        self.map_bounds = config.get('map_bounds', (-125, -120, 47, 50))
        self.model = None
        self.spill = None
        self._initialize_model()

    def _initialize_model(self):
        if not GNOME_AVAILABLE:
            return
        try:
            self.model = Model(start_time=datetime.now(), duration=self.config.get('duration_hours', 24) * 3600,
                               time_step=self.config.get('time_step', 3600))
            wind = Wind(filename=None, timeseries=[(datetime.now(), (10, 45))])
            self.model.environment += wind
            random_mover = RandomMover(diffusion_coeff=50000)
            wind_mover = WindMover(wind)
            self.model.movers += [random_mover, wind_mover]
            spill_location = (self.config.get('spill_lon', -122.5), self.config.get('spill_lat', 48.5))
            self.spill = point_line_spill(num_elements=1000, start_position=spill_location,
                                          release_time=datetime.now(), amount=self.config.get('spill_amount', 1000),
                                          units='barrels')
            self.model.spills += self.spill
        except Exception as e:
            print(f"PyGNOME initialization failed: {e}")
            self.model = None

    def step(self, current_time_step: int):
        if self.model is None:
            return self._fallback_simulation(current_time_step)
        try:
            self.model.step()
            return self._extract_spill_data()
        except Exception as e:
            print(f"PyGNOME simulation step failed: {e}")
            return self._fallback_simulation(current_time_step)

    def _extract_spill_data(self):
        if self.spill is None:
            return []
        positions = []
        for element in self.spill:
            positions.append((element['longitude'], element['latitude']))
        return positions

    def _fallback_simulation(self, current_time_step: int):
        base_positions = [(self.config.get('spill_lon', -122.5), self.config.get('spill_lat', 48.5))]
        num_particles = min(100 + current_time_step * 10, 1000)
        positions = []
        for i in range(num_particles):
            noise_x = np.random.normal(0, 0.01 * current_time_step)
            noise_y = np.random.normal(0, 0.01 * current_time_step)
            pos_x = base_positions[0][0] + noise_x
            pos_y = base_positions[0][1] + noise_y
            positions.append((pos_x, pos_y))
        return positions


# 增强的环境
class EnhancedOilSpillEnvironment:
    """增强的溢油扩散环境，集成 PyGNOME 和现实约束"""

    def __init__(self, config: dict):
        self.config = config
        self.num_ships = config.get('num_ships', 5)
        self.map_size = config.get('map_size', (100, 100))
        self.max_steps = config.get('max_steps', 200)
        self.agent_types = self._assign_agent_types()
        self.gnome_simulator = PyGNOMESimulator(config) if GNOME_AVAILABLE else None
        self.state_dim = 256
        self.action_dim = 6
        self.current_step = 0
        self.oil_particles = []
        self.ship_positions = []
        self.ship_histories = []
        self.oil_histories = []
        self.weather_conditions = {}
        self.cleanup_efficiency = {}
        self.fuel_levels = []
        self.equipment_status = []
        self.communication_range = config.get('communication_range', 20)
        self._initialize_environment()

    def _assign_agent_types(self):
        if not GLOBAL_CONFIG['heterogeneous_agents']:
            return [AgentType.CLEANUP_VESSEL] * self.num_ships
        types = [
            AgentType.CLEANUP_VESSEL,
            AgentType.BARRIER_VESSEL,
            AgentType.MONITORING_VESSEL,
            AgentType.SKIMMER_VESSEL,
            AgentType.DISPERSANT_VESSEL
        ]
        agent_types = [AgentType.CLEANUP_VESSEL]
        for i in range(1, self.num_ships):
            agent_types.append(np.random.choice(types))
        return agent_types

    def _initialize_environment(self):
        self.ship_positions = [
            np.random.uniform([10, 10], [self.map_size[0] - 10, self.map_size[1] - 10])
            for _ in range(self.num_ships)
        ]
        self.ship_histories = [[] for _ in range(self.num_ships)]
        self.oil_histories = []
        self.fuel_levels = [np.random.uniform(0.7, 1.0) for _ in range(self.num_ships)]
        self.equipment_status = [np.random.uniform(0.8, 1.0) for _ in range(self.num_ships)]
        self._update_weather_conditions()
        self.cleanup_efficiency = {i: 0.0 for i in range(self.num_ships)}
        if self.gnome_simulator:
            self.oil_particles = self.gnome_simulator.step(0)
        else:
            spill_center = (self.map_size[0] // 2, self.map_size[1] // 2)
            self.oil_particles = [spill_center + np.random.normal(0, 2, 2) for _ in range(100)]

    def _update_weather_conditions(self):
        if GLOBAL_CONFIG['random_weather']:
            self.weather_conditions = {
                'wind_speed': np.random.uniform(0, 15),
                'wind_direction': np.random.uniform(0, 360),
                'wave_height': np.random.uniform(0, 3),
                'current_speed': np.random.uniform(0, 2),
                'current_direction': np.random.uniform(0, 360),
                'visibility': np.random.uniform(0.5, 10),
                'temperature': np.random.uniform(-5, 30)
            }
        else:
            self.weather_conditions = {
                'wind_speed': 5.0,
                'wind_direction': 45.0,
                'wave_height': 1.0,
                'current_speed': 0.5,
                'current_direction': 90.0,
                'visibility': 8.0,
                'temperature': 15.0
            }

    def reset(self):
        self.current_step = 0
        self._initialize_environment()
        return self._get_observations()

    def step(self, actions):
        self.current_step += 1
        if self.current_step % 10 == 0:
            self._update_weather_conditions()
        for i, pos in enumerate(self.ship_positions):
            self.ship_histories[i].append(pos.copy())
        rewards = self._update_ships_and_calculate_rewards(actions)
        self._update_oil_spill()
        self.oil_histories.append([pos.copy() for pos in self.oil_particles])
        done = self._check_termination()
        observations = self._get_observations()
        info = self._get_info()
        return observations, rewards, [done] * self.num_ships, info

    def _update_ships_and_calculate_rewards(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            dx, dy, speed, operation_intensity, fuel_usage, equipment_usage = action
            actual_speed = speed * self.fuel_levels[i] * self.equipment_status[i]
            weather_factor = self._get_weather_impact_factor()
            actual_speed *= weather_factor
            self.ship_positions[i][0] += dx * actual_speed
            self.ship_positions[i][1] += dy * actual_speed
            self.ship_positions[i] = np.clip(self.ship_positions[i], [0, 0], [self.map_size[0], self.map_size[1]])
            self.fuel_levels[i] -= fuel_usage * 0.01
            self.fuel_levels[i] = max(0, self.fuel_levels[i])
            self.equipment_status[i] -= equipment_usage * 0.005
            self.equipment_status[i] = max(0.1, self.equipment_status[i])
            self._update_cleanup_efficiency(i, operation_intensity)
            reward = self._calculate_shaped_reward(i, action)
            rewards.append(reward)
        return rewards

    def _get_weather_impact_factor(self):
        wind_impact = max(0.3, 1.0 - self.weather_conditions['wind_speed'] / 20.0)
        wave_impact = max(0.3, 1.0 - self.weather_conditions['wave_height'] / 5.0)
        visibility_impact = max(0.5, self.weather_conditions['visibility'] / 10.0)
        return wind_impact * wave_impact * visibility_impact

    def _update_cleanup_efficiency(self, agent_id: int, operation_intensity: float):
        agent_type = self.agent_types[agent_id]
        base_efficiency = {
            AgentType.CLEANUP_VESSEL: 0.8,
            AgentType.BARRIER_VESSEL: 0.3,
            AgentType.MONITORING_VESSEL: 0.1,
            AgentType.SKIMMER_VESSEL: 0.9,
            AgentType.DISPERSANT_VESSEL: 0.6
        }
        efficiency = base_efficiency.get(agent_type, 0.5)
        efficiency *= operation_intensity
        efficiency *= self.equipment_status[agent_id]
        efficiency *= self._get_weather_impact_factor()
        self.cleanup_efficiency[agent_id] = efficiency

    def _update_oil_spill(self):
        if self.gnome_simulator:
            self.oil_particles = self.gnome_simulator.step(self.current_step)
        else:
            self._simulate_oil_spread_fallback()
        self._apply_cleanup_effects()

    def _simulate_oil_spread_fallback(self):
        new_particles = []
        for particle in self.oil_particles:
            wind_effect = np.array([
                self.weather_conditions['wind_speed'] * np.cos(np.radians(self.weather_conditions['wind_direction'])),
                self.weather_conditions['wind_speed'] * np.sin(np.radians(self.weather_conditions['wind_direction']))
            ]) * 0.1
            current_effect = np.array([
                self.weather_conditions['current_speed'] * np.cos(
                    np.radians(self.weather_conditions['current_direction'])),
                self.weather_conditions['current_speed'] * np.sin(
                    np.radians(self.weather_conditions['current_direction']))
            ]) * 0.1
            random_diffusion = np.random.normal(0, 0.5, 2)
            new_pos = np.array(particle) + wind_effect + current_effect + random_diffusion
            new_pos = np.clip(new_pos, [0, 0], [self.map_size[0], self.map_size[1]])
            new_particles.append(new_pos)
        if self.current_step < 50:
            center = (self.map_size[0] // 2, self.map_size[1] // 2)
            for _ in range(5):
                new_particle = np.array(center) + np.random.normal(0, 1, 2)
                new_particles.append(new_particle)
        self.oil_particles = new_particles

    def _apply_cleanup_effects(self):
        remaining_particles = []
        for particle in self.oil_particles:
            cleaned = False
            for i, ship_pos in enumerate(self.ship_positions):
                distance = np.linalg.norm(np.array(particle) - np.array(ship_pos))
                cleanup_range = self._get_cleanup_range(i)
                if distance <= cleanup_range:
                    if np.random.random() < self.cleanup_efficiency[i]:
                        cleaned = True
                        break
            if not cleaned:
                remaining_particles.append(particle)
        self.oil_particles = remaining_particles

    def _get_cleanup_range(self, agent_id: int):
        agent_type = self.agent_types[agent_id]
        base_ranges = {
            AgentType.CLEANUP_VESSEL: 5.0,
            AgentType.BARRIER_VESSEL: 8.0,
            AgentType.MONITORING_VESSEL: 15.0,
            AgentType.SKIMMER_VESSEL: 6.0,
            AgentType.DISPERSANT_VESSEL: 10.0
        }
        return base_ranges.get(agent_type, 5.0)

    def _calculate_shaped_reward(self, agent_id: int, action):
        reward = 0.0
        ship_pos = self.ship_positions[agent_id]
        oil_cleanup_reward = 0.0
        if self.oil_particles:
            distances = [np.linalg.norm(np.array(oil) - np.array(ship_pos)) for oil in self.oil_particles]
            min_distance = min(distances)
            oil_cleanup_reward += max(0, 20 - min_distance) * 0.2
            oil_cleanup_reward += self.cleanup_efficiency[agent_id] * 10
        reward += oil_cleanup_reward

        cooperation_reward = 0.0
        for j, other_pos in enumerate(self.ship_positions):
            if j != agent_id:
                distance = np.linalg.norm(np.array(ship_pos) - np.array(other_pos))
                if 8 < distance < 20:
                    cooperation_reward += 0.3
                elif distance < 3:
                    cooperation_reward -= 1.0
        reward += cooperation_reward

        resource_reward = 0.0
        if self.fuel_levels[agent_id] > 0.2:
            resource_reward += 0.1
        else:
            resource_reward -= 0.5
        if self.equipment_status[agent_id] > 0.6:
            resource_reward += 0.1
        else:
            resource_reward -= 0.3
        reward += resource_reward

        agent_specific_reward = self._get_agent_specific_reward(agent_id, action)
        reward += agent_specific_reward

        weather_adaptation_reward = 0.0
        if self.weather_conditions['wind_speed'] > 10:
            if action[2] < 0.5:
                weather_adaptation_reward += 0.2
        reward += weather_adaptation_reward

        global_reward = 0.0
        oil_remaining_ratio = len(self.oil_particles) / max(1,
                                                            len(self.oil_histories[0]) if self.oil_histories else 100)
        global_reward += (1.0 - oil_remaining_ratio) * 5.0
        reward += global_reward

        return reward

    def _get_agent_specific_reward(self, agent_id: int, action):
        agent_type = self.agent_types[agent_id]
        reward = 0.0
        if agent_type == AgentType.CLEANUP_VESSEL:
            reward += self.cleanup_efficiency[agent_id] * 2.0
        elif agent_type == AgentType.BARRIER_VESSEL:
            if self.oil_particles:
                ship_pos = self.ship_positions[agent_id]
                nearby_oil = sum(1 for oil in self.oil_particles
                                 if np.linalg.norm(np.array(oil) - np.array(ship_pos)) < 10)
                reward += nearby_oil * 0.3
        elif agent_type == AgentType.MONITORING_VESSEL:
            coverage_area = action[2] * 15
            reward += coverage_area * 0.1
        elif agent_type == AgentType.SKIMMER_VESSEL:
            reward += self.cleanup_efficiency[agent_id] * 2.5
        elif agent_type == AgentType.DISPERSANT_VESSEL:
            dispersant_efficiency = action[3] * self.equipment_status[agent_id]
            reward += dispersant_efficiency * 1.5
        return reward

    def _get_observations(self):
        observations = []
        for i in range(self.num_ships):
            obs = []
            ship_state = list(self.ship_positions[i])
            obs.extend(ship_state)
            agent_type_encoding = [0.0] * 5
            type_index = list(AgentType.__dict__.values()).index(self.agent_types[i])
            if type_index < 5:
                agent_type_encoding[type_index] = 1.0
            obs.extend(agent_type_encoding)
            obs.extend([self.fuel_levels[i], self.equipment_status[i]])
            for j in range(self.num_ships):
                if i != j:
                    rel_pos = np.array(self.ship_positions[j]) - np.array(self.ship_positions[i])
                    obs.extend(rel_pos.tolist())
                    other_type_encoding = [0.0] * 5
                    other_type_index = list(AgentType.__dict__.values()).index(self.agent_types[j])
                    if other_type_index < 5:
                        other_type_encoding[other_type_index] = 1.0
                    obs.extend(other_type_encoding)
                else:
                    obs.extend([0.0, 0.0])
                    obs.extend([0.0] * 5)
            if self.oil_particles:
                distances = [np.linalg.norm(np.array(oil) - np.array(self.ship_positions[i]))
                             for oil in self.oil_particles]
                nearest_indices = np.argsort(distances)[:10]
                for idx in nearest_indices:
                    oil_pos = self.oil_particles[idx]
                    rel_oil_pos = np.array(oil_pos) - np.array(self.ship_positions[i])
                    obs.extend(rel_oil_pos.tolist())
                    obs.append(distances[idx])
                while len(nearest_indices) < 10:
                    obs.extend([0.0, 0.0, 100.0])
            else:
                obs.extend([0.0] * 30)
            weather_obs = [
                self.weather_conditions['wind_speed'] / 20.0,
                self.weather_conditions['wind_direction'] / 360.0,
                self.weather_conditions['wave_height'] / 5.0,
                self.weather_conditions['current_speed'] / 5.0,
                self.weather_conditions['current_direction'] / 360.0,
                self.weather_conditions['visibility'] / 10.0,
                (self.weather_conditions['temperature'] + 10) / 40.0
            ]
            obs.extend(weather_obs)
            obs.append(self.current_step / self.max_steps)
            obs.append(len(self.oil_particles) / 1000.0)
            obs.append(self.cleanup_efficiency[i])
            if len(self.ship_histories[i]) >= 3:
                recent_positions = self.ship_histories[i][-3:]
                for pos in recent_positions:
                    obs.extend(pos.tolist())
            else:
                for _ in range(3):
                    obs.extend([0.0, 0.0])
            current_dim = len(obs)
            if current_dim < self.state_dim:
                obs.extend([0.0] * (self.state_dim - current_dim))
            elif current_dim > self.state_dim:
                obs = obs[:self.state_dim]
            observations.append(obs)
        return np.array(observations)

    def _get_info(self):
        return {
            'oil_particles_count': len(self.oil_particles),
            'total_cleanup_efficiency': sum(self.cleanup_efficiency.values()),
            'average_fuel_level': np.mean(self.fuel_levels),
            'average_equipment_status': np.mean(self.equipment_status),
            'weather_conditions': self.weather_conditions.copy(),
            'ship_positions': [pos.copy() for pos in self.ship_positions]
        }

    def _check_termination(self):
        if self.current_step >= self.max_steps:
            return True
        if len(self.oil_particles) < 10:
            return True
        if all(fuel < 0.05 for fuel in self.fuel_levels):
            return True
        return False

    def get_trajectory_data(self):
        return {
            'ship_histories': self.ship_histories,
            'oil_histories': self.oil_histories,
            'agent_types': self.agent_types,
            'map_size': self.map_size
        }


# 并行环境管理器
class ParallelEnvironmentManager:
    """并行环境管理器"""

    def __init__(self, config: dict, num_envs: int = 4):
        self.config = config
        self.num_envs = num_envs
        self.envs = []
        for i in range(num_envs):
            env_config = config.copy()
            env_config['env_id'] = i
            self.envs.append(EnhancedOilSpillEnvironment(env_config))

    def reset_all(self):
        return [env.reset() for env in self.envs]

    def step_all(self, actions_list):
        results = []
        for env, actions in zip(self.envs, actions_list):
            results.append(env.step(actions))
        return results

    def get_trajectory_data_all(self):
        return [env.get_trajectory_data() for env in self.envs]


# 增强的 MADDPG Agent
class EnhancedMADDPGAgent:
    """增强的 MADDPG 智能体"""

    def __init__(self, agent_id: int, state_dim: int, action_dim: int,
                 num_agents: int, agent_type: str, max_action: float = 1.0,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 tau: float = 0.005, gamma: float = 0.99):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma
        self.agent_type = agent_type
        self.actor = EnhancedActor(state_dim, action_dim, max_action, agent_type).to(device)
        self.actor_target = EnhancedActor(state_dim, action_dim, max_action, agent_type).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic = EnhancedCritic(state_dim, action_dim, num_agents).to(device)
        self.critic_target = EnhancedCritic(state_dim, action_dim, num_agents).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.actor_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99)
        self.critic_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        self.noise_scale = 0.2
        self.noise_decay = 0.999
        self.min_noise = 0.01
        self.update_count = 0

    def select_action(self, state, edge_index=None, temporal_seq=None, add_noise=True):
        if isinstance(state, list):
            state = np.array(state)
        state = torch.FloatTensor(state).to(device)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if temporal_seq is not None:
            temporal_seq = torch.FloatTensor(temporal_seq).to(device)
            if len(temporal_seq.shape) == 2:
                temporal_seq = temporal_seq.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state, edge_index, temporal_seq).cpu().numpy()
            if len(action.shape) > 1:
                action = action[0]
        if add_noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
            if GLOBAL_CONFIG['noise_decay']:
                self.noise_scale = max(self.min_noise, self.noise_scale * self.noise_decay)
        return action

    def update(self, replay_buffer, batch_size: int, other_agents: List['EnhancedMADDPGAgent']):
        if len(replay_buffer) < batch_size:
            return None, None
        sample_result = replay_buffer.sample(batch_size)
        if sample_result is None:
            return None, None
        if len(sample_result) == 7:
            states, actions, rewards, next_states, dones, indices, weights = sample_result
            weights = weights.to(device)
        else:
            states, actions, rewards, next_states, dones = sample_result
            weights = torch.ones(batch_size).to(device)
            indices = None
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)
        agent_rewards = rewards[:, self.agent_id].unsqueeze(1)
        agent_dones = dones[:, self.agent_id].unsqueeze(1)
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate([self] + other_agents):
                if i == self.agent_id:
                    continue
                next_action = agent.actor_target(next_states[:, i])
                next_actions.append(next_action)
            next_actions.insert(self.agent_id, self.actor_target(next_states[:, self.agent_id]))
            next_actions = torch.stack(next_actions, dim=1)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = agent_rewards + (self.gamma * target_q * (~agent_dones).float())
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss1 = F.mse_loss(current_q1, target_q, reduction='none')
        critic_loss2 = F.mse_loss(current_q2, target_q, reduction='none')
        critic_loss = (critic_loss1 + critic_loss2) * weights
        critic_loss = critic_loss.mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        actor_loss = None
        if self.update_count % 2 == 0:
            actions_pred = actions.clone()
            agent_state = states[:, self.agent_id]
            predicted_action = self.actor(agent_state)
            actions_pred[:, self.agent_id] = predicted_action
            q1, q2 = self.critic(states, actions_pred)
            actor_loss = -(torch.min(q1, q2) * weights).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic_target, self.critic)
            actor_loss = actor_loss.item()
        if indices is not None and hasattr(replay_buffer, 'update_priorities'):
            td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy()
            replay_buffer.update_priorities(indices, td_errors + 1e-6)
        self.update_count += 1
        if GLOBAL_CONFIG['lr_scheduling'] and self.update_count % 100 == 0:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        return critic_loss.item(), actor_loss

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


# 训练主函数
def train_enhanced_maddpg(config: dict):
    """增强的 MADDPG 训练主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if GLOBAL_CONFIG['parallel_envs']:
        env_manager = ParallelEnvironmentManager(config, num_envs=config.get('num_envs', 4))
        env = env_manager.envs[0]
    else:
        env = EnhancedOilSpillEnvironment(config)
        env_manager = None

    agents = []
    for i in range(env.num_ships):
        agent = EnhancedMADDPGAgent(
            agent_id=i,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            num_agents=env.num_ships,
            agent_type=env.agent_types[i],
            max_action=1.0,
            lr_actor=config.get('learning_rate_actor', 1e-4),
            lr_critic=config.get('learning_rate_critic', 1e-3)
        )
        agents.append(agent)

    replay_buffer = PrioritizedReplayBuffer(config.get('buffer_size', 100000))
    writer = None
    if GLOBAL_CONFIG['tensorboard_logging']:
        log_dir = f"runs/enhanced_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir)

    total_episodes = config.get('total_episodes', 2000)
    batch_size = config.get('batch_size', 64)
    update_freq = config.get('update_freq', 100)
    episode_rewards = []
    critic_losses = []
    actor_losses = []
    oil_cleanup_rates = []
    trajectories = []

    print("开始增强 MADDPG 训练...")
    print(f"配置: {GLOBAL_CONFIG}")

    for episode in range(total_episodes):
        if env_manager:
            states_list = env_manager.reset_all()
            episode_rewards_parallel = [0] * len(states_list)
            episode_trajectory = {'ships': [[] for _ in range(env.num_ships)], 'oil': [],
                                  'agent_types': env.agent_types}
        else:
            states = env.reset()
            episode_reward = 0
            episode_trajectory = {'ships': [[] for _ in range(env.num_ships)], 'oil': [],
                                  'agent_types': env.agent_types}

        step_count = 0

        while True:
            # 生成全连接图的 edge_index
            edge_index = torch.tensor([[i, j] for i in range(env.num_ships) for j in range(env.num_ships) if i != j],
                                      dtype=torch.long).t().contiguous().to(device)

            if env_manager:
                actions_list = []
                for env_idx, states in enumerate(states_list):
                    actions = []
                    for i, agent in enumerate(agents):
                        temporal_seq = states  # 使用整个状态序列
                        action = agent.select_action(states[i], edge_index, temporal_seq)
                        actions.append(action)
                    actions_list.append(actions)

                results = env_manager.step_all(actions_list)

                for env_idx, (next_states, rewards, dones, info) in enumerate(results):
                    replay_buffer.push(states_list[env_idx], actions_list[env_idx], rewards, next_states, dones, info)
                    episode_rewards_parallel[env_idx] += np.sum(rewards)
                    if episode == total_episodes - 1:
                        for i in range(env.num_ships):
                            episode_trajectory['ships'][i].append(info['ship_positions'][i].copy())
                        episode_trajectory['oil'].append([p.copy() for p in info['ship_positions']])
                    if all(dones):
                        states_list[env_idx] = env_manager.envs[env_idx].reset()
                    else:
                        states_list[env_idx] = next_states

                if all(all(env_manager.envs[i]._check_termination()) for i in range(len(env_manager.envs))):
                    break
            else:
                actions = []
                for i, agent in enumerate(agents):
                    temporal_seq = states
                    action = agent.select_action(states[i], edge_index, temporal_seq)
                    actions.append(action)

                next_states, rewards, dones, info = env.step(actions)
                replay_buffer.push(states, actions, rewards, next_states, dones, info)
                episode_reward += np.sum(rewards)
                if episode == total_episodes - 1:
                    for i in range(env.num_ships):
                        episode_trajectory['ships'][i].append(info['ship_positions'][i].copy())
                    episode_trajectory['oil'].append([p.copy() for p in info['ship_positions']])
                states = next_states
                step_count += 1
                if all(dones):
                    break

            if len(replay_buffer) > batch_size and step_count % update_freq == 0:
                batch_critic_losses = []
                batch_actor_losses = []
                for i, agent in enumerate(agents):
                    other_agents = [agents[j] for j in range(len(agents)) if j != i]
                    c_loss, a_loss = agent.update(replay_buffer, batch_size, other_agents)
                    if c_loss is not None:
                        batch_critic_losses.append(c_loss)
                        if a_loss is not None:
                            batch_actor_losses.append(a_loss)
                if batch_critic_losses:
                    critic_losses.extend(batch_critic_losses)
                if batch_actor_losses:
                    actor_losses.extend(batch_actor_losses)

        if env_manager:
            avg_episode_reward = np.mean(episode_rewards_parallel)
            episode_rewards.append(avg_episode_reward)
            cleanup_rate = np.mean([len(env.get_trajectory_data()['oil_histories'][-1])
                                    if env.get_trajectory_data()['oil_histories'] else 100
                                    for env in env_manager.envs]) / max(1, len(
                env_manager.envs[0].get_trajectory_data()['oil_histories'][0]) if
            env_manager.envs[0].get_trajectory_data()['oil_histories'] else 100)
        else:
            episode_rewards.append(episode_reward)
            cleanup_rate = len(env.oil_particles) / max(1, len(env.oil_histories[0]) if env.oil_histories else 100)
        oil_cleanup_rates.append(1.0 - cleanup_rate)

        if episode == total_episodes - 1:
            trajectories.append(episode_trajectory)

        if writer and episode % 10 == 0:
            writer.add_scalar('Episode/Reward', episode_rewards[-1], episode)
            if critic_losses:
                writer.add_scalar('Loss/Critic', np.mean(critic_losses[-100:]), episode)
            if actor_losses:
                writer.add_scalar('Loss/Actor', np.mean(actor_losses[-100:]), episode)
            writer.add_scalar('Environment/Oil_Cleanup_Rate', oil_cleanup_rates[-1], episode)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if episode >= 50 else episode_rewards[-1]
            avg_cleanup_rate = np.mean(oil_cleanup_rates[-50:]) if episode >= 50 else oil_cleanup_rates[-1]
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Cleanup Rate: {avg_cleanup_rate:.2%}")

        if episode % 200 == 0:
            save_models(agents, episode)

    if writer:
        writer.close()
    if env_manager:
        env_manager = None  # 释放资源

    plot_training_curves(episode_rewards, critic_losses, actor_losses, oil_cleanup_rates)
    if trajectories:
        generate_gif(trajectories[-1], total_episodes - 1, config['map_size'])

    return agents, trajectories


def save_models(agents: List[EnhancedMADDPGAgent], episode: int):
    """保存模型"""
    try:
        save_dir = f"models/episode_{episode}"
        os.makedirs(save_dir, exist_ok=True)
        for i, agent in enumerate(agents):
            torch.save(agent.actor.state_dict(), f"{save_dir}/agent_{i}_actor.pth")
            torch.save(agent.critic.state_dict(), f"{save_dir}/agent_{i}_critic.pth")
        logging.info(f"模型已保存至 {save_dir}")
    except OSError as e:
        logging.error(f"模型保存失败: {e}")


def plot_training_curves(episode_rewards: List[float], critic_losses: List[float],
                         actor_losses: List[float], oil_cleanup_rates: List[float]):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].plot(episode_rewards)
    axes[0].set_title('Episode Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[1].plot(critic_losses)
    axes[1].set_title('Critic Loss')
    axes[1].set_xlabel('Update')
    axes[1].set_ylabel('Loss')
    axes[2].plot(actor_losses)
    axes[2].set_title('Actor Loss')
    axes[2].set_xlabel('Update')
    axes[2].set_ylabel('Loss')
    axes[3].plot(oil_cleanup_rates)
    axes[3].set_title('Oil Cleanup Rate')
    axes[3].set_xlabel('Episode')
    axes[3].set_ylabel('Cleanup Rate')
    plt.tight_layout()
    os.makedirs('viz', exist_ok=True)
    plt.savefig('viz/training_curves.png')
    plt.close()
    if GLOBAL_CONFIG['tensorboard_logging']:
        writer = SummaryWriter()
        writer.add_image('Training Curves', plt.imread('viz/training_curves.png'), 0)
        writer.close()


def generate_gif(trajectory: Dict, episode: int, map_size: Tuple[int, int]):
    """生成最后一个 episode 的 GIF"""
    images = []
    agent_colors = {
        AgentType.CLEANUP_VESSEL: 'blue',
        AgentType.BARRIER_VESSEL: 'green',
        AgentType.MONITORING_VESSEL: 'red',
        AgentType.SKIMMER_VESSEL: 'purple',
        AgentType.DISPERSANT_VESSEL: 'orange'
    }

    for step in range(len(trajectory['oil'])):
        plt.figure(figsize=(8, 8))
        plt.scatter([p[0] for p in trajectory['oil'][step]], [p[1] for p in trajectory['oil'][step]],
                    c='black', label='Oil Particles', s=10, alpha=0.5)
        for i, agent_type in enumerate(trajectory['agent_types']):
            pos = trajectory['ships'][i][step]
            color = agent_colors.get(agent_type, 'blue')
            plt.scatter(pos[0], pos[1], c=color, label=f'{agent_type}' if step == 0 else "", s=50)
        plt.xlim(0, map_size[0])
        plt.ylim(0, map_size[1])
        plt.legend()
        plt.title(f"Episode {episode}, Step {step}")
        os.makedirs('viz', exist_ok=True)
        plt.savefig(f'viz/episode_{episode}_step_{step}.png')
        images.append(imageio.imread(f'viz/episode_{episode}_step_{step}.png'))
        plt.close()

    imageio.mimsave(f'viz/episode_{episode}_trajectory.gif', images, fps=5)
    print(f"GIF 已保存至 viz/episode_{episode}_trajectory.gif")
    if GLOBAL_CONFIG['tensorboard_logging']:
        writer = SummaryWriter()
        writer.add_image('Trajectory GIF', plt.imread(f'viz/episode_{episode}_trajectory.gif'), 0)
        writer.close()


# 主训练配置和执行
if __name__ == "__main__":
    config = {
        'num_ships': 5,
        'map_size': (100, 100),
        'max_steps': 200,
        'num_envs': 4,
        'total_episodes': 2000,
        'batch_size': 64,
        'buffer_size': 100000,
        'update_freq': 100,
        'state_dim': 256,
        'action_dim': 6,
        'learning_rate_actor': 1e-4,
        'learning_rate_critic': 1e-3,
        'gamma': 0.99,
        'tau': 0.005,
        'spill_lon': -122.5,
        'spill_lat': 48.5,
        'spill_amount': 1000,
        'duration_hours': 24,
        'time_step': 3600,
        'communication_range': 20
    }

    print("海上溢油应急调度增强 MADDPG 训练系统")
    print("=" * 50)
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)

    agents, trajectories = train_enhanced_maddpg(config)

    print("训练完成！")
    print("模型已保存至 models 目录，训练曲线保存至 viz/training_curves.png")
    print(f"GIF 已保存至 viz/episode_{config['total_episodes'] - 1}_trajectory.gif")