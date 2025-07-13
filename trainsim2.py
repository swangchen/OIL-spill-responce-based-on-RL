import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import glob

# ======= 可调参数区 =======
DATA_DIR = 'oil_spill_dataset'  # 数据目录
GRID_SIZE = 100
TIME_STEPS = 50
NUM_AGENTS = 3
LEARNING_RATE = 1e-3
EPISODE_LENGTH = 30
ALPHA = 1.0   # 污染控制效果权重
BETA = 0.1    # 资源调度成本权重
GAMMA = 0.5   # 响应延迟惩罚权重
# ==========================

print("Loading oil spill data...")
npz_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')))
oil_maps = []
for f in npz_files:
    data = np.load(f)
    oil_maps.append(data['oil_map'])
print(f"Loaded {len(oil_maps)} samples.")

class SimpleActor(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(SimpleActor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.network(x)

class SimpleCritic(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super(SimpleCritic, self).__init__()
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents
        self.network = nn.Sequential(
            nn.Linear(total_state_dim + total_action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, states, actions):
        states_flat = states.view(states.size(0), -1)
        actions_flat = actions.view(states.size(0), -1)
        combined = torch.cat([states_flat, actions_flat], dim=1)
        return self.network(combined)

class OilSpillEnvironment:
    def __init__(self, oil_maps, grid_size=GRID_SIZE, num_agents=NUM_AGENTS):
        self.oil_maps = oil_maps
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.current_step = 0
        self.current_oil_map = None
        self.agent_positions = []
        self.last_total_oil = None
        self.response_time = None  # 记录每个污染点被响应的时间
        self.reset()
    def reset(self):
        self.current_step = 0
        idx = np.random.randint(0, len(self.oil_maps))
        # 复制油污数据，使其可修改
        self.current_oil_map = self.oil_maps[idx].copy()
        self.agent_positions = []
        for _ in range(self.num_agents):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.agent_positions.append([x, y])
        self.last_total_oil = np.sum(self.current_oil_map[0])
        self.response_time = np.zeros((self.grid_size, self.grid_size))  # 记录每个格点最后被响应的时间
        return self._get_state()
    def _clean_oil_around(self, x, y, intensity):
        """清理智能体周围的油污"""
        radius = max(1, int(intensity * 3))  # 作业强度决定清理范围
        cleaned_amount = 0
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # 清理油污（减少浓度）
                    old_oil = self.current_oil_map[self.current_step, ny, nx]
                    clean_rate = intensity * 0.2  # 清理效率
                    self.current_oil_map[self.current_step, ny, nx] *= (1 - clean_rate)
                    self.current_oil_map[self.current_step, ny, nx] = max(0, self.current_oil_map[self.current_step, ny, nx])
                    cleaned_amount += old_oil - self.current_oil_map[self.current_step, ny, nx]
        return cleaned_amount
    def _get_state(self):
        states = []
        current_oil = self.current_oil_map[self.current_step]
        for i, pos in enumerate(self.agent_positions):
            state = []
            state.extend(pos)
            x, y = pos
            oil_around = 0
            count = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        oil_around += current_oil[ny, nx]
                        count += 1
            state.append(oil_around / max(count, 1))
            total_oil = np.sum(current_oil)
            state.append(total_oil / (self.grid_size * self.grid_size))
            state.append(self.current_step / TIME_STEPS)
            states.append(state)
        return np.array(states, dtype=np.float32)
    def step(self, actions):
        rewards = []
        prev_total_oil = self.last_total_oil
        # 智能体移动和清理油污
        total_cleaned = 0
        resource_cost = 0
        for i, action in enumerate(actions):
            dx, dy, op_intensity = action
            old_x, old_y = self.agent_positions[i]
            new_x = old_x + int(dx * 2)
            new_y = old_y + int(dy * 2)
            new_x = np.clip(new_x, 0, self.grid_size - 1)
            new_y = np.clip(new_y, 0, self.grid_size - 1)
            move_dist = np.sqrt((new_x - old_x) ** 2 + (new_y - old_y) ** 2)
            resource_cost += move_dist + abs(op_intensity)
            self.agent_positions[i] = [new_x, new_y]
            
            # 清理油污
            if op_intensity > 0:
                cleaned = self._clean_oil_around(new_x, new_y, op_intensity)
                total_cleaned += cleaned
                # 响应时间记录
                if self.current_oil_map[self.current_step, new_y, new_x] > 0.01:
                    self.response_time[new_y, new_x] = self.current_step
        
        # 计算油污减少量（基于实际清理效果）
        current_total_oil = np.sum(self.current_oil_map[self.current_step])
        oil_reduction = prev_total_oil - current_total_oil
        self.last_total_oil = current_total_oil
        
        # 响应延迟惩罚（未被及时响应的污染点）
        delay_penalty = 0
        if self.current_step > 0:
            # 统计当前油污点中，长时间未被响应的格点
            mask = (self.current_oil_map[self.current_step] > 0.01) & (self.response_time < self.current_step - 5)
            delay_penalty = np.sum(mask)
        
        # 计算最终奖励
        reward = ALPHA * oil_reduction - BETA * resource_cost - GAMMA * delay_penalty
        rewards = [reward] * self.num_agents  # 所有智能体共享全局奖励
        
        self.current_step += 1
        done = self.current_step >= min(EPISODE_LENGTH, self.current_oil_map.shape[0])
        next_state = self._get_state() if not done else None
        return next_state, rewards, done
    def render(self):
        current_oil = self.current_oil_map[self.current_step-1]
        plt.figure(figsize=(10, 8))
        plt.imshow(current_oil, cmap='hot', alpha=0.7)
        for i, pos in enumerate(self.agent_positions):
            plt.scatter(pos[0], pos[1], c=f'C{i}', s=100, marker='^', label=f'Agent {i}')
        plt.title(f'Step {self.current_step}, Sample (random)')
        plt.colorbar(label='Oil Concentration')
        plt.legend()
        plt.show()

def train_one_episode():
    print("Starting training...")
    env = OilSpillEnvironment(oil_maps)
    state_dim = 5
    action_dim = 3
    agents = []
    for i in range(NUM_AGENTS):
        actor = SimpleActor(state_dim, action_dim)
        critic = SimpleCritic(state_dim, action_dim, NUM_AGENTS)
        actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
        critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)
        agents.append({
            'actor': actor,
            'critic': critic,
            'actor_optimizer': actor_optimizer,
            'critic_optimizer': critic_optimizer
        })
    state = env.reset()
    episode_rewards = []
    episode_actions = []
    for step in range(EPISODE_LENGTH):
        actions = []
        for i, agent in enumerate(agents):
            state_tensor = torch.FloatTensor(state[i]).unsqueeze(0)
            with torch.no_grad():
                action = agent['actor'](state_tensor).numpy()[0]
            actions.append(action)
        episode_actions.append(actions)
        next_state, rewards, done = env.step(actions)
        episode_rewards.append(rewards)
        for i, agent in enumerate(agents):
            # Critic update
            states_tensor = torch.FloatTensor(state).unsqueeze(0)
            actions_tensor = torch.FloatTensor(np.array(actions)).unsqueeze(0)
            value = agent['critic'](states_tensor, actions_tensor)
            reward_signal = rewards[i]
            critic_loss = nn.MSELoss()(value, torch.tensor([[reward_signal]], dtype=torch.float32))
            agent['critic_optimizer'].zero_grad()
            critic_loss.backward()
            agent['critic_optimizer'].step()
            # Actor update
            actions_pred = np.array(actions)
            state_tensor_i = torch.FloatTensor(state[i]).unsqueeze(0)
            actions_pred[i] = agent['actor'](state_tensor_i).detach().numpy()[0]
            actions_pred_tensor = torch.FloatTensor(actions_pred).unsqueeze(0)
            actor_loss = -agent['critic'](states_tensor, actions_pred_tensor).mean()
            agent['actor_optimizer'].zero_grad()
            actor_loss.backward()
            agent['actor_optimizer'].step()
        state = next_state
        if done:
            break
        if step % 5 == 0:
            print(f"Step {step}: Rewards = {rewards}, Total oil = {np.sum(env.current_oil_map[env.current_step-1]):.2f}")
    print(f"\nEpisode finished!")
    print(f"Average rewards per agent: {np.mean(episode_rewards, axis=0)}")
    print(f"Total episode reward: {np.sum(episode_rewards)}")
    env.render()
    episode_rewards = np.array(episode_rewards)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    for i in range(NUM_AGENTS):
        plt.plot(episode_rewards[:, i], label=f'Agent {i}')
    plt.title('Rewards per Agent')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(np.sum(episode_rewards, axis=1))
    plt.title('Total Reward')
    plt.xlabel('Step')
    plt.ylabel('Total Reward')
    plt.tight_layout()
    plt.savefig('training_results2.png')
    plt.show()
    return episode_rewards, episode_actions

if __name__ == '__main__':
    train_one_episode() 