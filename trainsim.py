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
# ==========================

# 批量加载所有npz文件
print("Loading oil spill data...")
npz_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npz')))
oil_maps = []
for f in npz_files:
    data = np.load(f)
    oil_maps.append(data['oil_map'])
    # print(f, data['oil_map'].shape)  # 可选：调试用，检查shape
print(f"Loaded {len(oil_maps)} samples.")

# 简单的Actor网络
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
        self.oil_maps = oil_maps  # list of oil_map arrays
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.current_step = 0
        self.current_oil_map = None
        self.agent_positions = []
        self.reset()
    def reset(self):
        self.current_step = 0
        idx = np.random.randint(0, len(self.oil_maps))
        self.current_oil_map = self.oil_maps[idx]
        self.agent_positions = []
        for _ in range(self.num_agents):
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            self.agent_positions.append([x, y])
        return self._get_state()
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
        for i, action in enumerate(actions):
            dx, dy, op_intensity = action
            new_x = self.agent_positions[i][0] + int(dx * 2)
            new_y = self.agent_positions[i][1] + int(dy * 2)
            new_x = np.clip(new_x, 0, self.grid_size - 1)
            new_y = np.clip(new_y, 0, self.grid_size - 1)
            self.agent_positions[i] = [new_x, new_y]
            current_oil = self.current_oil_map[self.current_step]
            oil_at_position = current_oil[new_y, new_x]
            reward = oil_at_position * op_intensity * 10
            rewards.append(reward)
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

            # Actor update: 重新 forward，且只对 actor 参与梯度
            actions_pred = np.array(actions)
            state_tensor_i = torch.FloatTensor(state[i]).unsqueeze(0)
            # 让当前 agent 的动作由 actor 网络输出，其余 agent 的动作保持原样
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
    plt.savefig('training_results.png')
    plt.show()
    return episode_rewards, episode_actions

if __name__ == '__main__':
    train_one_episode() 