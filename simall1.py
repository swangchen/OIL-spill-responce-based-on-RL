import numpy as np
import os

# ======= 可调参数区 =======
NUM_SAMPLES = 1000         # 生成样本数
SAVE_DIR = 'oil_spill_dataset'  # 保存目录
GRID_SIZE = 100            # 网格大小
TIME_STEPS = 50            # 扩散时长
INIT_THICKNESS_RANGE = (5, 20)  # 初始油污厚度范围
DIFFUSION_RATE_RANGE = (0.05, 0.25)  # 扩散率范围
WIND_STRENGTH_RANGE = (0.2, 1.0)     # 风速范围
CURRENT_STRENGTH_RANGE = (0.1, 0.5)  # 洋流速度范围
DRIFT_NOISE_STD = 0.1      # 漂移场扰动标准差
SEED = 42                  # 随机种子（如需复现）
# ==========================

np.random.seed(SEED)

def random_oil_sim(grid_size=GRID_SIZE, time_steps=TIME_STEPS):
    oil_map = np.zeros((time_steps, grid_size, grid_size))
    # 随机初始点
    source_x = np.random.randint(10, grid_size-10)
    source_y = np.random.randint(10, grid_size-10)
    initial_thickness = np.random.uniform(*INIT_THICKNESS_RANGE)
    oil_map[0, source_y, source_x] = initial_thickness

    # 随机扩散率
    diffusion_rate = np.random.uniform(*DIFFUSION_RATE_RANGE)
    # 随机风/流主方向和强度
    wind_strength = np.random.uniform(*WIND_STRENGTH_RANGE)
    wind_angle = np.random.uniform(0, 2*np.pi)
    main_wind = np.array([wind_strength * np.sin(wind_angle), wind_strength * np.cos(wind_angle)])
    # 洋流
    current_strength = np.random.uniform(*CURRENT_STRENGTH_RANGE)
    current_angle = np.random.uniform(0, 2*np.pi)
    main_current = np.array([current_strength * np.sin(current_angle), current_strength * np.cos(current_angle)])

    # 生成漂移场
    drift_field = np.zeros((grid_size, grid_size, 2))
    for y in range(grid_size):
        for x in range(grid_size):
            # 主风+主流+扰动
            drift_field[y, x, :] = main_wind + main_current + np.random.normal(0, DRIFT_NOISE_STD, 2)

    for t in range(1, time_steps):
        prev = oil_map[t - 1]
        curr = np.zeros_like(prev)
        # 扩散
        for y in range(1, grid_size - 1):
            for x in range(1, grid_size - 1):
                spread = diffusion_rate * prev[y, x]
                curr[y, x] += prev[y, x] - 4 * spread
                curr[y+1, x] += spread
                curr[y-1, x] += spread
                curr[y, x+1] += spread
                curr[y, x-1] += spread
        # 漂移
        drifted = np.zeros_like(curr)
        for y in range(grid_size):
            for x in range(grid_size):
                dy, dx = drift_field[y, x]
                new_y = int(round(y + dy))
                new_x = int(round(x + dx))
                if 0 <= new_y < grid_size and 0 <= new_x < grid_size:
                    drifted[new_y, new_x] += curr[y, x]
        oil_map[t] = drifted
    return oil_map

def generate_dataset(num_samples=NUM_SAMPLES, save_dir=SAVE_DIR, grid_size=GRID_SIZE, time_steps=TIME_STEPS):
    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_samples):
        oil_map = random_oil_sim(grid_size, time_steps)
        np.savez_compressed(os.path.join(save_dir, f'oil_{i:05d}.npz'), oil_map=oil_map)
        if i % 50 == 0:
            print(f"Generated {i} samples")

if __name__ == '__main__':
    generate_dataset() 