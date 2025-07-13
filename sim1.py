import numpy as np
import matplotlib.pyplot as plt
import os

# 模拟参数
grid_size = 100        # 海面网格大小 100x100 km
time_steps = 50        # 总时间步数
oil_map = np.zeros((time_steps, grid_size, grid_size))

# 初始油污释放点
source_x, source_y = 50, 50
initial_thickness = 10.0
oil_map[0, source_y, source_x] = initial_thickness

# 模拟参数（扩散 + 漂移）
diffusion_rate = 0.22             # 油污在四个方向的扩散强度
drift_field = np.zeros((grid_size, grid_size, 2))  # (dy, dx) 漂移向量场
wind_strength = 0.4              # 控制漂移量大小

# 构造一个不规则漂移场：向右上角但有扰动（模拟风/流不均）
for y in range(grid_size):
    for x in range(grid_size):
        angle = np.pi / 4 + 0.2 * np.sin(y / 10) + 0.2 * np.cos(x / 15)
        drift_field[y, x, 0] = wind_strength * np.sin(angle)  # dy
        drift_field[y, x, 1] = wind_strength * np.cos(angle)  # dx

# 时间演化
for t in range(1, time_steps):
    prev = oil_map[t - 1]
    curr = np.zeros_like(prev)

    # 扩散过程（四邻域）
    for y in range(1, grid_size - 1):
        for x in range(1, grid_size - 1):
            spread = diffusion_rate * prev[y, x]
            curr[y, x] += prev[y, x] - 4 * spread
            curr[y+1, x] += spread
            curr[y-1, x] += spread
            curr[y, x+1] += spread
            curr[y, x-1] += spread

    # 漂移过程（使用向量场）
    drifted = np.zeros_like(curr)
    for y in range(grid_size):
        for x in range(grid_size):
            dy, dx = drift_field[y, x]
            new_y = int(round(y + dy))
            new_x = int(round(x + dx))
            if 0 <= new_y < grid_size and 0 <= new_x < grid_size:
                drifted[new_y, new_x] += curr[y, x]
    oil_map[t] = drifted

# 保存为 npz 文件
np.savez("oil_spill_simulation.npz", oil_map=oil_map)

# 可视化最后一帧
plt.imshow(oil_map[-1], cmap='hot')
plt.title("Oil Distribution at Final Time Step")
plt.colorbar(label='Oil Thickness')
plt.show()
