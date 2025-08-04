import heapq
import numpy as np
import matplotlib.pyplot as plt
import random
import time  # 导入time模块用于计时

# 定义机器人的大小（以grid为单位）
ROBOT_SIZE = 1

# 定义离散的运动控制
# 修改后的运动控制，加入更多旋转角度和前进距离
MOTIONS = [
    (1, 0, 0),  # 向前移动1个grid，不旋转
    (-1, 0, 0),  # 向后移动1个grid，不旋转
    # (0, 0, 10),  # 向左旋转10度
    # (0, 0, 20),  # 向左旋转20度
    # (0, 0, 30),  # 向左旋转30度
    (0, 0, 45),  # 向左旋转45度
    #(0, 0, 90),  # 向左旋转90度
    # (0, 0, -10),  # 向右旋转10度
    # (0, 0, -20),  # 向右旋转20度
    # (0, 0, -30),  # 向右旋转30度
    (0, 0, -45),  # 向右旋转45度
    #(0, 0, -90),  # 向右旋转90度
]

# 人工势场参数
ATTRACTION_GAIN = 3.0  # 引力增益
REPULSION_GAIN = 100.0  # 斥力增益
REPULSION_DISTANCE = 3  # 斥力作用的最大距离

# 新增开关，控制是否开启人工势场，默认开启
use_artificial_potential_field = True

def is_collision(grid, x, y, theta):
    """
    检查机器人在给定位置和角度是否与障碍物碰撞
    """
    robot_center_x = x
    robot_center_y = y
    half_size = ROBOT_SIZE // 2
    for dx in range(-half_size, half_size + 1):
        for dy in range(-half_size, half_size + 1):
            # 计算旋转后的坐标
            rotated_x = int(robot_center_x + dx * np.cos(np.radians(theta)) - dy * np.sin(np.radians(theta)))
            rotated_y = int(robot_center_y + dx * np.sin(np.radians(theta)) + dy * np.cos(np.radians(theta)))
            # 检查坐标是否在grid范围内且是否为障碍物
            if 0 <= rotated_x < grid.shape[0] and 0 <= rotated_y < grid.shape[1]:
                if grid[rotated_x, rotated_y] == 1:
                    return True
    return False

def attraction_force(current, goal):
    """
    计算引力
    """
    dx = goal[0] - current[0]
    dy = goal[1] - current[1]
    return ATTRACTION_GAIN * np.sqrt(dx ** 2 + dy ** 2)

def generate_grid_map(size, obstacle_ratio):
    """
    生成随机的grid map，保证起点和终点之间至少有一条最窄处大于ROBOT_SIZE的路径，
    并在地图边缘多加一圈障碍物
    :param size: grid map的大小，如 (100, 100)
    :param obstacle_ratio: 障碍物的比例，范围是 [0, 1]
    :return: 生成的grid map
    """
    height, width = size
    # 由于要添加边缘障碍物，实际填充障碍物的区域大小要相应调整
    inner_height = height - 2
    inner_width = width - 2
    grid = np.zeros((height, width), dtype=int)

    # 确保起点和终点之间有一条路径
    start_x, start_y = 1, 1  # 起点位置调整，避开边缘障碍物
    end_x, end_y = inner_height, inner_width

    # 生成路径
    path_width = ROBOT_SIZE + 1
    if random.random() < 0.5:  # 随机选择路径是水平优先还是垂直优先
        # 水平优先
        for y in range(start_y, end_y + 1):
            for x in range(max(start_x, 0), min(end_x + 1, inner_width)):
                grid[x, y] = 0
        for x in range(start_x, end_x + 1):
            for y in range(max(start_y, 0), min(end_y + 1, inner_height)):
                grid[x, y] = 0
    else:
        # 垂直优先
        for x in range(start_x, end_x + 1):
            for y in range(max(start_y, 0), min(end_y + 1, inner_height)):
                grid[x, y] = 0
        for y in range(start_y, end_y + 1):
            for x in range(max(start_x, 0), min(end_x + 1, inner_width)):
                grid[x, y] = 0

    # 计算需要添加的障碍物数量
    total_cells = inner_height * inner_width
    num_obstacles = int(total_cells * obstacle_ratio)

    # 随机添加障碍物
    obstacle_count = 0
    while obstacle_count < num_obstacles:
        x = random.randint(1, inner_height)
        y = random.randint(1, inner_width)
        if grid[x, y] == 0 and (x, y) not in [(start_x, start_y), (end_x, end_y)]:
            grid[x, y] = 1
            obstacle_count += 1

    # 在地图边缘添加一圈障碍物
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    print("生成grid map成功")
    return grid

def repulsion_force(grid, current, goal):
    """
    计算斥力，加入调节因子
    """
    repulsion = 0
    dist_to_goal = np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                dist = np.sqrt((current[0] - i) ** 2 + (current[1] - j) ** 2)
                if dist <= REPULSION_DISTANCE:
                    # 加入调节因子
                    factor = dist_to_goal / (dist_to_goal + dist)
                    repulsion += REPULSION_GAIN * factor * ((1 / dist) - (1 / REPULSION_DISTANCE)) * (1 / (dist ** 2))
    return repulsion

def heuristic(a, b):
    """
    计算两点之间的曼哈顿距离作为启发式函数
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    """
    单向A*算法进行路径规划，结合人工势场
    """
    # 搜索的open list和closed set
    open_list = []
    heapq.heappush(open_list, (0, start, [], []))
    closed_set = set()

    # 记录已经访问过的状态
    visited = {}

    while open_list:
        _, current, path, action_list = heapq.heappop(open_list)
        x, y, theta = current

        if current in closed_set:
            continue
        closed_set.add(current)

        # 检查是否到达终点
        dx = goal[0] - x
        dy = goal[1] - y
        dist_to_goal = np.sqrt(dx ** 2 + dy ** 2)
        angle_to_goal = np.degrees(np.arctan2(dy, dx))
        angle_diff = abs(angle_to_goal - theta)
        if dist_to_goal <= 1 and angle_diff <= 45:
            return path + [current], action_list

        visited[current] = path

        for i, (dx, dy, dtheta) in enumerate(MOTIONS):
            new_theta = (theta + dtheta) % 360
            new_x = x + dx * np.cos(np.radians(new_theta))
            new_y = y + dx * np.sin(np.radians(new_theta))

            new_state = (new_x, new_y, new_theta)

            if not is_collision(grid, new_x, new_y, new_theta):
                new_path = path + [current]
                new_action_list = action_list + [i]
                # 根据开关决定是否计算人工势场力
                if use_artificial_potential_field:
                    attr_force = attraction_force((new_x, new_y), goal)
                    rep_force = repulsion_force(grid, (new_x, new_y), goal)
                else:
                    attr_force = 0
                    rep_force = 0
                cost = len(new_path) + heuristic((new_x, new_y), goal) + attr_force + rep_force
                heapq.heappush(open_list, (cost, new_state, new_path, new_action_list))

    return None, None

def plot_path(grid, path):
    """
    绘制地图和路径
    """
    plt.imshow(grid, cmap='gray_r')
    if path:
        x_coords = [state[0] for state in path]
        y_coords = [state[1] for state in path]
        plt.plot(y_coords, x_coords, 'r-')
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 生成grid map
    grid_size = (30, 30)
    obstacle_ratio = 0.05

    # 记录地图生成开始时间
    start_time_map = time.time()
    grid = generate_grid_map(grid_size, obstacle_ratio)
    # 记录地图生成结束时间
    end_time_map = time.time()

    start = (1, 1, 0)  # 起点 (x, y, theta)
    goal = (grid_size[0] - 2, grid_size[1] - 2, 0)  # 终点 (x, y, theta)，添加初始角度值

    # 记录轨迹规划开始时间
    start_time_path = time.time()
    path, action_list = a_star(grid, start, goal)
    # 记录轨迹规划结束时间
    end_time_path = time.time()

    if path:
        print("找到路径:", path)
        print("action list:", action_list)
        plot_path(grid, path)
    else:
        print("未找到路径")

    # 输出地图生成和轨迹规划所耗费的时间
    print(f"地图生成所耗费的时间: {end_time_map - start_time_map} 秒")
    print(f"轨迹规划所耗费的时间: {end_time_path - start_time_path} 秒")