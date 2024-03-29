import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
from scipy.spatial import ConvexHull
from datetime import datetime


class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()

        # 定义环境的动作空间和观测空间
        self.action_space = spaces.Discrete(256)  # 16 * 16 = 256
        self.observation_space = spaces.Box(low=0, high=255, shape=(16, 16), dtype=np.int64)

        # 初始化环境状态
        self.state = np.zeros((16, 16), dtype=np.int64)

        # 初始化奖励权重
        self.weight_num_points = 0.0001  # 点的数量的权重
        self.weight_area = 0.01   # 最大化围成的面积的权重

        self.reset_threshold = 10
    def step(self, action):
        # 执行动作
        x, y = divmod(action, 16)
        self.state[x, y] = 1

        # 计算奖励
        num_selected_points = np.sum(self.state, dtype=np.int64)
        area = self.calculate_selected_area()
        # 使用最小化点的数量和最大化围成的面积作为奖励
        reward_num_points = -num_selected_points * self.weight_num_points
        reward_area = area * self.weight_area

        # 定义完成任务的条件
        done = num_selected_points > self.reset_threshold

        return self.state.copy(), reward_num_points + reward_area, done, {}

    # def calculate_selected_area(self):
    #     selected_indices = np.where(self.state == 1)
    #     if len(selected_indices) < 3:
    #         # 如果点的数量小于3，无法形成凸包，面积为0
    #         return 0
    #
    #     # 计算凸包
    #     hull = ConvexHull(selected_indices)
    #
    #     # 获取凸包的顶点坐标
    #     convex_hull_points = selected_indices[hull.vertices]
    #
    #     # 使用shoelace formula计算凸包的面积
    #     area = 0.5 * np.abs(np.dot(convex_hull_points[:, 0], np.roll(convex_hull_points[:, 1], 1)) -
    #                         np.dot(np.roll(convex_hull_points[:, 0], 1), convex_hull_points[:, 1]))
    #
    #     return area

    def calculate_selected_area(self):
        #将点的坐标按逆时针排序
        point_x, point_y = np.where(self.state == 1)

        center_x, center_y = np.mean(point_x), np.mean(point_y)
        angles = np.arctan2(point_y - center_y, point_x - center_x)
        sorted_indices = np.argsort(angles)
        point_x, point_y = np.take(point_x, sorted_indices), np.take(point_y, sorted_indices)

        n = len(point_x)
        if n < 3:
            #少于三个点时无法构成封闭图形
            return 0.0

        # 计算鞋带公式中的部分面积
        area = 0.0
        for i in range(n - 1):
            area += point_x[i] * point_y[i + 1] - point_x[i + 1] * point_y[i]
        area += point_x[n - 1] * point_y[0] - point_x[0] * point_y[n - 1]

        # 计算多边形的有向面积
        area = 0.5 * abs(area)

        return area

    def reset(self):
        # 重置环境状态
        if(np.sum(self.state) > self.reset_threshold):
            self.state = np.zeros((16, 16), dtype=np.int64)
        return self.state.copy()

    def render(self, mode='human'):
        # 输出整个图
        print(self.state)

# 创建环境
env = Env()

# 创建日志
log_dir = "./ppo_custom_env_tensorboard/PPO_" + datetime.now().strftime("%Y%m%d-%H%M%S")


# 用PPO算法进行训练
model = PPO("MlpPolicy", env,
            learning_rate=0.00013,
            clip_range=0.1,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            tensorboard_log=log_dir)

model.learn(total_timesteps=450000, log_interval=1)

# 保存训练好的模型
model.save("ppo_custom_env")

# 加载已经训练好的模型
# model = PPO.load("ppo_custom_env")
selected_points_count = 0

# 环境重置
obs = env.reset()


# 在环境中执行一个随机动作
for _ in range(16 * 16):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)



selected_points_count += np.sum(obs)
print(f"被选中的点的数量：{selected_points_count}")
print(f"reward={reward}")

env.render()

# 关闭环境
env.close()
