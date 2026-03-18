import numpy as np
import gymnasium as gym
from gymnasium import spaces

# 定义动作空间
up = 0
right = 1
down = 2
left = 3

# 定义宝藏所在状态
done_location = 8



class GridworldEnv(gym.Env):

    def __init__(self, shape=(5, 5)):
        super(GridworldEnv, self).__init__()

        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("shape must be length 2")

        self.shape = shape

        nS = np.prod(shape)   # 状态数 25
        nA = 4                # 动作数

        self.nS = nS
        self.nA = nA

        # Gymnasium必须定义
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        max_Y, max_X = shape

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:

            s = it.iterindex
            y, x = it.multi_index

            P[s] = {a: [] for a in range(nA)}

            is_done = lambda s: s == done_location
            reward = 0.0 if is_done(s) else -1.0

            if is_done(s):

                for a in range(nA):
                    P[s][a] = [(1.0, s, reward, True)]

            else:

                ns_up = s if y == 0 else s - max_X
                ns_right = s if x == max_X - 1 else s + 1
                ns_down = s if y == max_Y - 1 else s + max_X
                ns_left = s if x == 0 else s - 1

                P[s][up] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][right] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][down] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][left] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        self.P = P
        self.state = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(self.nS)
        return self.state, {}

    def step(self, action):

        prob, next_state, reward, done = self.P[self.state][action][0]

        self.state = next_state

        terminated = done
        truncated = False

        return next_state, reward, terminated, truncated, {}