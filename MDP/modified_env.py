import numpy as np

# 动作空间
up = 0
right = 1
down = 2
left = 3

# 特殊状态
TREASURE_LOCATION = 8     # 宝藏区域
FORBIDDEN_LOCATION = 12   # 禁止区域
SPECIAL_LOCATION = 7      # 进入奖励-2


class ModifiedGridworldEnv:
    """修改后的5x5网格世界
    [[ 0   1   2   3   4]
     [ 5   6   7   8   9]    S8=宝藏(reward=0), S7=特殊(reward=-2)
     [10  11  12  13  14]    S12=禁止区域(reward=-5)
     [15  16  17  18  19]
     [20  21  22  23  24]]
    """

    def __init__(self, shape=[5, 5]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')
        self.shape = shape

        nS = np.prod(shape)
        nA = 4
        max_Y = shape[0]
        max_X = shape[1]

        def reward(next_state):
            if next_state == TREASURE_LOCATION:
                return 0.0
            elif next_state == FORBIDDEN_LOCATION:
                return -5.0
            elif next_state == SPECIAL_LOCATION:
                return -2.0
            else:
                return -1.0

        def is_done(s):
            return s == TREASURE_LOCATION

        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            P[s] = {a: [] for a in range(nA)}

            if is_done(s):
                P[s][up] = [(1.0, s, reward(s), True)]
                P[s][right] = [(1.0, s, reward(s), True)]
                P[s][down] = [(1.0, s, reward(s), True)]
                P[s][left] = [(1.0, s, reward(s), True)]
            else:
                ns_up = s if y == 0 else s - max_X
                ns_right = s if x == (max_X - 1) else s + 1
                ns_down = s if y == (max_Y - 1) else s + max_X
                ns_left = s if x == 0 else s - 1
                P[s][up] = [(1.0, ns_up, reward(ns_up), is_done(ns_up))]
                P[s][right] = [(1.0, ns_right, reward(ns_right), is_done(ns_right))]
                P[s][down] = [(1.0, ns_down, reward(ns_down), is_done(ns_down))]
                P[s][left] = [(1.0, ns_left, reward(ns_left), is_done(ns_left))]
            it.iternext()

        isd = np.ones(nS) / nS
        self.P = P
        self.nS = nS
        self.nA = nA
        self.isd = isd
