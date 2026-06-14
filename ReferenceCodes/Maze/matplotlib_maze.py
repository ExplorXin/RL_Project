"""
基于 matplotlib 的网格世界迷宫环境
替代原 tkinter 版 Maze.py，使用现代可视化方案
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython import display
import time

UNIT = 40
ROWS = 5
COLS = 5


class MazeEnv:
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.ax.set_xlim(0, COLS * UNIT)
        self.ax.set_ylim(ROWS * UNIT, 0)  # y 向下增大，与原版一致
        self.ax.set_xticks(np.arange(0, COLS * UNIT + 1, UNIT))
        self.ax.set_yticks(np.arange(0, ROWS * UNIT + 1, UNIT))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.tick_params(labelleft=False, labelbottom=False)

        origin = np.array([20, 20])
        self.agent_center = origin.copy()
        self.init_center = origin.copy()

        # 陷阱中心坐标
        self.trap_centers = [
            origin + np.array([UNIT, UNIT]),          # (1,1)
            origin + np.array([UNIT * 2, UNIT]),      # (2,1)
            origin + np.array([UNIT * 3, UNIT]),      # (3,1)
            origin + np.array([UNIT, UNIT * 3]),      # (1,3)
            origin + np.array([UNIT * 3, UNIT * 3]),  # (3,3)
            origin + np.array([0, UNIT * 4]),         # (0,4)
            origin + np.array([UNIT * 4, UNIT * 4]),  # (4,4)
        ]
        # 宝藏中心坐标
        self.treasure_center = origin + np.array([UNIT * 2, UNIT * 4])  # (2,4)

        self._draw_static()
        self.agent_patch = self._create_agent_patch()
        self.ax.add_patch(self.agent_patch)
        self.policy_arrows = []

    # ── 内部绘制 ────────────────────────────────────────────

    def _draw_static(self):
        """绘制陷阱（黑色方块）和宝藏（黄色椭圆）"""
        for tc in self.trap_centers:
            self.ax.add_patch(patches.Rectangle(
                (tc[0] - 15, tc[1] - 15), 30, 30,
                facecolor='black', edgecolor='#333'))

        self.ax.add_patch(patches.Ellipse(
            (self.treasure_center[0], self.treasure_center[1]),
            30, 30, facecolor='yellow', edgecolor='gold'))

    def _create_agent_patch(self):
        """创建智能体红色方块"""
        return patches.Rectangle(
            (self.agent_center[0] - 15, self.agent_center[1] - 15),
            30, 30, facecolor='red', edgecolor='darkred')

    def _get_state(self):
        """返回当前状态的边界框 (与原版一致)"""
        c = self.agent_center
        return [c[0] - 15, c[1] - 15, c[0] + 15, c[1] + 15]

    # ── 环境接口 ────────────────────────────────────────────

    def reset(self):
        time.sleep(0.3)
        self.agent_center = self.init_center.copy()
        self.agent_patch.set_xy(
            (self.init_center[0] - 15, self.init_center[1] - 15))
        return self._get_state()

    def step(self, action):
        c = self.agent_center.copy()

        if action == 0 and c[1] > UNIT:                 # 上
            c[1] -= UNIT
        elif action == 1 and c[1] < (ROWS - 1) * UNIT:  # 下
            c[1] += UNIT
        elif action == 2 and c[0] > 0:                   # 左
            c[0] -= UNIT
        elif action == 3 and c[0] < (COLS - 1) * UNIT:   # 右
            c[0] += UNIT

        self.agent_center = c
        self.agent_patch.set_xy((c[0] - 15, c[1] - 15))

        # 判断终止
        if np.array_equal(c, self.treasure_center):
            return 'terminal', 1, True, True
        if any(np.array_equal(c, tc) for tc in self.trap_centers):
            return 'terminal', -1, True, False
        return self._get_state(), 0, False, False

    def render(self):
        display.clear_output(wait=True)
        display.display(self.fig)
        time.sleep(0.1)

    # ── 策略可视化 ──────────────────────────────────────────

    def render_by_policy(self, policy_grid):
        """在网格上绘制策略方向箭头 (0:上 1:下 2:左 3:右)"""
        # 清除旧箭头
        for arrow in self.policy_arrows:
            arrow.remove()
        self.policy_arrows.clear()

        arrow_len = 10
        for i in range(policy_grid.shape[0]):
            for j in range(policy_grid.shape[1]):
                action = policy_grid[i, j]
                cx = j * UNIT + 20
                cy = i * UNIT + 20

                if action == -1:
                    continue  # 陷阱或宝藏

                for a in action:
                    if a == 0:
                        dx, dy = 0, -arrow_len       # 上
                    elif a == 1:
                        dx, dy = 0, arrow_len         # 下
                    elif a == 2:
                        dx, dy = -arrow_len, 0        # 左
                    elif a == 3:
                        dx, dy = arrow_len, 0         # 右
                    else:
                        continue

                    arrow = self.ax.arrow(
                        cx, cy, dx, dy,
                        head_width=6, head_length=6,
                        fc='blue', ec='blue',
                        length_includes_head=True)
                    self.policy_arrows.append(arrow)

        display.display(self.fig)

    def close(self):
        plt.close(self.fig)
