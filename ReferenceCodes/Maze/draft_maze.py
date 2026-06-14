"""
迷宫网格世界环境 —— 基于 Sarsa 算法的马尔可夫决策过程求解。

环境布局 (5×5 网格):
    ┌───┬───┬───┬───┬───┐
    │ S │   │   │   │   │   S: 起点 (0,0)
    ├───┼───┼───┼───┼───┤
    │   │ H │ H │ H │   │   H: 陷阱 (reward = -1)
    ├───┼───┼───┼───┼───┤
    │   │   │   │   │   │
    ├───┼───┼───┼───┼───┤
    │   │ H │   │ H │   │   G: 宝藏 (reward = +1)
    ├───┼───┼───┼───┼───┤
    │ H │   │ G │   │ H │
    └───┴───┴───┴───┴───┘
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


class MazeEnv:
    """网格世界迷宫环境。

    Attributes:
        rows, cols: 网格行数和列数
        start_state: 起点坐标 (row, col)
        trap_states: 陷阱位置列表
        goal_state: 宝藏（终点）位置
        current_state: 当前智能体所在状态
        n_actions: 动作空间大小 (4)
        action_space: 动作名称列表
    """

    def __init__(self, rows=5, cols=5):
        self.rows = rows
        self.cols = cols

        # 动作空间: 0=上, 1=下, 2=左, 3=右
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)

        # 状态定义
        self.start_state = (0, 0)
        self.goal_state = (4, 2)

        # 陷阱位置 (与原 Maze.py 布局一致)
        self.trap_states = [
            (1, 1), (1, 2), (1, 3),
            (3, 1), (3, 3),
            (4, 0), (4, 4),
        ]

        # 所有终止状态
        self.terminal_states = {self.goal_state} | set(self.trap_states)

        self.current_state = self.start_state

    # ---------- 环境核心方法 ----------

    def reset(self):
        """重置环境到起点，返回初始状态。"""
        self.current_state = self.start_state
        return self.current_state

    def step(self, action):
        """执行动作，返回 (next_state, reward, done)。

        动作编号: 0-上, 1-下, 2-左, 3-右
        碰到边界则停留在原地。
        进入陷阱 reward=-1，到达宝藏 reward=+1，其他 reward=0。
        """
        r, c = self.current_state

        if action == 0:    # 上
            r = max(r - 1, 0)
        elif action == 1:  # 下
            r = min(r + 1, self.rows - 1)
        elif action == 2:  # 左
            c = max(c - 1, 0)
        elif action == 3:  # 右
            c = min(c + 1, self.cols - 1)

        self.current_state = (r, c)

        if self.current_state == self.goal_state:
            return self.current_state, 1, True
        elif self.current_state in self.trap_states:
            return self.current_state, -1, True
        else:
            return self.current_state, 0, False

    # ---------- 状态编码 ----------

    def state_to_index(self, state):
        """将 (row, col) 状态映射为 0..24 的线性索引。"""
        return state[0] * self.cols + state[1]

    def state_to_str(self, state):
        """将状态转为字符串键（兼容原 Q 表风格）。"""
        return f"({state[0]},{state[1]})"

    def get_all_states(self):
        """返回所有非终止状态的列表。"""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if (r, c) not in self.terminal_states]

    # ---------- 可视化 ----------

    def render(self, policy=None, q_values=None, visit_count=None,
               title="Grid World Maze", ax=None):
        """绘制网格世界。

        Args:
            policy: dict, state -> best_action (0-3) 或 action tuple
            q_values: dict, state -> [q0, q1, q2, q3]  用于绘制 Q 值热力图
            visit_count: dict, state -> count  用于绘制访问频次
            title: 图表标题
            ax: 可选 matplotlib axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        # 背景色矩阵: 0=普通, 1=陷阱, 2=宝藏, 3=起点
        grid = np.zeros((self.rows, self.cols))
        for tr, tc in self.trap_states:
            grid[tr, tc] = 1
        g_r, g_c = self.goal_state
        grid[g_r, g_c] = 2
        grid[self.start_state] = 3

        cmap = ListedColormap(['#ecf0f1', '#2c3e50', '#f1c40f', '#3498db'])
        ax.imshow(grid, cmap=cmap, aspect='equal', origin='upper')

        # 网格线
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)

        # 在每个格子标注信息
        for r in range(self.rows):
            for c in range(self.cols):
                state = (r, c)
                if state in self.terminal_states:
                    continue

                text_parts = []

                # Q 值显示
                if q_values is not None and state in q_values:
                    qv = q_values[state]
                    directions = ['↑', '↓', '←', '→']
                    for i, (d, v) in enumerate(zip(directions, qv)):
                        text_parts.append(f"{d}:{v:.2f}")

                # 策略箭头
                if policy is not None and state in policy:
                    act = policy[state]
                    if isinstance(act, (list, tuple)):
                        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
                        act_str = '/'.join(arrows[a] for a in act)
                    else:
                        arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
                        act_str = arrows.get(act, '?')
                    text_parts.insert(0, act_str)

                if text_parts:
                    ax.text(c, r, '\n'.join(text_parts[:2]),
                            ha='center', va='center', fontsize=7,
                            bbox=dict(boxstyle='round,pad=0.3',
                                      facecolor='white', alpha=0.8))

        # 图例
        legend_patches = [
            mpatches.Patch(color='#3498db', label='Start'),
            mpatches.Patch(color='#f1c40f', label='Goal'),
            mpatches.Patch(color='#2c3e50', label='Trap'),
        ]
        ax.legend(handles=legend_patches, loc='upper left',
                  bbox_to_anchor=(1.02, 1), borderaxespad=0)

        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xticklabels(range(self.cols))
        ax.set_yticklabels(range(self.rows))
        ax.set_title(title)
        ax.tick_params(length=0)

        return ax

    def render_policy(self, policy, title="Optimal Policy", ax=None):
        """只绘制最优策略箭头（简洁版）。"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        grid = np.zeros((self.rows, self.cols))
        for tr, tc in self.trap_states:
            grid[tr, tc] = 1
        g_r, g_c = self.goal_state
        grid[g_r, g_c] = 2
        grid[self.start_state] = 3

        cmap = ListedColormap(['#ecf0f1', '#2c3e50', '#f1c40f', '#3498db'])
        ax.imshow(grid, cmap=cmap, aspect='equal', origin='upper')

        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)

        arrows = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        for (r, c), act in policy.items():
            if (r, c) in self.terminal_states:
                continue
            acts = act if isinstance(act, (list, tuple)) else [act]
            for a in acts:
                dx, dy = arrows.get(a, (0, 0))
                ax.arrow(c, r, dx, dy, head_width=0.15, head_length=0.15,
                         fc='#e74c3c', ec='#e74c3c', alpha=0.8)

        legend_patches = [
            mpatches.Patch(color='#3498db', label='Start'),
            mpatches.Patch(color='#f1c40f', label='Goal'),
            mpatches.Patch(color='#2c3e50', label='Trap'),
        ]
        ax.legend(handles=legend_patches, loc='upper left',
                  bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_title(title)
        ax.tick_params(length=0)
        return ax

    def render_path(self, path, title="Agent Path", ax=None):
        """在网格上绘制智能体的行走路径。"""
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))

        grid = np.zeros((self.rows, self.cols))
        for tr, tc in self.trap_states:
            grid[tr, tc] = 1
        g_r, g_c = self.goal_state
        grid[g_r, g_c] = 2
        grid[self.start_state] = 3

        cmap = ListedColormap(['#ecf0f1', '#2c3e50', '#f1c40f', '#3498db'])
        ax.imshow(grid, cmap=cmap, aspect='equal', origin='upper')

        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)

        if len(path) > 1:
            rows_coords = [p[0] for p in path]
            cols_coords = [p[1] for p in path]
            ax.plot(cols_coords, rows_coords, 'o-', color='#e74c3c',
                    linewidth=2, markersize=8, markerfacecolor='white',
                    markeredgecolor='#e74c3c', markeredgewidth=2)
            # 标注步数
            for i, (r, c) in enumerate(path):
                ax.text(c, r, str(i), ha='center', va='center',
                        fontsize=8, color='#e74c3c', fontweight='bold')

        legend_patches = [
            mpatches.Patch(color='#3498db', label='Start'),
            mpatches.Patch(color='#f1c40f', label='Goal'),
            mpatches.Patch(color='#2c3e50', label='Trap'),
        ]
        ax.legend(handles=legend_patches, loc='upper left',
                  bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_title(title)
        ax.tick_params(length=0)
        return ax


# 便捷函数：绘制学习曲线
def plot_learning_curve(rewards, window=10, title="Learning Curve", ax=None):
    """绘制每个 episode 的奖励及其平滑曲线。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.3, color='#3498db', linewidth=0.8, label='Episode Reward')
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed,
                color='#e74c3c', linewidth=2, label=f'Moving Avg (n={window})')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_q_heatmap(q_table, env, title="Q-value Heatmap", ax=None):
    """绘制各状态各动作的 Q 值热力图 (5×5 格 × 4 动作 = 热力图)。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    q_grid = np.full((env.rows, env.cols), np.nan)
    for (r, c), qs in q_table.items():
        q_grid[r, c] = max(qs)
    im = ax.imshow(q_grid, cmap='RdYlGn', aspect='equal', origin='upper', vmin=-1, vmax=1)
    for r in range(env.rows):
        for c in range(env.cols):
            if not np.isnan(q_grid[r, c]):
                ax.text(c, r, f"{q_grid[r,c]:.2f}", ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_title(title)
    return ax
