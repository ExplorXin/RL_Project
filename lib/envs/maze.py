import numpy as np
import time
import sys
import tkinter as tk

unit = 40 #每个格子的大小
rows = 5  # 行数
cols = 5  # 列数

class MazeEnv(tk.Tk, object):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.action_space = [0, 1, 2, 3] # 上-下-左-右
        self.n_actions = len(self.action_space)
        self.title('寻宝')
        self.geometry('{0}x{1}'.format(cols*unit, rows*unit))
        self._build_maze()

    def _build_maze(self):
        # 创建一个画布
        self.canvas = tk.Canvas(self, bg='white',
                                height=rows * unit,
                                width=cols * unit)

        # 在画布上画出列
        for c in range(0, cols * unit, unit):
            # 每一列起点与终点
            x0, y0, x1, y1 = c, 0, c, rows * unit
            self.canvas.create_line(x0, y0, x1, y1)

        # 在画布上画出行
        for r in range(0, rows * unit, unit):
            x0, y0, x1, y1 = 0, r, cols * unit, r
            self.canvas.create_line(x0, y0, x1, y1)

        # 创建探险者起始位置（默认为左上角）
        origin = np.array([20, 20])  # 单位：像素；注意：20 = unit // 2

        # 陷阱 1
        hell1_center = origin + np.array([unit, unit])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')

        # 陷阱 2
        hell2_center = origin + np.array([unit * 2, unit])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')
        
        # 陷阱 3
        hell3_center = origin + np.array([unit * 3, unit])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - 15, hell3_center[1] - 15,
            hell3_center[0] + 15, hell3_center[1] + 15,
            fill='black')

        # 陷阱 4
        hell4_center = origin + np.array([unit, unit * 3])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - 15, hell4_center[1] - 15,
            hell4_center[0] + 15, hell4_center[1] + 15,
            fill='black')

        # 陷阱 5
        hell5_center = origin + np.array([unit * 3, unit * 3])
        self.hell5 = self.canvas.create_rectangle(
            hell5_center[0] - 15, hell5_center[1] - 15,
            hell5_center[0] + 15, hell5_center[1] + 15,
            fill='black')

        # 陷阱 6
        hell6_center = origin + np.array([0, unit * 4])
        self.hell6 = self.canvas.create_rectangle(
            hell6_center[0] - 15, hell6_center[1] - 15,
            hell6_center[0] + 15, hell6_center[1] + 15,
            fill='black')

        # 陷阱 7
        hell7_center = origin + np.array([unit * 4, unit * 4])
        self.hell7 = self.canvas.create_rectangle(
            hell7_center[0] - 15, hell7_center[1] - 15,
            hell7_center[0] + 15, hell7_center[1] + 15,
            fill='black')

        # 宝藏位置（终点）
        oval_center = origin + np.array([unit * 2, unit * 4])  # 第 3 列，第 5 行（0-indexed）
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # 将探险者用矩形表示（初始位置在起点 origin）
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # 画布展示
        self.canvas.pack()


    # 根据当前的状态重置画布（为展示动态效果）
    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        return self.canvas.coords(self.rect)

    # 根据当前行为，确定下一步的位置
    def step(self, action):
        # 智能体当前坐标(x_0,y_0,x_1,y_1)，矩形 [左-上-右-下] 坐标
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        if action == 0:                  # 上
            if s[1] > unit:              # 在不离开网格条件下向上移动
                base_action[1] -= unit
        elif action == 1:                # 下
            if s[1] < (rows - 1) * unit:
                base_action[1] += unit
        elif action == 2:                # 左
            if s[0] > 0:
                base_action[0] -= unit
        elif action == 3:                # 右
            if s[0] < (cols - 1) * unit:
                base_action[0] += unit

        # 在画布上将探险者移动至下一位置
        self.canvas.move(self.rect, base_action[0], base_action[1])
        # 重新渲染整个界面
        s_ = self.canvas.coords(self.rect)
        oval_flag = False

        # 根据当前位置来获得回报值及是否终止
        if s_ == self.canvas.coords(self.oval):
            reward = 5
            done = True
            s_ = 'terminal'
            oval_flag = True
        elif s_ in [self.canvas.coords(self.hell1), 
                    self.canvas.coords(self.hell2), 
                    self.canvas.coords(self.hell3), 
                    self.canvas.coords(self.hell4), 
                    self.canvas.coords(self.hell5), 
                    self.canvas.coords(self.hell6), self.canvas.coords(self.hell7)]:
            reward = -10
            done = True
            s_ = 'terminal'
        else:
            reward = -1
            done = False
        return s_, reward, done, oval_flag

    def render(self):
        time.sleep(0.1)
        self.update()

    def render_by_policy(self, policy_grid):
        """在画布上绘制策略箭头 (0:上 1:下 2:左 3:右)"""
        # 清除之前绘制的策略箭头
        self.canvas.delete("policy_arrow")

        pixels = 40   # 每格像素
        origin = 20   # 格子中心偏移
        arrow_len = 12

        for i in range(policy_grid.shape[0]):
            for j in range(policy_grid.shape[1]):
                action = policy_grid[i, j]
                cx = j * pixels + origin
                cy = i * pixels + origin

                if action == -1:
                    continue  # 陷阱/宝藏, 不画箭头

                for a in action:
                    if a == 0:    # 上
                        dx, dy = 0, -arrow_len
                    elif a == 1:  # 下
                        dx, dy = 0, arrow_len
                    elif a == 2:  # 左
                        dx, dy = -arrow_len, 0
                    elif a == 3:  # 右
                        dx, dy = arrow_len, 0
                    else:
                        continue

                    self.canvas.create_line(
                        cx, cy, cx + dx, cy + dy,
                        arrow='last', arrowshape=(8, 10, 4),
                        fill='blue', tags='policy_arrow', width=2)

        self.render()