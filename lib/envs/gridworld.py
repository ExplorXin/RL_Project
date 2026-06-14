import numpy as np
import sys

import gymnasium as gym
from gymnasium import spaces


	# 定义动作空间
up = 0
right = 1
down = 2
left = 3
	# 定义宝藏所在状态
done_location = 8
	# 定义网格世界环境模型
class GridworldEnv(gym.Env):
	'''定义5x5网格
		[[o,o,o,o,o],
		[o,o,o,T,o],
		[o,o,o,o,o],
		[o,x,o,o,o],
		o,o,o,o,o]]
		x为智能体位置，T宝藏位置'''
	def __init__(self, shape = [5,5]):
		if not isinstance(shape, (list, tuple)) or not len(shape) ==2:
			raise ValueError('shape argument must be a list/tuple of length 2')
		self.shape = shape
		
		nS = np.prod(shape) # 状态个数，5x5=25
		nA = 4 # 动作个数
		max_Y = shape[0] # 5
		max_X = shape[1] # 5
		
		# grid 创建5x5表格，np.nditer为该表格索引排序
		# flags 表示对grid进行多重索引，如(1,3)
		P={}
		grid = np.arange(nS).reshape(shape)
		it = np.nditer(grid, flags = ['multi_index'])


		# 定义即时奖励R(s,a)
		def reward(next_state):
			if next_state == done_location:
				reward = 0
			else:
				reward = -1.0
			return reward
		
		while not it.finished:
			s = it.iterindex
			y, x= it.multi_index
			P[s] = {a:[] for a in range(nA)}
			is_done = lambda s: s == done_location
			# reward = 0.0 if is_done(s) else -1.0
			
			# P[s][a]第一参数为状态转移概率，为1，第二参数代表到达下一个的位置
			# 第三个参数reward是回报，最后一参数True表示到达宝藏区

			# 位于宝藏区，则停留原地
			if is_done(s):
				P[s][up] = [(1,s,reward(s),True)]
				P[s][right] = [(1,s,reward(s),True)]
				P[s][down] = [(1,s,reward(s),True)]
				P[s][left] = [(1,s,reward(s),True)]

			# 位于宝藏区外，进行状态转换
			else:
				ns_up = s if y == 0 else s- max_X
				ns_right = s if x == (max_X-1) else s+1
				ns_down = s if y == (max_Y-1) else s+ max_X
				ns_left = s if x == 0 else s-1
				P[s][up] = [(1,ns_up, reward(ns_up), is_done(ns_up))]
				P[s][right] = [(1,ns_right,reward(ns_right), is_done(ns_right))]
				P[s][down] = [(1,ns_down,reward(ns_down), is_done(ns_down))]
				P[s][left] = [(1,ns_left,reward(ns_left), is_done(ns_left))]
			it.iternext()
		isd = np.ones(nS)/nS
		self.P =P

		self.nS = nS
		self.nA = nA
		self.isd = isd



