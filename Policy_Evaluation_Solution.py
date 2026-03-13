import numpy as np
import sys

from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()


## 策略评估方法
def policy_eval(policy, env, discount_factor =1, threshold=0.00001):
	# 初始化各状态的状态值函数
	V = np.zeros(env.nS)
	i = 0

	while True:
		value_delta = 0
		# 遍历各状态
		for s in range(env.nS):
			v = 0

			for a, action_prob in enumerate(policy[s]):
				for prob, next_state, reward, done in env.P[s][a]:
					v += action_prob * prob * (reward + discount_factor * V[next_state])
			value_delta = max(value_delta, np,abs(v - V[s]))
			V[s] =v


		i += 1
		if value_delta < threshold:
			break
	return np.array(V)