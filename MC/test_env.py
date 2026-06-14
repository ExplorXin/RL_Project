from lib.envs.halften import HalftenEnv
import gymnasium as gym

env = HalftenEnv()

obs, info = env.reset()

print("初始状态:", obs)

for _ in range(10):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print("obs:", obs, "reward:", reward)

    if terminated or truncated:
        print("游戏结束")
        break