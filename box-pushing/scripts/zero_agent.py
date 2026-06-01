import gymnasium as gym
import numpy as np

from box_pushing.gym_env.base_env import PusingEnv

env = PusingEnv(render_mode="rgb_array")
observation, info = env.reset()

for _ in range(1000000):
    action = np.array([0.0, -0.1, 0.0])
    observation, _, _, _, info = env.step(action)
    print()
    print(observation)
    print()

env.close()