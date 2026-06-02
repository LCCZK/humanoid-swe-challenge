import numpy as np

from humanoid_swe_challenge.sims.box_pushing.base_env import BoxPusingEnv
from humanoid_swe_challenge.sims.pusher_manip.base_env import PusherManipEnv

seed = 42

env = PusherManipEnv(render_mode="human")
observation, info = env.reset(seed=seed)

for _ in range(1000000):
    action = np.array([0.0, 0.1, 0.0])
    observation, _, _, _, info = env.step(action)
    print()
    print(observation)
    print()

env.close()