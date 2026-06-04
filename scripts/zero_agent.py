import numpy as np

from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.sims.pusher_manip.env import PusherManipEnv

seed = 42

env = PusherManipEnv(render_mode="human", render_realtime=True)
observation, info = env.reset(seed=seed)

for _ in range(1000):
    action = np.array([0.0, 0.1, 0.0])
    observation, _, _, _, info = env.step(action)

env.close()