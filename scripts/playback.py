import numpy as np

from humanoid_swe_challenge.config import NP_RANDOM_SEED
from humanoid_swe_challenge.sims.pusher_manip.env import PusherManipEnv

steps=np.load("log/04-06-2026/22-11-01.npy")
env = PusherManipEnv(render_mode="human", 
                     render_realtime=True)
env.reset(seed=NP_RANDOM_SEED)

for s in steps:
    env.step(action=s)

env.close()