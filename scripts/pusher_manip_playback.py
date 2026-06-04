import numpy as np

from humanoid_swe_challenge.config import NP_RANDOM_SEED
from humanoid_swe_challenge.sims.pusher_manip.env import PusherManipEnv
from humanoid_swe_challenge.sims.config.video_cfg import VideoCfg

steps=np.load("log/04-06-2026/22-21-47.npy")
env = PusherManipEnv(render_mode="human", 
                     render_realtime=True,
                     video_cfg=VideoCfg())
env.reset(seed=NP_RANDOM_SEED)

for s in steps:
    env.step(action=s)

env.close()