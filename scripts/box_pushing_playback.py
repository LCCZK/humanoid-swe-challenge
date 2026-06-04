import numpy as np

from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.sims.config.video_cfg import VideoCfg

steps=np.load("log/04-06-2026/22-58-02.npy")
env = BoxPusingEnv(render_mode="human", 
                     render_realtime=True,
                     video_cfg=VideoCfg())
env.reset()

for s in steps:
    env.step(action=s)

env.close()