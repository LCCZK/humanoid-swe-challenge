import numpy as np

from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.sims.config.video_cfg import VideoCfg

steps=np.load("log/demo/box_pushing.npy")
env = BoxPusingEnv(render_mode="human", 
                     render_realtime=True)
env.reset()

for s in steps:
    env.step(action=s)

env.close()