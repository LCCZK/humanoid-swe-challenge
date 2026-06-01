import numpy as np
from humanoid_swe_challenge.sims.box_pushing.base_env import PusingEnv

env = PusingEnv(render_mode="human")
observation, info = env.reset()

for _ in range(1000000):
    action = np.array([0.0, 0.1, 0.0])
    observation, _, _, _, info = env.step(action)
    print()
    print(observation)
    print()

env.close()