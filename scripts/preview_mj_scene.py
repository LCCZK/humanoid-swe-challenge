import mujoco
import mujoco.viewer

from humanoid_swe_challenge.sims.box_pushing.base_env import PusingEnv

env = PusingEnv()
mujoco.viewer.launch(env.model, env.data)