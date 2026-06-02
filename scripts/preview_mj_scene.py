import mujoco
import mujoco.viewer

from humanoid_swe_challenge.sims.box_pushing.base_env import BoxPusingEnv
from humanoid_swe_challenge.sims.pusher_manip.base_env import PusherManipEnv


env = PusherManipEnv()
mujoco.viewer.launch(env.model, env.data)