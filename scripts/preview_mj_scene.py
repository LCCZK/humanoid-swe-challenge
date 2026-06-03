import mujoco
import mujoco.viewer

from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.sims.pusher_manip.env import PusherManipEnv


env = BoxPusingEnv()
mujoco.viewer.launch(env.model, env.data)