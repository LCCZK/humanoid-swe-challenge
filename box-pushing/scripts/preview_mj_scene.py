import mujoco
import mujoco.viewer

from box_pushing.mj_scene.scene import PushingScene

scene = PushingScene()
mujoco.viewer.launch(scene.model, scene.data)