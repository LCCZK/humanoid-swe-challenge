import mujoco as mj
import cv2
from pathlib import Path
from mujoco import viewer
import time

class PushingScene:
    def __init__(self):
        self.model = mj.MjModel.from_xml_path(str(Path(__file__).parent / "box_pushing_scene.xml"))# type: ignore
        self.data = mj.MjData(self.model) # type: ignore

    def render(self, render_mode):
        if render_mode == "human":
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        elif render_mode == "rgb_array":
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mj.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=-1)
            frame = self._renderer.render()
            cv2.imshow("box_pushing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return frame

        return None