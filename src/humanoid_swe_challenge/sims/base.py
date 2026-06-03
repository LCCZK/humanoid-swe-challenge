import gymnasium as gym
import mujoco as mj
import cv2
import atexit
from typing import Any

from mujoco import viewer
from humanoid_swe_challenge.config import VideoCfg, SimCfg

class BaseEnv(gym.Env):
    def __init__(self, 
                 task_name :str,
                 mjcf_path: str,
                 sim_cfg: SimCfg,
                 video_cfg: VideoCfg,
                 render_mode: str | None = None) -> None:
        
        super().__init__()

        self.taskname = task_name
        self.model = mj.MjModel.from_xml_path(mjcf_path)# type: ignore
        self.data = mj.MjData(self.model) # type: ignore
        
        self.physics_fps = sim_cfg.physics_fps
        self.model.opt.timestep = 1/self.physics_fps
        self.decimation = sim_cfg.decimation

        self.render_mode = render_mode
        self.record_video = video_cfg.record_video
        self.video_path = video_cfg.video_path
        self.video_size = video_cfg.video_size
        self._frames: list = []
        self._renderer = None

        if self.record_video:
            atexit.register(self.save_video)
            self._renderer = mj.Renderer(self.model, width=self.video_size[0], height=self.video_size[1])

    def get_current_frame(self):
        if not hasattr(self, "_renderer") or self._renderer is None:
            self._renderer = mj.Renderer(self.model, width=self.video_size[0], height=self.video_size[1])
        self._renderer.update_scene(self.data, camera=-1)

        return self._renderer.render()
    
    def buffer_video(self):
        frame = self.get_current_frame()
        self._frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return frame

    def save_video(self):
        if not self._frames:
            return
        fourcc = cv2.VideoWriter.fourcc(*"avc1")
        writer = cv2.VideoWriter(filename=self.video_path, 
                                 fourcc=fourcc, 
                                 fps=self.physics_fps / self.decimation, 
                                 frameSize=self.video_size)
        for frame in self._frames:
            writer.write(frame)
        writer.release()
    
    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            if self.record_video:
                self.buffer_video()
            return None

        elif self.render_mode == "rgb_array":
            if self.record_video:
                frame = self.buffer_video()
            else:
                frame = self.get_current_frame()
            cv2.imshow("box_pushing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return frame
        
        return None

    def step(self, action: Any):
        self.data.ctrl = action.copy()
        for _ in range(self.decimation):
            mj.mj_step(self.model, self.data)# type: ignore
        self.render()

    def close(self) -> None:
        if self.record_video:
            self.save_video()
        return super().close()