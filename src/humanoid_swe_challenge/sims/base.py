import gymnasium as gym
import mujoco as mj
import cv2
import atexit
import time
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any

from mujoco import viewer
from humanoid_swe_challenge.sims.config.video_cfg import VideoCfg

class BaseEnv(gym.Env):
    physics_fps = 120
    decimation = 5
    def __init__(self, 
                 task_name :str,
                 mjcf_path: str,
                 video_cfg: VideoCfg | None = None,
                 render_mode: str | None = None,
                 render_realtime:bool = False,
                 log_path: str | None = None) -> None:
        
        super().__init__()

        self.taskname = task_name
        self.model = mj.MjModel.from_xml_path(mjcf_path)# type: ignore
        self.data = mj.MjData(self.model) # type: ignore        
        self.model.opt.timestep = 1/self.physics_fps


        self.render_mode = render_mode
        self.render_realtime = render_realtime

        self.record_video = False
        if video_cfg:
            Path(video_cfg.video_path).mkdir(parents=True, exist_ok=True)
            self.record_video = True
            self._frame_buffer: list = []
            self.video_path = os.path.join(video_cfg.video_path,video_cfg.video_name)
            self.video_size = video_cfg.video_size
            self.fourcc = video_cfg.fourcc
            self._renderer = mj.Renderer(self.model, width=self.video_size[0], height=self.video_size[1])
            atexit.register(self._save_video)

        self.log_steps=False
        if log_path:
            curr_time = str(datetime.now().strftime("%H-%M-%S"))
            curr_date = str(datetime.now().strftime("%d-%m-%Y"))
            log_path = os.path.join(log_path,curr_date)
            Path(log_path).mkdir(parents=True, exist_ok=True)
            log_name = curr_time+".npy"
            

            self.log_steps=True
            self._step_history = []
            self.log_path = os.path.join(log_path,log_name)
            atexit.register(self._save_steps)


    def get_current_frame(self):
        if not hasattr(self, "_renderer") or self._renderer is None:
            self._renderer = mj.Renderer(self.model, width=640, height=480)
        self._renderer.update_scene(self.data, camera=-1)

        return self._renderer.render()
    
    def _buffer_video(self):
        frame = self.get_current_frame()
        self._frame_buffer.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return frame

    def _save_video(self):
        if self.record_video and self._frame_buffer:
            fourcc = cv2.VideoWriter.fourcc(*self.fourcc)
            writer = cv2.VideoWriter(filename=self.video_path, 
                                    fourcc=fourcc, 
                                    fps=self.physics_fps / self.decimation, 
                                    frameSize=self.video_size)
            for frame in self._frame_buffer:
                writer.write(frame)
            writer.release()
    
    def _save_steps(self):
        if self.log_steps:
            np.save(file=self.log_path, arr=np.array(self._step_history))
            
    def render(self):
        if self.record_video:
            frame = self._buffer_video()

        if self.render_mode == "rgb_array":
            if not self.record_video:
                frame = self.get_current_frame()
            cv2.imshow("box_pushing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return frame
        
        elif self.render_mode == "human":
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            if self.render_realtime:
                time.sleep(self.decimation/self.physics_fps)
            return None

        return None

    def step(self, action: np.ndarray):
        self.data.ctrl = action.copy()
        for _ in range(self.decimation):
            mj.mj_step(self.model, self.data)# type: ignore
        
        if self.log_steps:
            self._step_history.append(action)
        
        self.render()

    def close(self) -> None:
        if hasattr(self, "_viewer") and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        return super().close()