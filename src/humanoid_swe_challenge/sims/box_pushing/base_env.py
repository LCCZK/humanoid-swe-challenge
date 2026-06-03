import atexit
import gymnasium as gym
import mujoco as mj
import numpy as np
import time
import cv2
from pathlib import Path
from mujoco import viewer

from humanoid_swe_challenge.sims.utils import quat_to_yaw
from humanoid_swe_challenge.sims.box_pushing.env_config import BoxPushingEnvCfg

class BoxPusingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "record_video"], "render_fps": 120}

    def __init__(self, render_mode: str | None = None, video_path: str = "video/recording.mp4") -> None:
        super().__init__()

        self.cfg = BoxPushingEnvCfg()

        self.model = mj.MjModel.from_xml_path(str(Path(__file__).parent / "mjcf_box_pushing.xml"))# type: ignore
        self.data = mj.MjData(self.model) # type: ignore

        
        self.model.opt.timestep = self.cfg.dt
        self.render_mode = render_mode
        self.video_path = video_path

        self.id_box = self.model.body(self.cfg.box_body_name).id
        self.id_pusher = self.model.body(self.cfg.pusher_body_name).id
        self.id_goal_r = self.model.body(self.cfg.goal_r_body_name).id
        self.id_goal_g = self.model.body(self.cfg.goal_g_body_name).id
        self.id_goal_b = self.model.body(self.cfg.goal_b_body_name).id

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "pusher_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "box_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "box_yaw": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "goal_red_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal_red_yaw": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "goal_green_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal_green_yaw": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "goal_blue_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal_blue_yaw": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            "pusher_in_contact_with_box": gym.spaces.MultiBinary(1)
        })

        self._obs = {}
        self._frames: list = []

        if render_mode == "record_video":
            atexit.register(self.save_video)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        mj.mj_resetData(self.model, self.data)# type: ignore
        mj.mj_forward(self.model, self.data)# type: ignore
        self.render()

        return self.get_obs(), {}
        
    def step(self, action: np.ndarray, step_count:int = 1):
        step_count = min(step_count,10)
        for s in range(step_count):
            self.data.ctrl = action.copy()
            for _ in range(self.cfg.decimation):
                mj.mj_step(self.model, self.data)# type: ignore
            self.render()
        self._step_count += step_count
        
        self.save_video()
        return self.get_obs(), 0, False, False, {}

    def check_box_pusher_contact(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if ((contact.geom1 == self.id_pusher and contact.geom2 == self.id_box) or 
                (contact.geom1 == self.id_box and contact.geom2 == self.id_pusher)):
                return True
        return False


    def get_obs(self):
        return {"pusher_xyz": self.data.xpos[self.id_pusher][:3].copy(),
                "box_xyz": self.data.xpos[self.id_box][:3].copy(),
                "box_yaw": quat_to_yaw(self.data.xquat[self.id_box].copy()),
                "goal_red_xyz": self.data.xpos[self.id_goal_r][:3].copy(),
                "goal_red_yaw": quat_to_yaw(self.data.xquat[self.id_goal_r].copy()),
                "goal_green_xyz": self.data.xpos[self.id_goal_g][:3].copy(),
                "goal_green_yaw": quat_to_yaw(self.data.xquat[self.id_goal_g].copy()),
                "goal_blue_xyz": self.data.xpos[self.id_goal_b][:3].copy(),
                "goal_blue_yaw": quat_to_yaw(self.data.xquat[self.id_goal_b].copy()),
                "pusher_in_contact_with_box": self.check_box_pusher_contact()}
    
    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            return None

        elif self.render_mode == "rgb_array":
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mj.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=-1)
            frame = self._renderer.render()
            cv2.imshow("box_pushing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return frame
        
        elif self.render_mode == "record_video":
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mj.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=-1)
            frame = self._renderer.render()
            self._frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return frame

        return None

    def save_video(self):
        if not self._frames:
            return
        fourcc = cv2.VideoWriter.fourcc(*"avc1")
        writer = cv2.VideoWriter(self.video_path, fourcc, 30, (640, 480))
        for frame in self._frames:
            writer.write(frame)
        writer.release()
    
    def close(self) -> None:
        self.save_video()
        return super().close()
