import gymnasium as gym
import mujoco as mj
import numpy as np
import time
import cv2
from pathlib import Path
from mujoco import viewer

from humanoid_swe_challenge.sims.utils import quat_to_yaw
from humanoid_swe_challenge.sims.box_pushing.env_config import PushingEnvCfg

class PusingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.cfg = PushingEnvCfg()

        self.model = mj.MjModel.from_xml_path(str(Path(__file__).parent / "box_pushing_scene.xml"))# type: ignore
        self.data = mj.MjData(self.model) # type: ignore

        
        self.model.opt.timestep = self.cfg.dt
        self.render_mode = render_mode

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
        })

        self._obs = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        mj.mj_resetData(self.model, self.data)# type: ignore
        mj.mj_forward(self.model, self.data)# type: ignore
        self.render()

        return self.get_obs(), {}
        
    def step(self, action: np.ndarray, step_count:int = 1):
        for s in range(step_count):
            self.data.ctrl = action.copy()
            for _ in range(self.cfg.decimation):
                mj.mj_step(self.model, self.data)# type: ignore
            self.render()
        self._step_count += step_count
        
        return self.get_obs(), 0, False, False, {}
    
    def get_obs(self):
        return {"pusher_xyz": self.data.xpos[self.id_pusher][:3].copy(),
                "box_xyz": self.data.xpos[self.id_box][:3].copy(),
                "box_yaw": quat_to_yaw(self.data.xquat[self.id_box].copy()),
                "goal_red_xyz": self.data.xpos[self.id_goal_r][:3].copy(),
                "goal_red_yaw": quat_to_yaw(self.data.xquat[self.id_goal_r].copy()),
                "goal_green_xyz": self.data.xpos[self.id_goal_g][:3].copy(),
                "goal_green_yaw": quat_to_yaw(self.data.xquat[self.id_goal_g].copy()),
                "goal_blue_xyz": self.data.xpos[self.id_goal_b][:3].copy(),
                "goal_blue_yaw": quat_to_yaw(self.data.xquat[self.id_goal_b].copy())}
    
    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "_viewer") or self._viewer is None:
                self._viewer = viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
            time.sleep(1/120)
            return None

        elif self.render_mode == "rgb_array":
            if not hasattr(self, "_renderer") or self._renderer is None:
                self._renderer = mj.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data, camera=-1)
            frame = self._renderer.render()
            cv2.imshow("box_pushing", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            return frame

        return None
