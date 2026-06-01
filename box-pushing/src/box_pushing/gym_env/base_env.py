import gymnasium as gym
import mujoco as mj
import numpy as np
import time

from box_pushing.mj_scene.scene import PushingScene
from box_pushing.gym_env.env_config import PushingEnvCfg
from box_pushing.gym_env.util import quat_to_yaw

class PusingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 120}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        self.cfg = PushingEnvCfg()
        self.scene = PushingScene()
        self.scene.model.opt.timestep = self.cfg.dt
        self.render_mode = render_mode

        self.id_box = self.scene.model.body(self.cfg.box_body_name).id
        self.id_pusher = self.scene.model.body(self.cfg.pusher_body_name).id
        self.id_goal_r = self.scene.model.body(self.cfg.goal_r_body_name).id
        self.id_goal_g = self.scene.model.body(self.cfg.goal_g_body_name).id
        self.id_goal_b = self.scene.model.body(self.cfg.goal_b_body_name).id

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
        mj.mj_resetData(self.scene.model, self.scene.data)# type: ignore
        mj.mj_forward(self.scene.model, self.scene.data)# type: ignore
        self.scene.render(self.render_mode)

        return self.get_obv(), {}
        
    def step(self, action: np.ndarray, step_count:int = 1):
        for s in range(step_count):
            self.scene.data.ctrl = action.copy()
            for _ in range(self.cfg.decimation):
                mj.mj_step(self.scene.model, self.scene.data)# type: ignore
            self.render
            time.sleep(1/120)
        self._step_count += step_count
        
        return self.get_obv(), 0, False, False, {}
    
    def get_obv(self):
        return {"pusher_xyz": self.scene.data.xpos[self.id_pusher][:3].copy(),
                "box_xyz": self.scene.data.xpos[self.id_box][:3].copy(),
                "box_yaw": quat_to_yaw(self.scene.data.xquat[self.id_box].copy()),
                "goal_red_xyz": self.scene.data.xpos[self.id_goal_r][:3].copy(),
                "goal_red_yaw": quat_to_yaw(self.scene.data.xquat[self.id_goal_r].copy()),
                "goal_green_xyz": self.scene.data.xpos[self.id_goal_g][:3].copy(),
                "goal_green_yaw": quat_to_yaw(self.scene.data.xquat[self.id_goal_g].copy()),
                "goal_blue_xyz": self.scene.data.xpos[self.id_goal_b][:3].copy(),
                "goal_blue_yaw": quat_to_yaw(self.scene.data.xquat[self.id_goal_b].copy())}
    
    def render(self):
        return self.scene.render(render_mode=self.render_mode)
