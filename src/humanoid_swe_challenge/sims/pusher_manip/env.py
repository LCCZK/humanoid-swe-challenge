import gymnasium as gym
import mujoco as mj
import numpy as np
from pathlib import Path

from humanoid_swe_challenge.sims.pusher_manip.env_config import PusherManipEnvCfg
from humanoid_swe_challenge.sims.base import BaseEnv

class PusherManipEnv(BaseEnv):
    mjcf_path: str = str(Path(__file__).parent / "mjcf_pusher_manip.xml")

    #pusher
    pusher_body_name: str = "pusher"
    pusher_joint_x_name: str = "pusher_x"
    pusher_joint_y_name: str = "pusher_y"
    pusher_joint_z_name: str = "pusher_z"
    pusher_radius = 0.0125

    #goal
    goal_r_body_name: str = "goal_r"
    goal_g_body_name: str = "goal_g"
    goal_b_body_name: str = "goal_b"

    def __init__(self, render_mode: str | None = None) -> None:

        self.cfg = PusherManipEnvCfg()

        super().__init__(task_name=self.cfg.task_name,
                         mjcf_path=self.mjcf_path,
                         sim_cfg = self.cfg.sim_cfg,
                         video_cfg=self.cfg.video_cfg,
                         render_mode=render_mode)
        
        self.id_pusher = self.model.body(self.pusher_body_name).id
        self.id_goal_r = self.model.body(self.goal_r_body_name).id
        self.id_goal_g = self.model.body(self.goal_g_body_name).id
        self.id_goal_b = self.model.body(self.goal_b_body_name).id

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "pusher_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal_red_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal_green_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "goal_blue_xyz": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })

        self._obs = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        mj.mj_resetData(self.model, self.data)# type: ignore

        if self.cfg.random_goal_pose:
            self._randomise_goals(self.id_goal_r)
            self._randomise_goals(self.id_goal_g)
            self._randomise_goals(self.id_goal_b)

        mj.mj_forward(self.model, self.data)# type: ignore
        self.render()

        return self.get_obs(), {}
        
    def step(self, action: np.ndarray, step_count:int = 1):
        step_count = min(step_count,10)
        for _ in range(step_count):
            super().step(action)
        self._step_count += step_count

        if self.record_video:
           self.save_video()

        return self.get_obs(), 0, False, False, {}
    
    def _randomise_goals(self, goal_id):
        goal_xyz = np.random.rand(1,3) * (self.cfg.random_goal_pose_high - self.cfg.random_goal_pose_low) + self.cfg.random_goal_pose_low
        self._set_goal(goal_xyz, goal_id)

    def _set_goal(self, goal_xyz, goal_id):
        mid = self.model.body_mocapid[goal_id]
        self.data.mocap_pos[mid][:3] = goal_xyz.copy()
    
    def get_obs(self):
        return {"pusher_xyz": self.data.xpos[self.id_pusher][:3].copy(),
                "goal_red_xyz": self.data.xpos[self.id_goal_r][:3].copy(),
                "goal_green_xyz": self.data.xpos[self.id_goal_g][:3].copy(),
                "goal_blue_xyz": self.data.xpos[self.id_goal_b][:3].copy(),}

