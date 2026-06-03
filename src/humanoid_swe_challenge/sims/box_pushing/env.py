import gymnasium as gym
import mujoco as mj
import numpy as np
from pathlib import Path

from humanoid_swe_challenge.sims.utils import quat_to_yaw
from humanoid_swe_challenge.config import BoxPushingEnvCfg
from humanoid_swe_challenge.sims.base import BaseEnv

class BoxPusingEnv(BaseEnv):
    mjcf_path: str = str(Path(__file__).parent / "mjcf_box_pushing.xml")
    #pusher
    pusher_body_name = "pusher"
    pusher_joint_x_name = "pusher_x"
    pusher_joint_y_name = "pusher_y"
    pusher_joint_z_name = "pusher_z"

    #box
    box_body_name = "box"
    box_random_init_pos = False

    #goal
    goal_r_body_name = "goal_r"
    goal_g_body_name = "goal_g"
    goal_b_body_name = "goal_b"
    goal_pos_only = False
    random_goal_pose = False
    random_goal_pose_range = []

    def __init__(self, render_mode: str | None = None) -> None:
        
        self.cfg = BoxPushingEnvCfg()
        super().__init__(task_name=self.cfg.task_name,
                         mjcf_path=self.mjcf_path,
                         sim_cfg = self.cfg.sim_cfg,
                         video_cfg=self.cfg.video_cfg,
                         render_mode=render_mode)

        self.id_box = self.model.body(self.box_body_name).id
        self.id_pusher = self.model.body(self.pusher_body_name).id
        self.id_goal_r = self.model.body(self.goal_r_body_name).id
        self.id_goal_g = self.model.body(self.goal_g_body_name).id
        self.id_goal_b = self.model.body(self.goal_b_body_name).id

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        mj.mj_resetData(self.model, self.data)# type: ignore
        mj.mj_forward(self.model, self.data)# type: ignore
        self.render()

        return self.get_obs(), {}
        
    def step(self, action: np.ndarray, step_count:int = 1):
        step_count = min(step_count, self.cfg.max_step_duration)
        for _ in range(step_count):
            super().step(action)
        self._step_count += step_count

        if self.record_video:
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
