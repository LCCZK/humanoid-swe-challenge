import numpy as np

class SimCfg():
    physics_fps = 120
    decimation = 5
    max_step_count = 1000

class VideoCfg():
    record_video = True
    video_size = (640, 480)
    video_path = "video/recording.mp4"

class PusherManipEnvCfg():
    task_name: str = "Pusher Manipulation"
    random_goal_pose: bool = True
    random_goal_pose_low = np.array([-0.3, -0.3, 0.05])
    random_goal_pose_high = np.array([0.3, 0.3, 0.65])

    sim_cfg = SimCfg()
    video_cfg = VideoCfg()

