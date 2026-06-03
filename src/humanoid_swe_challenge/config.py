import os

# MCP_Server
MCP_HOST="localhost"
MCP_PORT=8000

# Local_LM_Studio
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = "qwen/qwen3.6-35b-a3b"
LLM_API_KEY = "not-needed"

# MuJoCo configuration
class SimCfg():
    physics_fps = 120
    decimation = 5

# Video recording options
class VideoCfg():
    record_video = True
    video_size = (640, 480)
    video_path = "video/recording.mp4"

# Task configuration for Pusher Manipulation
class PusherManipEnvCfg():
    import numpy as np

    task_name: str = "Pusher Manipulation"
    random_goal_pose: bool = True
    random_goal_pose_low = np.array([-0.3, -0.3, 0.05])
    random_goal_pose_high = np.array([0.3, 0.3, 0.65])

    sim_cfg = SimCfg()
    video_cfg = VideoCfg()

class BoxPushingEnvCfg():
    # import numpy as np

    task_name: str = "Box Pushing"
    # random_goal_pose: bool = True
    # random_goal_pose_low = np.array([-0.3, -0.3, 0.05])
    # random_goal_pose_high = np.array([0.3, 0.3, 0.65])

    sim_cfg = SimCfg()
    video_cfg = VideoCfg()

