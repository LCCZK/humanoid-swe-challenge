import os

# Local_LM_Studio
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.6-35b-a3b:2")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")

MCP_SERVER_COMMAND = "pusher-manip-mcp"
USER_PROMPT = "run the pusher manipulation simulation, check the simulation description. Complete the manipulation task, move the pusher to where the blue goal is. After that moved to the red goal. After that moved to the green goal."

# MCP_SERVER_COMMAND = "box-pushing-mcp"
# USER_PROMPT = "run the 3D box-pushing simulation, check the simulation description. Then push the purple box into the blue goal. Use the image render of the environment to better understand the task. The pusher and the box can break contact, and the box will not move if the pusher is not in contact with the box. If contact is lost or you want to push the box to another direction, it is better to use the updated image render to try to re-establish contact an to confirom how the pusher is making contact with the box. Make sure to check the visual occasionally to update your understanding. Multiple actions can be applied, and the task is defintly achieveable. In the rendered image, the pusher is the sphere coloured orange"

# MuJoCo configuration
class SimCfg():
    physics_fps = 120
    decimation = 5

# Video recording options
class VideoCfg():
    record_video = True
    video_size = (640, 480)
    video_path = "video/recording.mp4"

# Task 1: Pusher Manipulation
class PusherManipEnvCfg():
    import numpy as np

    task_name: str = "Pusher Manipulation"
    random_goal_pose: bool = True
    random_goal_pose_low = np.array([-0.3, -0.3, 0.05])
    random_goal_pose_high = np.array([0.3, 0.3, 0.45])
    max_step_duration = 100

    sim_cfg = SimCfg()
    video_cfg = VideoCfg()

# Task 2: Box pushing
class BoxPushingEnvCfg():
    task_name: str = "Box Pushing"
    max_step_duration = 100

    sim_cfg = SimCfg()
    video_cfg = VideoCfg()
    

