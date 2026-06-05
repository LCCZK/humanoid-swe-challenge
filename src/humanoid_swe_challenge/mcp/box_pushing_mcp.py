import signal
import numpy as np
import functools
from fastmcp import FastMCP

from humanoid_swe_challenge.config import LOG_PATH
from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.mcp.utils import obs_to_dict, frame_to_base64


import logging, sys
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.propagate = False


mcp = FastMCP("Tools for Box-Pushing simulation")

ENV: BoxPusingEnv | None = None
def require_simulation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ENV is None:
            return "Simulation not running. Call start_simulation() first."
        return func(*args, **kwargs)
    return wrapper

@mcp.tool
def start_simulation():
    """Start the box-pushing simulation in MuJoCo. Returns a rendered image of the initial environment for spatial orientation. If the simulation is already running, returns an error message."""
    global ENV
    if ENV is not None:
        return "Simulation already running."
    ENV = BoxPusingEnv(log_path=LOG_PATH)
    ENV.reset()
    return get_visual()

@mcp.tool
@require_simulation
def close_simulation():
    """Close the box-pushing simulation. Call this after the task is complete."""
    ENV.close()
    return "Simulation terminated."

@mcp.tool
@require_simulation
def get_observation():
    """Return the current simulation state as a dictionary with keys:
    - pusher_xyz: [x, y, z] position of the pusher in metres
    - box_xyz: [x, y, z] position of the box centre in metres
    - box_yaw: box rotation around z-axis in radians
    - goal_red_xyz, goal_green_xyz, goal_blue_xyz: [x, y, z] goal centre positions in metres
    - goal_red_yaw, goal_green_yaw, goal_blue_yaw: goal yaw in radians
    - pusher_in_contact_with_box: True if the pusher is currently touching the box, False otherwise
    Coordinate frame: x=right(+)/left(-), y=forward(+)/backward(-), z=up(+)/down(-)."
    Use this tool for precise position-based planning and to verify contact before pushing."""
    return obs_to_dict(ENV.get_obs())

@mcp.tool
@require_simulation
def get_visual() -> dict:
    """Return a rendered image of the current environment. The orange sphere is the pusher, the purple box is the target object, and the semi-transparent red/green/blue boxes are the goal regions. Use this to verify pusher contact angle and spatial layout after repositioning. For precise positions, prefer get_observation()."""
    frame_b64=frame_to_base64(ENV.get_current_frame())
    return {
        "type": "image",
        "media_type": "image/jpeg",
        "data": frame_b64,
    }

@mcp.tool
@require_simulation
def control_pusher(vx: float, vy: float, vz: float, step_size: int):
    """
    Apply linear velocity control to the pusher for step_size simulation steps, then return the updated observation dict (same format as get_observation()).
    vx, vy, vz: velocity in metres/s, range [-1.0, 1.0], up to 8 decimal places.
    step_size: number of control steps to apply; each step advances the simulation by ~0.042 s (5/120 s). Capped at 100.
    Check pusher_in_contact_with_box in the returned observation before applying a sustained push.
    """
    obs,_,_,_, info = ENV.step(action=np.array([vx,vy,vz]),step_count=step_size)
    return obs_to_dict(obs)

@mcp.tool
def get_simulation_description() -> str:
    """Describe the simulation environment."""
    return (
        "Task: push the purple box with the pusher to align it with the BLUE goal. "+
        "The box and goals each have an xyz position and yaw angle measured from the centre of the shapes. "+
        "The box has xyz full dimensions approximately 0.12 x 0.10 x 0.07 (half-extents: 0.06, 0.05, 0.035). "+
        "The pusher is an orange sphere with radius 0.015, controlled by velocity. "+
        "The pusher and box are separate objects; contact can break if the pusher moves away. "+
        "The workspace is approximately ±0.4 m in x and ±0.6 m in y. "+
        "Simulation must be closed after the task is completed."
    )

def main():
    mcp.run(transport="stdio")
    # mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)

if __name__ == "__main__":
    main()