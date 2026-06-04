import signal
import numpy as np
import functools
from fastmcp import FastMCP

from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.mcp.utils import obs_to_dict, frame_to_base64


import logging
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
    """Start the box-pusing simulation in MuJoCo. If simulation is initialise successfuly the tool will return a render image in Base 64 of the current environment"""
    global ENV
    if ENV is not None:
        return "Simulation already running."
    ENV = BoxPusingEnv()
    ENV.reset()
    return get_visual()

@mcp.tool
@require_simulation
def close_simulation():
    """Close the box-pusing simulation."""
    ENV.close()
    return "Simulation terminated."

@mcp.tool
@require_simulation
def get_observation():
    """Return the current simulation state in a dictionary contains: pusher xyz, box xyz/yaw, xyz/yaw for three goals with colour red/green/blue and whether the pusher is currently in contact with the box."""
    return obs_to_dict(ENV.get_obs())

@mcp.tool
@require_simulation
def get_visual() -> dict:
    """Return a rendered image in Base 64 of the state of the current environment"""
    print("get visual called")
    frame_b64=frame_to_base64(ENV.get_current_frame())
    return {
        "type": "image",
        "media_type": "image/jpeg",
        "data": frame_b64,
    }

@mcp.tool
@require_simulation
def control_pusher(vx: float, vy: float, vz: float, duration: int):
    """
    Apply linear velocity control to the pusher end effector for duration simulation steps.
    vx, vy, vz: linear velocity in [-1.000, 1.000], can take up to 8 decimal places. 
    duration:  duration=1 applys the control signal once and advance the simulation for approximately 5/120 s, duration is capped at 100. 
    prioritise reducing duration for precise control. 
    Returns updated observation.
    """
    obs,_,_,_, info = ENV.step(action=np.array([vx,vy,vz]),step_count=duration)
    return obs_to_dict(obs)

# @mcp.tool
# @require_simulation
# def reset_simulation() -> dict:
#     """Reset the simulation to its initial state. Returns the initial observation."""
#     obs, _ = ENV.reset()
#     return _obs_to_dict(obs)

@mcp.tool
def get_simulation_description() -> str:
    """Describe the simulation environment."""
    return (
        "Push the box with a pusher to align with one of the goal positions (red/green/blue). "+
        "Only pushing is allowed."
        "The box and goals each have an xyz position and yaw angle measured from the center of the shapes. "+
        "The box has xyz dimensions about 0.12 0.1 0.07. "+
        "The pusher is a sphere with radius about 0.0125 you move controling its velocity. "+
        "The pusher and the box is not attached to each other, they are both their own objects. "+
        "The box is resting on a flat serface, can cannot be lifted with the pusher"
        "Simulation must be closed after a task is completed"
    )

def main():
    def _shutdown(signum, frame):
        if ENV is not None:
            ENV.save_video()

    signal.signal(signal.SIGTERM, _shutdown)
    mcp.run(transport="stdio")
    # mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)

if __name__ == "__main__":
    main()