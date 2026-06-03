import signal
import numpy as np
import functools
from fastmcp import FastMCP
from humanoid_swe_challenge.sims.box_pushing.env import BoxPusingEnv
from humanoid_swe_challenge.config import MCP_HOST, MCP_PORT

# import sys
# import logging
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

mcp = FastMCP("Tools for Box-Pushing simulation")

ENV: BoxPusingEnv | None = None
def require_simulation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ENV is None:
            return "Simulation not running. Call start_simulation() first."
        return func(*args, **kwargs)
    return wrapper

def _obs_to_dict(obs: dict) -> dict:
    obs_c = obs.copy()
    for  k, v in obs_c.items():
        if isinstance(v, np.ndarray):
            print("np_array")
            obs_c[k]=v.tolist()
            
    return obs_c

@mcp.tool
def start_simulation():
    """Start the box-pusing simulation."""
    global ENV
    if ENV is not None:
        return "Simulation already running."
    ENV = BoxPusingEnv(render_mode="record_video")
    ENV.reset()
    return "Simulation started."

@mcp.tool
@require_simulation
def close_simulation():
    """Close the box-pusing simulation."""
    ENV.close()
    return "Simulation terminated."

@mcp.tool
@require_simulation
def get_observation():
    """Return the current simulation state: pusher xyz, box xyz/yaw, xyz/yaw for three goals with colour red/green/blue and whether the pusher is currently in contact with the box."""
    return _obs_to_dict(ENV.get_obs())

@mcp.tool
@require_simulation
def control_pusher(vx: float, vy: float, vz: float, duration: int):
    """
    Apply velocity control to the pusher end effector for duration simulation steps.
    vx, vy, vz: velocity in [-1.0, 1.0]. (duration=1 is approximately 5/120 s), duration is capped at 10
    Returns updated observation.
    """
    obs,_,_,_, info = ENV.step(action=np.array([vx,vy,vz]),step_count=duration)
    return _obs_to_dict(obs)

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
    # mcp.run(transport="stdio")
    mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)

if __name__ == "__main__":
    main()