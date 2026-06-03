import signal
import numpy as np
import functools
from fastmcp import FastMCP

from humanoid_swe_challenge.sims.pusher_manip.env import PusherManipEnv
from humanoid_swe_challenge.config import MCP_HOST, MCP_PORT
from humanoid_swe_challenge.mcp.utils import obs_to_dict

import sys
import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)



mcp = FastMCP("Tools for interacting with simulation for the pusher manipulation task in MuJoCo")

ENV: PusherManipEnv | None = None
def require_simulation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ENV is None:
            return "Simulation not running. Call start_simulation() first."
        return func(*args, **kwargs)
    return wrapper


@mcp.tool
def start_simulation() -> str:
    """Start a simulation for pusher manipulation task in MuJoCo."""
    global ENV
    if ENV is not None:
        return "Simulation already running."
    ENV = PusherManipEnv(render_mode="record_video")
    ENV.reset(seed=42)
    return "Simulation started."

@mcp.tool
@require_simulation
def close_simulation() -> str:
    """Close the pusher manipulation simulation."""
    ENV.close()
    return "Simulation terminated."

@mcp.tool
@require_simulation
def get_observation() -> dict|str:
    """
    Return the current observation from the simulation. Return dict contains the current xyz coordinate of the pusher, and the goals (red/green/blue).
    """
    obs = ENV.get_obs()
    logger.debug(msg="get_observation called")
    return obs_to_dict(obs)

@mcp.tool
@require_simulation
def control_pusher(vx: float, vy: float, vz: float, duration: int)  -> dict|str:
    """
    Apply linear velocity control to the pusher end effector for duration simulation steps.
    vx, vy, vz: linear velocity in [-1.000, 1.000], can take up to 8 decimal places. 
    duration:  duration=1 applys the control signal once and advance the simulation for approximately 5/120 s, duration is capped at 100. 
    Returns updated observation.
    """
    obs,_,_,_, info = ENV.step(action=np.array([vx,vy,vz]),step_count=duration)
    return obs_to_dict(obs)

@mcp.tool
def get_simulation_description() -> str:
    """Describe the simulation environment."""
    return (
        "Manipulate the xyz linear velocity of a pusher to align with one of the goal positions (red/green/blue). "+ 
        "Pusher is considered have reached the goal if the distance is less than 0.005 in all 3 x, y and z axis. " +
        "Simulation must be start before any interactions/observations and closed after a task is completed"
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