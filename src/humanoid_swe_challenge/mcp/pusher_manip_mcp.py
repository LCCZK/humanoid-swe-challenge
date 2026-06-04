import signal
import numpy as np
import functools
from fastmcp import FastMCP

from humanoid_swe_challenge.config import LOG_PATH,NP_RANDOM_SEED
from humanoid_swe_challenge.sims.pusher_manip.env import PusherManipEnv
from humanoid_swe_challenge.mcp.utils import obs_to_dict

import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger(__name__)
# logger.propagate = False



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
def start_simulation() -> dict|str:
    """Start the pusher manipulation simulation in MuJoCo. Returns the initial observation dict so you immediately know the pusher and all goal positions. Always read the returned positions before planning."""
    global ENV
    if ENV is not None:
        return "Simulation already running."
    ENV = PusherManipEnv(log_path=LOG_PATH)
    obs, _ = ENV.reset(seed=NP_RANDOM_SEED)
    return obs_to_dict(obs)

@mcp.tool
@require_simulation
def close_simulation() -> str:
    """Close the pusher manipulation simulation. Call this after all goals have been reached."""
    ENV.close()
    return "Simulation terminated."

@mcp.tool
@require_simulation
def get_observation() -> dict|str:
    """Return the current simulation state as a dictionary with keys:
    - pusher_xyz: [x, y, z] position of the pusher in metres
    - goal_red_xyz: [x, y, z] position of the red goal in metres
    - goal_green_xyz: [x, y, z] position of the green goal in metres
    - goal_blue_xyz: [x, y, z] position of the blue goal in metres
    Coordinate frame: x=right(+)/left(-), y=forward(+)/backward(-), z=up(+)/down(-)."""
    obs = ENV.get_obs()
    logger.debug(msg="get_observation called")
    return obs_to_dict(obs)

@mcp.tool
@require_simulation
def control_pusher(vx: float, vy: float, vz: float, step_size: int)  -> dict|str:
    """
    Apply linear velocity control to the pusher for step_size simulation steps, then return the updated observation dict (same format as get_observation()).
    vx, vy, vz: velocity in metres/s, range [-1.0, 1.0], up to 8 decimal places.
    step_size: number of control steps to apply; each step advances the simulation by ~0.042 s (5/120 s). Capped at 100.
    Setting vx=vy=vz=0 brakes the pusher and holds it in place.
    Use small step_size (5-20) for precise control near a goal; use large step_size (30-50) when moving in open space.
    """
    obs,_,_,_, info = ENV.step(action=np.array([vx,vy,vz]), step_count=step_size)
    return obs_to_dict(obs)

@mcp.tool
def get_simulation_description() -> str:
    """Describe the simulation environment."""
    return (
        "Task: move the pusher sequentially to each goal in the order specified by the user prompt. " +
        "Always read positions from the observation before planning. " +
        "The pusher is controlled by xyz linear velocity. " +
        "SUCCESS CONDITION: pusher position must be within 0.005 m of the goal in ALL THREE axes (x, y, and z). The z axis matters — you must match the goal height precisely. " +
        "Coordinate frame: x=right(+)/left(-), y=forward(+)/backward(-), z=up(+)/down(-). " +
        "There is no visual render tool — use get_observation() for all navigation. " +
        "Simulation must be started before any interactions and closed after all goals are reached."
    )

def main():
    mcp.run(transport="stdio")
    # mcp.run(transport="streamable-http", host=MCP_HOST, port=MCP_PORT)

if __name__ == "__main__":
    main()