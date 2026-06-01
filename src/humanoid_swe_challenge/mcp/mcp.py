import numpy as np
import functools
from fastmcp import FastMCP
from humanoid_swe_challenge.sims.box_pushing.base_env import PusingEnv

# import sys
# import logging
# logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

mcp = FastMCP("Tools for Box-Pushing simulation")

ENV: PusingEnv | None = None
def require_simulation(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ENV is None:
            return "Simulation not running. Call start_simulation() first."
        return func(*args, **kwargs)
    return wrapper

def _obs_to_dict(obs: dict) -> dict:
    return {k: v.tolist() for k, v in obs.items()}

@mcp.tool
def start_simulation():
    """Start the box-pusing simulation."""
    global ENV
    if ENV is not None:
        return "Simulation already running."
    ENV = PusingEnv(render_mode=None)
    ENV.reset()
    return "Simulation started."

@mcp.tool
@require_simulation
def get_observation():
    """Return the current simulation state: pusher xyz, box xyz/yaw, and all three goal poses."""
    return _obs_to_dict(ENV.get_obs())

@mcp.tool
@require_simulation
def control_pusher(vx: float, vy: float, vz: float, duration: int):
    """
    Apply velocity to the pusher for `duration` simulation steps.
    vx, vy, vz: velocity in [-1.0, 1.0]. duration: number of steps (1 step ≈ 5/120 s).
    Returns updated observation.
    """
    obs,_,_,_, info = ENV.step(action=np.array([vx,vy,vz]),step_count=duration)
    return _obs_to_dict(obs)

@mcp.tool
@require_simulation
def reset_simulation() -> dict:
    """Reset the simulation to its initial state. Returns the initial observation."""
    obs, _ = ENV.reset()
    return _obs_to_dict(obs)

@mcp.tool
def get_simulation_description() -> str:
    """Describe the simulation environment."""
    return (
        "Push the box to align with one of the goal positions (red/green/blue). "
        "The box and goals each have an xyz position and yaw angle. "
        "The pusher is a sphere you move via velocity commands. "
        "Workspace is roughly +-0.4m in x and +-1.0m in y. "
    )

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()