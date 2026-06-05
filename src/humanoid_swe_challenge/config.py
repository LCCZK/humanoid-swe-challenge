import os

# LLM Configuration
LLM_URL = os.environ.get("LLM_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.6-35b-a3b:2")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
LLM_TOKEN_LIMIT = -1


LOG_PATH="log" # Path for action logging
NP_RANDOM_SEED=42 # Random seed used by the pusher manipulation task to generate random goal


# ==================Task 1: Pusher Manipulation ========================

MCP_SERVER_COMMAND = "pusher-manip-mcp"
USER_PROMPT = "run the pusher manipulation simulation, check the simulation description. Complete the manipulation task, move the pusher to where the blue goal is. After that moved to the red goal. After that moved to the green goal."


# ==================Task 2: Box Pushing ========================

# MCP_SERVER_COMMAND = "box-pushing-mcp"
# USER_PROMPT = """Run the 3D box-pushing simulation, then push the purple box to overlap with the blue goal.

# SETUP: Call start_simulation(), then get_simulation_description(), then get_observation() to know exact starting positions.

# COORDINATE SYSTEM:
# - x-axis: right(+) / left(-), y-axis: forward(+) / backward(-), z-axis: up(+) / down(-)
# - SUCCESS: box_xyz matches goal_blue_xyz within ±0.03 in both x and y

# GEOMETRY — use these offsets for precise pusher positioning relative to box center:
# - Box half-extents: x=0.06, y=0.05, z=0.035; box center height z=0.045
# - Pusher radius: 0.015; keep pusher z=0.045 to push at box center height
# - To contact box from -y side (push in +y): set pusher_y = box_y - 0.07
# - To contact box from +x side (push in -x): set pusher_x = box_x + 0.08
# - To contact diagonally, position the pusher behind the box along the vector (goal - box)

# STRATEGY:
# 1. Compute the push direction: direction = goal_blue_xyz - box_xyz (x and y only). Normalise it.
# 2. Position pusher on the OPPOSITE side of the box from the goal:
#    a. First raise pusher above the box (vz>0, a few steps) to avoid side-contact while re-positioning.
#    b. Move pusher laterally to the back-of-box position (pusher = box_xyz - direction * 0.09).
#    c. Lower pusher back to z≈0.045, then nudge forward until pusher_in_contact_with_box is True.
# 3. Push toward the goal: apply vx, vy proportional to the direction vector, use step_size=15-25.
# 4. After every 2-3 pushes: call get_observation(), recompute direction and re-check contact.
# 5. Use lower velocities (0.1-0.2) when box is within 0.1 of the goal to avoid overshooting — the box has very low friction and slides easily.
# 6. If the box overshoots or drifts off-axis, re-position the pusher on the new required side and correct.
# 7. When success condition is met, call close_simulation().

# TIPS:
# - Prefer get_observation() over visuals for precise position feedback; use get_visual() to confirm contact angle after repositioning.
# - When moving the pusher in open space (no box contact needed): use velocity 0.5-1.0 and step_size=30-50 for speed.
# - The pusher sphere (orange) is visible in the render; the box is purple.
# - Multiple repositioning cycles will be needed — this is expected and the task is achievable."""

