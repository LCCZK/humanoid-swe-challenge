# Humanoid SWE Challenge

MuJoCo robotic simulation environments controlled by an LLM agent via the Model Context Protocol (MCP). The agent uses tool calls to observe and control simulations in a feedback loop until tasks are complete.

## Overview

Two simulation tasks are included:

- **Pusher Manipulation** — move a pusher to sequentially reach coloured goal positions (blue → red → green)
- **Box Pushing** — use the pusher to slide a purple box onto a blue goal region

The LLM agent connects to an MCP server (one per task) via stdio, receives tool descriptions, and drives the simulation step-by-step using position observations and velocity control.

## Architecture

- **`src/humanoid_swe_challenge/llm_agent/`** — Agent loop; calls the LLM with MCP tools, parses tool calls, trims context when it grows long
- **`src/humanoid_swe_challenge/mcp/`** — FastMCP servers exposing `start_simulation`, `get_observation`, `control_pusher`, `get_visual`, `close_simulation`
- **`src/humanoid_swe_challenge/sims/`** — Gymnasium environments wrapping MuJoCo; base class handles rendering, video recording, and action logging
- **`scripts/`** — playback scripts to replay logged action sequences with the MuJoCo viewer

## Installation

Requires Python ≥ 3.11 and a working MuJoCo installation.

```bash
pip install -e .
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_URL` | `http://localhost:1234/v1` | OpenAI-compatible API base URL |
| `LLM_MODEL` | `qwen/qwen3.6-35b-a3b:2` | Model name |
| `LLM_API_KEY` | `not-needed` | API key (if required) |
| `LLM_TOKEN_LIMIT` | `-1` | Max tokens for LLM responses |
| `MCP_SERVER_CMD` | `pusher-manip-mcp` | Command to start the MCP server subprocess |
| `USER_PROMPT` |  | Initial prompt for the LLM agent  |

The active MCP server command and user prompt are set in [src/humanoid_swe_challenge/config.py](src/humanoid_swe_challenge/config.py) the default configuration runs the pusher manipulation task. Uncomment the box-pushing lines and comment the pusher manipulation lines to switch tasks.
## Usage

### Run the agent

```bash
humanoid-llm-agent
```

The agent will start the MCP server as a subprocess, initialise the simulation, and autonomously issue control commands until the task succeeds. Action sequences are saved automatically to `log/`.

### Replay a logged run

Edit the `np.load(...)` path in the script to point to the desired log file under `log/`.

```bash
python scripts/box_pushing_playback.py
python scripts/pusher_manip_playback.py
```

### Run an MCP server standalone (for debugging or use with custom agents)

```bash
pusher-manip-mcp
# or
box-pushing-mcp
```

## Logs & Videos
- Action sequences are saved automatically to `log/<date>/<time>.npy` after each run.
- Demo logs are in `log/demo/`.
- Example videos are in `video/`.

## MCP Tools
MCP servers host the following tools for the LLM agent to interact with the simulations:

| Tool | Description |
|---|---|
| `start_simulation()` | Initialise the environment and return the first observation |
| `get_observation()` | Return current positions of pusher and goals (and box, for box-pushing) |
| `control_pusher(vx, vy, vz, step_size)` | Apply velocity for N steps; returns updated observation |
| `get_simulation_description()` | Return task description and success criteria |
| `close_simulation()` | Tear down the environment |
| `get_visual()` | For Box-pushing tasks only, return a base64-encoded JPEG render of the current frame |
