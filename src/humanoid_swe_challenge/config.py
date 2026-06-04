import os
import numpy as np

# Local_LM_Studio
LLM_URL = os.environ.get("LLM_URL", "http://localhost:1234/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.6-35b-a3b:2")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
LLM_TOKEN_LIMIT = -1
LOG_PATH="log"
NP_RANDOM_SEED=42

MCP_SERVER_COMMAND = "pusher-manip-mcp"
USER_PROMPT = "run the pusher manipulation simulation, check the simulation description. Complete the manipulation task, move the pusher to where the blue goal is. After that moved to the red goal. After that moved to the green goal."

# MCP_SERVER_COMMAND = "box-pushing-mcp"
# USER_PROMPT = "run the 3D box-pushing simulation, check the simulation description. Then push the purple box into the blue goal. Use the image render of the environment to better understand the task. The pusher and the box can break contact, and the box will not move if the pusher is not in contact with the box. If contact is lost or you want to push the box to another direction, it is better to use the updated image render to try to re-establish contact an to confirm how the pusher is making contact with the box. Make sure to check the visual occasionally to update your understanding. Multiple actions can be applied, and the task is defiantly achievable. In the rendered image, the pusher is the sphere coloured orange"

