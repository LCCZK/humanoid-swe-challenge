import os

# MCP_Server
MCP_HOST="localhost"
MCP_PORT=8000

# Local_LM_Studio
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = "qwen/qwen3.6-35b-a3b"
LLM_API_KEY = "not-needed"