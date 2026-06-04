import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
# from mcp.client.streamable_http import streamable_http_client

# from humanoid_swe_challenge.config import MCP_HOST, MCP_PORT
from humanoid_swe_challenge.config import USER_PROMPT, MCP_SERVER_COMMAND
from humanoid_swe_challenge.llm_agent.utils import list_tools, call_tool, call_llm

# _SERVER_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"
_SERVER_PARAMS = StdioServerParameters(command=MCP_SERVER_COMMAND)
REDUCED_CONTEXT_MSG = [{"role": "assistant", "content": "context have been reduced"}]


async def _run_agent(prompt: str) -> dict:
    messages: list = [{"role": "user", "content": prompt}]

    # async with streamable_http_client(_SERVER_URL) as (read, write, _):
    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_tools(session)

            while True:
                trimmed = messages[:5] + REDUCED_CONTEXT_MSG + messages[-100:] if len(messages) > 106 else messages
                response = await call_llm(trimmed, tools)
                msg = response.choices[0].message

                if getattr(msg, "content", None):
                    print(msg.content)
                if getattr(msg, "reasoning_content", None):
                    print(msg.reasoning_content)
                # print(msg)

                if not msg.tool_calls:
                    # print("No tool calls")
                    return msg.content

                messages.append(msg.model_dump())

                for call in msg.tool_calls:
                    tool_name = call.function.name  # type: ignore
                    args = json.loads(call.function.arguments)  # type: ignore
                    # print(tool_name)
                    # print(args)
                    result = await call_tool(session, tool_name, args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    })


def main():
    print(asyncio.run(_run_agent(USER_PROMPT))) 


if __name__ == "__main__":
    main()
