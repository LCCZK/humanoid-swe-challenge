import asyncio
import json
from mcp import ClientSession
# from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from humanoid_swe_challenge.config import MCP_HOST, MCP_PORT

from humanoid_swe_challenge.llm_agent.utils import list_tools, call_tool, call_llm
_SERVER_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

# _SERVER_PARAMS = StdioServerParameters(command="humanoid-gym-mcp")


async def _run_agent(prompt: str) -> dict:
    messages: list = [{"role": "user", "content": prompt}]

    async with streamable_http_client(_SERVER_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_tools(session)

            while True:
                trimmed = messages[:1] + messages[-50:] if len(messages) > 51 else messages
                response = call_llm(trimmed, tools)
                msg = response.choices[0].message
                if msg.reasoning_content:
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



def run_agent(prompt: str):
    print(asyncio.run(_run_agent(prompt))) 



def main():
    # run_agent("run the 3D box-pushing simulation, check the simulation description. Then push the box into the blue goal. The pusher and the box can break contact, and the box will not move if the pusher is not in contact with the box. Multiple actions can be applied, and the task is defintly achieveable. You should pay close attention to the position of the pusher and the xyz coordinate and yaw of the box, think about where the pusher is making contact with the box. This is a complex task so think carefully before commit to any significant actions.")
    # run_agent("run the pusher manipulation simulation, check the simulation description. " \
    # "Complete the manipulation task, move the pusher to where the blue goal is.")
    run_agent("run the pusher manipulation simulation, check the simulation description. " \
    "Complete the manipulation task, move the pusher to where the blue goal is. After that moved to the red goal. After that moved to the green goal.")


if __name__ == "__main__":
    main()
