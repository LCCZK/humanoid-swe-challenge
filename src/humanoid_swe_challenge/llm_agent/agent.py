import asyncio
import json
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

from humanoid_swe_challenge.config import USER_PROMPT, MCP_SERVER_COMMAND
from humanoid_swe_challenge.llm_agent.utils import list_tools, call_tool, call_llm


_SERVER_PARAMS = StdioServerParameters(command=MCP_SERVER_COMMAND)
# Injected when message history is trimmed to signal the LLM that earlier context was dropped
REDUCED_CONTEXT_MSG = [{"role": "assistant", "content": "context have been reduced"}]


async def _run_agent(prompt: str) -> dict:
    messages: list = [{"role": "user", "content": prompt}]

    async with stdio_client(_SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await list_tools(session)

            while True:
                # Keep first 5 messages for context, then the most recent 30 to stay within context limits
                trimmed = messages[:5] + REDUCED_CONTEXT_MSG + messages[-30:] if len(messages) > 36 else messages
                response = await call_llm(trimmed, tools)
                msg = response.choices[0].message

                if getattr(msg, "content", None):
                    print(msg.content)
                if getattr(msg, "reasoning_content", None):
                    print(msg.reasoning_content)

                # No tool calls means the LLM has finished the task
                if not msg.tool_calls:
                    return msg.content

                messages.append(msg.model_dump())

                for call in msg.tool_calls:
                    tool_name = call.function.name  # type: ignore
                    args = json.loads(call.function.arguments)  # type: ignore
                    result = await call_tool(session, tool_name, args)

                    # Append tool result so the LLM can observe the outcome
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    })


def main():
    print(asyncio.run(_run_agent(USER_PROMPT))) 


if __name__ == "__main__":
    main()
