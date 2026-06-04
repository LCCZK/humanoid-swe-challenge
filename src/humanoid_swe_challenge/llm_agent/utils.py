import json
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.types import TextContent

from humanoid_swe_challenge.config import LLM_API_KEY, LLM_URL, LLM_MODEL, LLM_TOKEN_LIMIT

client = AsyncOpenAI(base_url=LLM_URL, api_key=LLM_API_KEY)


async def list_tools(session: ClientSession) -> list[dict]:
    mcp_tools = await session.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools.tools
    ]


async def call_tool(session: ClientSession, name: str, args: dict) -> dict:
    result = await session.call_tool(name, args)

    if result.isError:
        error_text = result.content[0].text if result.content else "Unknown MCP error"
        raise Exception(f"MCP tool error: {error_text}")

    for content in result.content:
        if isinstance(content, TextContent):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                return {"result": content.text}

    return {"error": "No content in MCP response"}


async def call_llm(messages: list, tools: list):
    return await client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=LLM_TOKEN_LIMIT,
    )
