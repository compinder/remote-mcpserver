from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio

load_dotenv()


class ToolDefinition(TypedDict):
    type: str
    name: str
    description: str
    parameters: dict


class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()

        # OpenAI client
        self.openai = OpenAI()

        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)

            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            read, write = stdio_transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()
            self.sessions.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools

            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session

                # Convert MCP tool schema to OpenAI function tool schema
                self.available_tools.append({
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {
                        "type": "object",
                        "properties": {}
                    }
                })

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)

        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    def _serialize_mcp_result(self, result) -> str:
        """
        Convert MCP tool result content into a string suitable for OpenAI function_call_output.

        MCP result.content is often a list of content blocks.
        OpenAI function_call_output expects output to be a string.
        """
        try:
            serialized_blocks = []

            for item in result.content:
                # Common MCP text content block
                if hasattr(item, "text"):
                    serialized_blocks.append(item.text)
                else:
                    # Fallback for non-text MCP content
                    serialized_blocks.append(str(item))

            return "\n".join(serialized_blocks)

        except Exception:
            return str(result)

    async def process_query(self, query: str):
        """
        Process a user query using OpenAI Responses API and MCP tools.
        """

        response = self.openai.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "user",
                    "content": query
                }
            ],
            tools=self.available_tools,
            max_output_tokens=2024,
        )

        while True:
            # Print any assistant text
            if response.output_text:
                print(response.output_text)

            function_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            # No tool calls means the model is done
            if not function_calls:
                break

            tool_outputs = []

            for function_call in function_calls:
                tool_name = function_call.name
                tool_args = json.loads(function_call.arguments or "{}")

                print(f"Calling tool {tool_name} with args {tool_args}")

                session = self.tool_to_session[tool_name]

                result = await session.call_tool(
                    tool_name,
                    arguments=tool_args
                )

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": function_call.call_id,
                    "output": self._serialize_mcp_result(result)
                })

            # Send tool outputs back to OpenAI
            response = self.openai.responses.create(
                model="gpt-4.1",
                previous_response_id=response.id,
                input=tool_outputs,
                tools=self.available_tools,
                max_output_tokens=2024,
            )

    async def chat_loop(self):
        """Run an interactive chat loop."""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                await self.process_query(query)
                print("\n")

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()

    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
