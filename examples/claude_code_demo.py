"""
The simplest demo, demonstrating how to integrate the pruner in the Claude Agent SDK
This demo demonstrates:
1. How to create a tool with a pruner (in bash, cat, grep)
2. How to create an MCP server and configure ClaudeAgentOptions
3. How to run queries using ClaudeSDKClient

Preparation before use:
1. Set the environment variable ANTHROPIC_API_KEY (required)
2. Optional: Set ANTHROPIC_BASE_URL and ANTHROPIC_MODEL
3. Ensure that the pruner service is running (default address: http://localhost:8000/prune)
The address can be customized through the PRUNER_URL environment variable
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    AssistantMessage,
    TextBlock,
)
from .tool_utils import bash as bash_tool, cat as cat_tool

load_dotenv()

assert os.getenv("ANTHROPIC_API_KEY"), "Set the ANTHROPIC_API_KEY environment variable"

pruner_url = os.getenv("PRUNER_URL", "http://localhost:8000/prune")
print(f"Pruner service address: {pruner_url}")
print("Note: If the pruner service is not running, the tools will not work properly\n")


def create_pruned_tools():
    """
    Create a set of tools with a pruner

    These tools will automatically prune the output content based on the context_focus_question parameter,
    only keeping the parts related to the query.
    """

    @tool(
        "bash",
        "Execute bash commands, supporting automatic pruning of output content based on queries."
        "ARGS: "
        "- command (str): The command to execute"
        "- context_focus_question (str | None): The full question used to focus the context."
        "  If provided, only the output content related to this question will be returned. If left empty, the complete output will be returned.",
        {"command": str, "context_focus_question": str | None},
    )
    async def bash(args) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": await bash_tool(
                        args["command"], args.get("context_focus_question", None)
                    ),
                }
            ]
        }

    @tool(
        "cat",
        "Read file contents, supporting automatic pruning of content based on queries."
        "ARGS: "
        "- file_path (str): The path to the file to read"
        "- context_focus_question (str | None): The full question used to focus the context."
        "  If provided, only the code parts related to this question will be returned. If left empty, the complete file content will be returned.",
        {"file_path": str, "context_focus_question": str | None},
    )
    async def cat(args) -> dict:
        return {
            "content": [
                {
                    "type": "text",
                    "text": await cat_tool(
                        args["file_path"], args.get("context_focus_question", None)
                    ),
                }
            ]
        }

    return {"bash": bash, "cat": cat}


async def main():
    """Demonstrate how to use tools with a pruner"""

    # 1. Create tools with a pruner
    print("Creating tools with a pruner...")
    tools_dict = create_pruned_tools()
    tools_list = list(tools_dict.values())

    # 2. Create MCP server
    print("Creating MCP server...")
    server = create_sdk_mcp_server(
        name="pruned_tools", version="1.0.0", tools=tools_list
    )

    # 3. Configure MCP tool prefix and allowed tools
    mcp_prefix = "mcp__pruned_tools__"
    allowed_tools = [f"{mcp_prefix}{t.name}" for t in tools_list]

    # Disable default Claude tools (use our pruned version)
    disallowed_tools = [
        "Bash",
    ]

    # 4. Create ClaudeAgentOptions
    print("Creating ClaudeAgentOptions...")
    options = ClaudeAgentOptions(
        mcp_servers={"pruned_tools": server},
        allowed_tools=allowed_tools,
        disallowed_tools=disallowed_tools,
        cwd=Path("."),  # Set working directory
        model=os.getenv(
            "ANTHROPIC_MODEL", "claude-haiku-4-5"
        ),  # Optional: Specify model
        env={
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL"),
        },
    )

    print("\nInteracting with Claude Agent...")
    print("=" * 60)

    user_query = "List the Python files in the current directory and read the first few lines of claude_code.py"

    async with ClaudeSDKClient(options=options) as client:
        # Send query
        await client.query(user_query)

        # Receive and print response
        print(f"User query: {user_query}\n")
        print("Agent response:")
        print("-" * 60)

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)

        print("-" * 60)
        print("\nDone!")

    print("\n" + "=" * 60)
    print("Demo description:")
    print(
        "1. The tools will automatically prune the output based on the context_focus_question parameter"
    )
    print(
        "2. If the agent provides the context_focus_question parameter when calling tools, only the code/output related to the question will be returned, thus saving tokens"
    )
    print(
        "3. If the agent does not provide the context_focus_question parameter (set to None), the complete output content will be returned"
    )


if __name__ == "__main__":
    asyncio.run(main())
