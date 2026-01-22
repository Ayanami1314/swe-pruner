import asyncio
import os
from dotenv import load_dotenv
from openhands.sdk import LLM, Conversation, Agent
from openhands.sdk.tool import Tool, register_tool
from .tool_utils_openhands import pruner_bash, origin_bash

load_dotenv()
experiment = os.getenv("EXPERIMENT_TYPE", "pruner")
if experiment == "baseline":
    register_tool("Bash", origin_bash)
elif experiment == "pruner":
    register_tool("Bash", pruner_bash)


async def main():
    tools = [Tool(name="Bash")]
    LLM_CONFIG = {
        "model": "gpt-5-nano",
        "api_key": os.getenv("OPENAI_API_KEY"),
    }
    llm = LLM(**LLM_CONFIG)
    agent = Agent(llm=llm, tools=tools)
    conversation = Conversation(
        agent=agent,
        workspace=".",
        max_iteration_per_run=50,
    )

    user_query = input("Enter your query: ")
    conversation.send_message(user_query)
    async for message in conversation.receive_response():
        print(message)

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
