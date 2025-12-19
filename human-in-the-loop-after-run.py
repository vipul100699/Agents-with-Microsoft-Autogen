import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model_client = OpenAIChatCompletionClient(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    # model="meta-llama/llama-4-maverick-17b-128e-instruct",
    model="llama-3.1-8b-instant",
    model_info={
        "family": "llama",
        "vision": False,
        "function_calling": True,
        "json_output": True,
    },
)

writer_agent = AssistantAgent(
    name="Writer",
    description="You are a great writer",
    model_client=model_client,
    system_message="You are a really helpful writer who writes in less than 30 words."
)

reviewer_agent = AssistantAgent(
    name="Reviewer",
    description="You are a great reviewer",
    model_client=model_client,
    system_message="You are a really helpful reviewer who writes in less than 30 words."
)

editor_agent = AssistantAgent(
    name="Editor",
    description="You are a great editor",
    model_client=model_client,
    system_message="You are a really helpful editor who writes in less than 30 words."
)

team = RoundRobinGroupChat(
    participants=[writer_agent, reviewer_agent, editor_agent],
    max_turns=3
)

async def main():
    task = "Write a nice 4 line poem about oceans"
    while True:
        stream = team.run_stream(task=task)
        await Console(stream)

        feedback = input("Please provide your feedback (type 'Exit' to stop.)")
        if feedback.lower().strip() == "exit":
            break

        task = feedback

if __name__ == "__main__":
    asyncio.run(main())