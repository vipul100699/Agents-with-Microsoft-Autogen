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

assistant_agent = AssistantAgent(
    name="Assistant",
    description="You are a great assistant",
    model_client=model_client,
    system_message="You are a really helpful assistant who helps on the given task."
)

user_agent = UserProxyAgent(
    name="UserProxy",
    description="A proxy agent that represents a user",
    input_func=input
)

termination_condition = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat(
    participants=[assistant_agent, user_agent],
    termination_condition=termination_condition
)

stream = team.run_stream(task="Write a nice 4 line poem about India.")

async def main():
    await Console(stream)

if __name__ == "__main__":
    asyncio.run(main())