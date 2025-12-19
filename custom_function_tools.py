import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

model_client = OpenAIChatCompletionClient(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    model_info={
        "family": "llama",
        "vision": False,
        "function_calling": True,
        "json_output": True,
    },
)

# Define a custom function to reverse a string
def reverse_string(text: str) -> str:
    """Reverse the given string.
    input: str
    output: str
    Reversed String is returned
    """
    return text[::-1]

# Register the custom function as a tool
reverse_tool = FunctionTool(reverse_string, description="A tool to reverse a string")

agent = AssistantAgent(
    name="Reverse_Agent",
    model_client=model_client,
    system_message="You are a helpful assistant who can reverse a string using the reverse_string tool.",
    tools=[reverse_tool],
    reflect_on_tool_use=True,  # Provide something extra in the response along with the function return value.
)

task = "Reverse the string 'Hello World'"

async def main():
    result = await agent.run(task=task)
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
