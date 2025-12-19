import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool
import os
from dotenv import load_dotenv
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.tools.http import HttpTool

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


schema = {
    "type": "object",
    "properties": {
        "fact": {
            "type": "string",
            "description": "A random cat fact"
        },
        "length": {
            "type": "integer",
            "description": "The length of the cat fact"
        },
    },
    "required": ["fact", "length"],
}

# URL: https://catfact.ninja/fact

http_tool = HttpTool(
    name='cat_facts_api',
    description='Fetch random cat facts from the Cat Facts API',
    scheme='https',
    host='catfact.ninja',
    port=443,
    path='/fact',
    method='GET',
    return_type='json',
    json_schema=schema,
)

async def main():
    agent = AssistantAgent(
        name="Cat_Facts_Agent",
        model_client=model_client,
        system_message="You are a helpful assistant who can fetch random cat facts using the cat_facts_api tool.",
        tools=[http_tool],
        reflect_on_tool_use=True,
    )

    response = await agent.on_messages(
        [TextMessage(content="Fetch a random cat fact.")],
        CancellationToken()
    )

    print(response.chat_message)