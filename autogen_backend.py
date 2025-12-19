"""
litrev_backend.py
=================
Core logic for the multi-agent literature-review assistant built with the
**AutoGen** AgentChat stack (>v0.4). It exposes a single public coroutine 
'run_litrev' that drives a two-agent team.

* **search_agent** - Crafts an arXiv query and fetches papers via the provided
'arxiv_search' tool.
* **summarize** - Writes a short Markdown literature review from the selected papers.

The module is deliberately self-contained so it can be reused in CLI apps,
Streamlit, FastAPI, Gradio, etc.
"""

from __future__ import annotations
import asyncio
from typing import AsyncGenerator, Dict, List
import arxiv
from autogen_core.tools import FunctionTool
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------
# 1. Tool Definition
# ---------------------------------------------------------------------

def arxiv_search(query: str, max_results: int = 5) -> List[Dict]:
    """Return a compact list of arXiv papers matching the query.
    Each element contains: ``title``, ``authors``, ``published``, ``summary`` and ``pdf_url``.
    The helper is wrapped as an AutoGen *FunctionTool* below so it can invoked by agents 
    through the normal tool-use mechanism.
    """

    client=arxiv.Client()
    search=arxiv.Search(
        query=query, 
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers: List[Dict] = []
    for result in client.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary,
            "pdf_url": result.pdf_url,
        })
    
arxiv_tool = FunctionTool(
    arxiv_search,
    description=(
        "Searches arXiv and returns upto *max_results* papers each containing "
        "title, authors, published date, summary and pdf_url."
    )
)

# ---------------------------------------------------------------------
# 2. Agent & Team Factory
# ---------------------------------------------------------------------

def build_team(model: str = "llama-3.1-8b-instant") -> RoundRobinGroupChat:
    """Create and return a two-agent *RoundRobinGroupChat* team."""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    llm_client = OpenAIChatCompletionClient(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    model=model,
    model_info={
        "family": "llama",
        "vision": False,
        "function_calling": True,
        "json_output": True,
        },
    )

    search_agent = AssistantAgent(
        name="Search_Agent",
        description="Crafts arXiv queries and retrieves candidate papers.",
        model_client=llm_client,
        system_message=(
            "Given a user topic, think of the best arXiv query and call the "
            "provided tool. Always fetch five-times the papers requested so "
            "that you can down-select the most relevant ones. When the tool "
            "returns, choose exactly the number of papers requested and pass "
            "them as concise JSON to the summarizer."
        ),
        tools=[arxiv_tool],
        reflect_on_tool_use=True,
    )

    summarizer = AssistantAgent(
        name="summarizer",
        description="Produces a short Markdown review from provided papers.",
        system_message=(
            "You are an expert researcher. When you receive the JSON list of "
            "papers, write a literature-review style report in Markdown:\n" \
            "1. Start with a 2-3 sentence introduction of the topic.\n" \
            "2. Then include one bullet per paper with: title (as Markdown "
            "link), authors, the specific problem tackled, and its key "
            "contribution.\n" \
            "3. Close with a single-sentence takeaway."
        ),
        model_client=llm_client,
    )

    return RoundRobinGroupChat(
        participants=[search_agent, summarizer],
        max_turns=2,
    )


# ---------------------------------------------------------------------
# 3. Orchestrator
# ---------------------------------------------------------------------

async def run_litrev(
    topic: str, 
    num_papers: int = 5, 
    model: str = "llama-3.1-8b-instant"
) -> AsyncGenerator[str, None]:
    """Yield strings representing the conversation in real-time."""

    team = build_team(model=model)
    task_prompt = (
        f"Conduct a literature review on **{topic}** and return exactly {num_papers} papers."
    )

    async for msg in team.run_stream(task=task_prompt):
        if isinstance(msg, TextMessage):
            yield f"{msg} \n\n\n {msg.source}: {msg.content}"


# ---------------------------------------------------------------------
# 4. CLI Testing
# ---------------------------------------------------------------------

if __name__ == "__main__":
    async def _demo() -> None:
        async for line in run_litrev("Graph neural networks for chemistry", num_papers=5):
            print(line)

    asyncio.run(_demo())