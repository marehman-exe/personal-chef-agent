"""
Personal Chef Agent
-------------------
An AI-powered personal chef that takes your leftover ingredients,
searches the web for matching recipes, and holds a full multi-turn
conversation — remembering context across every message.

Built with: LangChain · LangGraph · Tavily · Groq
Author: Muhammad Ali Rehman (marehman-exe)
Course: Introduction to LangChain – Python (LangChain Academy, Module 1)
"""

from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from typing import Dict, Any
from tavily import TavilyClient

# ── Tool ──────────────────────────────────────────────────────────────────────

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for up-to-date recipes and cooking information."""
    return tavily_client.search(query)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a personal chef assistant. The user will tell you what leftover
ingredients they have at home.

Your job:
1. Use the web_search tool to find real recipes that can be made with
   those ingredients.
2. Return 2–3 clear recipe suggestions, each with a short ingredient list
   and quick cooking steps.
3. If the user asks for more detail on any recipe, provide full
   step-by-step instructions.
4. Answer any follow-up cooking questions using the conversation history.

Keep your responses practical, friendly, and concise.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

# InMemorySaver checkpointer: persists the conversation state in RAM so the
# agent can remember previous messages within a session (short-term memory).
checkpointer = InMemorySaver()

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[web_search],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
)

# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "chef-session-1"}}
    print("Personal Chef Agent  |  type 'quit' to exit\n")
    print("-" * 50)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not user_input:
            continue

        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )
        print(f"\nChef: {response['messages'][-1].content}")
