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

from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Dict, Any
from tavily import TavilyClient

# ── Custom state ──────────────────────────────────────────────────────────────
# Extends the default AgentState with one extra field.
# This field holds only ingredients the USER explicitly mentioned —
# never ingredients suggested by the assistant in a recipe.

class ChefState(AgentState):
    user_ingredients: str  # e.g. "leftover chicken and rice"

# ── Tools ─────────────────────────────────────────────────────────────────────

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for up-to-date recipes and cooking information."""
    return tavily_client.search(query)

@tool
def remember_ingredients(ingredients: str, runtime: ToolRuntime) -> Command:
    """Save the ingredients the user said they have to the conversation state.
    Call this whenever the user tells you what ingredients they have.
    Only save what the user explicitly stated — not recipe suggestions.
    """
    # Command.update writes directly into the LangGraph state.
    # The ToolMessage is required so the agent sees a result for the tool call.
    return Command(update={
        "user_ingredients": ingredients,
        "messages": [ToolMessage(
            f"Saved your ingredients: {ingredients}",
            tool_call_id=runtime.tool_call_id,
        )],
    })

@tool
def recall_ingredients(runtime: ToolRuntime) -> str:
    """Read the ingredients the user said they have from the conversation state.
    Call this when the user asks what ingredients they mentioned.
    """
    # runtime.state is the live LangGraph state dict for this thread.
    ingredients = runtime.state.get("user_ingredients", "")
    if ingredients:
        return f"The user said they have: {ingredients}"
    return "The user has not mentioned any ingredients yet in this conversation."

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a personal chef assistant. The user will tell you what leftover
ingredients they have at home.

Your job:
1. When the user tells you their ingredients, FIRST call remember_ingredients
   to save exactly what they said — use their words, nothing added.
2. Use the web_search tool to find real recipes that can be made with
   those ingredients.
3. Return 2–3 clear recipe suggestions, each with a short ingredient list
   and quick cooking steps.
4. If the user asks for more detail on any recipe, provide full
   step-by-step instructions.
5. If the user asks what ingredients they mentioned, call recall_ingredients
   to retrieve what was saved — do NOT guess from recipe suggestions.
6. Answer any follow-up cooking questions using the conversation history.

Keep your responses practical, friendly, and concise.
"""

# ── Agent ─────────────────────────────────────────────────────────────────────

# InMemorySaver checkpointer: persists the conversation state in RAM so the
# agent can remember previous messages within a session (short-term memory).
checkpointer = InMemorySaver()

agent = create_agent(
    model="groq:openai/gpt-oss-120b",
    tools=[web_search, remember_ingredients, recall_ingredients],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    # state_schema adds user_ingredients to the state tracked per thread_id.
    # The checkpointer snapshots this field along with messages after every turn.
    state_schema=ChefState,
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
