# 🍽️ Personal Chef Agent

> An AI-powered personal chef built with **LangChain + LangGraph + Tavily + Groq**.  
> Tell it what leftovers you have — it searches the web for real recipes and holds a full multi-turn conversation, remembering everything you said.

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)](https://www.python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.3%2B-informational)](https://python.langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-informational)](https://langchain-ai.github.io/langgraph)
[![Groq](https://img.shields.io/badge/Groq-free%20tier-orange)](https://console.groq.com)
[![Tavily](https://img.shields.io/badge/Tavily-search%20API-green)](https://tavily.com)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## What It Does

You type something like:

> *"I have leftover chicken and rice"*

The agent:
1. **Searches the web** via Tavily for real, up-to-date recipes matching your ingredients
2. **Returns 2–3 recipe options** with ingredient tables and quick-step instructions
3. **Remembers the full conversation** — ask for full instructions, substitutions, or variations as follow-ups
4. Keeps going back and forth — it knows what you asked before

### Sample conversation

```
You:   I have some leftover chicken and rice. What can I make?

Chef:  Here are a few tasty ways to turn your leftovers into a new meal:

       1️⃣  Chicken Fried Rice (15 min, one pan)
           Ingredients: olive oil, sesame oil, frozen peas, garlic, eggs,
           soy sauce... [full table]

       2️⃣  One-Pot Chicken & Rice Casserole (35 min, baked)
           ...

       3️⃣  Asian-Inspired Stovetop Chicken & Rice (20 min)
           ...

       Which one would you like full instructions for?

---

You:   Give me the full recipe for the first one

Chef:  Here are the full step-by-step instructions for Chicken Fried Rice:
       Prep: 5 min | Cook: 10 min | Serves: 4
       ...  [complete recipe]

---

You:   What ingredients did I say I had?

Chef:  You told me you had leftover chicken and rice. Based on that I
       suggested three recipes: Chicken Fried Rice, Casserole, and...
```

The last message proves the agent's memory — it accurately recalls context from two turns ago without being re-told.

---

## Architecture

```
User Input  ─────────────────────────────────────────────────────────────┐
                                                                          │
                                create_agent()                            │
                    ┌───────────────────────────────────┐                 │
                    │  model:   groq:openai/gpt-oss-120b │◄────────────────┘
                    │  tools:   [web_search]             │
                    │  prompt:  "You are a chef..."      │
                    │  memory:  InMemorySaver             │
                    └──────────────┬────────────────────┘
                                   │
                        ┌──────────▼──────────┐
                        │   ReAct Agent Loop  │
                        └──────────┬──────────┘
                                   │
               ┌───────────────────▼──────────────────────┐
               │  1. Reason  → decide to call web_search   │
               │  2. Act     → web_search(query) fires     │
               │  3. Observe → ToolMessage with results    │
               │  4. Respond → final AIMessage to user     │
               └───────────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  InMemorySaver checkpoints  │
                    │  state after every turn     │
                    │  (keyed by thread_id)       │
                    └─────────────────────────────┘
```

### Message flow inside a single turn

| Step | Message type | What it contains |
|------|-------------|-----------------|
| 1 | `HumanMessage` | User's ingredient list / question |
| 2 | `AIMessage` | Empty `content`; `tool_calls` field with `web_search` query |
| 3 | `ToolMessage` | Tavily's live search results (JSON) |
| 4 | `AIMessage` | Final synthesised answer to the user |

---

## Key Concepts Implemented

| Concept | Where | Why it matters |
|---------|-------|----------------|
| `create_agent()` | `agent.py` | Core LangGraph agent — wraps model, tools, memory in one call |
| `@tool` decorator | `agent.py` | Turns `web_search()` into an agent-callable action with a name and description |
| **ReAct loop** | Agent internals | Reason → Act (tool) → Observe → Respond — the industry standard for tool-using agents |
| **Tavily Search** | `web_search()` tool | Real-time web results — fixes the model's knowledge-cutoff problem |
| `InMemorySaver` | `agent.py` | LangGraph checkpointer: snapshots the full agent state after each turn |
| **Thread ID** | `config` dict | Groups all invocations into one continuous conversation |
| **System prompt** | `SYSTEM_PROMPT` | Sets persona, rules, and output format before any user message |
| **Structured output** | Response tables | Agent formats ingredient lists as markdown tables |
| `HumanMessage` | `agent.py`, `demo.ipynb` | Typed message wrapper for user input |
| `ChefState(AgentState)` | `agent.py` | Custom LangGraph state — adds `user_ingredients` field tracked per thread |
| `remember_ingredients` | `agent.py` | Tool that writes user-stated ingredients into state via `Command.update` |
| `recall_ingredients` | `agent.py` | Tool that reads `user_ingredients` from state via `ToolRuntime` |
| `Command(update={...})` | `remember_ingredients` | LangGraph primitive for writing to state from inside a tool |
| `ToolRuntime` | `recall_ingredients` | Gives a tool read access to the live LangGraph state dict |

---

## Project Structure

```
personal-chef-agent/
├── agent.py          # Main agent — run directly as a CLI chat loop
├── demo.ipynb        # Step-by-step notebook with real outputs & explanations
├── requirements.txt  # Python dependencies
├── .env.example      # Template for required API keys
├── .gitignore        # Keeps .env and __pycache__ out of git
└── README.md         # This file
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/marehman-exe/personal-chef-agent.git
cd personal-chef-agent
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Set up API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env  # Windows
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

| Key | Cost | Where to get it |
|-----|------|-----------------|
| `GROQ_API_KEY` | Free | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | Free tier (1 000 searches/month) | [tavily.com](https://tavily.com) |

### 4. Run

**CLI (interactive chat loop):**

```bash
python agent.py
```

```
Personal Chef Agent  |  type 'quit' to exit
--------------------------------------------------

You: I have eggs, spinach, and feta cheese
Chef: Here are some great ideas...

You: Give me the full recipe for the first one
Chef: [remembers the previous message and gives full recipe]
```

**Jupyter notebook (step-by-step with real outputs):**

```bash
jupyter lab
# Open demo.ipynb
```

---

## How Memory Works

### 1. Conversation memory (messages)

`InMemorySaver` (a LangGraph checkpointer) snapshots the full agent **state** after each `invoke()` call. On the next call with the same `thread_id`, the state is restored — so the agent has the full message history in its context window.

```python
# Thread ID is the key — everything with the same ID shares memory
config = {"configurable": {"thread_id": "chef-session-1"}}

# Turn 1 — agent learns you have chicken and rice
agent.invoke({"messages": [HumanMessage("I have chicken and rice")]}, config)

# Turn 2 — agent REMEMBERS turn 1 without being told again
agent.invoke({"messages": [HumanMessage("What was that first recipe?")]}, config)
```

```
Turn 1                    Turn 2                    Turn 3
┌──────────────┐          ┌────────────────────┐    ┌──────────────────────┐
│ HumanMessage │          │ HumanMessage ×2    │    │ HumanMessage ×3      │
│ AIMessage    │ ─save──► │ AIMessage    ×2    │ ─► │ AIMessage    ×3      │
└──────────────┘          │ ToolMessage  ×1    │    │ ToolMessage  ×2      │
  InMemorySaver           └────────────────────┘    └──────────────────────┘
  snapshots state           state grows each turn      full history replayed
```

### 2. Ingredient memory (custom state field)

A `ChefState` class extends `AgentState` with a dedicated `user_ingredients` field. This field holds **only what the user explicitly said** — it is never populated from recipe suggestions.

```python
class ChefState(AgentState):
    user_ingredients: str  # e.g. "leftover chicken and rice"
```

When the user mentions their ingredients, the agent calls `remember_ingredients`, which uses LangGraph's `Command` to write to that field:

```python
@tool
def remember_ingredients(ingredients: str, runtime: ToolRuntime) -> Command:
    """Save the ingredients the user said they have to the conversation state."""
    return Command(update={
        "user_ingredients": ingredients,   # ← written into ChefState
        "messages": [ToolMessage("Saved your ingredients: ...", ...)],
    })
```

When the user asks *"what did I say I had?"*, the agent calls `recall_ingredients`, which reads directly from the state — not from the conversation text:

```python
@tool
def recall_ingredients(runtime: ToolRuntime) -> str:
    """Read the ingredients the user said they have from the conversation state."""
    return runtime.state.get("user_ingredients", "")
    # → "leftover chicken and rice"  (never "eggs" or "soy sauce")
```

**Why this prevents the confusion:** without this field, the model reads all messages and may accidentally include recipe-suggested ingredients (eggs, soy sauce, garlic…) in its answer. With the dedicated field, the answer is always sourced from the exact string the user typed.

> **Memory scope:** Both mechanisms use `InMemorySaver`, which stores state in RAM. Memory lasts for the duration of the current Python process and is scoped to the `thread_id`. Restarting the process clears it. For persistence across restarts, replace `InMemorySaver` with a database-backed checkpointer such as `langgraph-checkpoint-sqlite`.

---

## What I Learned Building This

This project was built as the capstone of **Module 1: Create an Agent** from [LangChain Academy's Introduction to LangChain (Python)](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python).

### Concepts applied end-to-end

**1. Model initialisation & agent construction**  
LangChain's `create_agent()` is provider-agnostic. Swapping Groq for OpenAI or Anthropic is a one-line change — just replace the model string.

**2. The `@tool` decorator**  
Any Python function becomes an agent-callable tool by adding `@tool`. The function name and docstring become the tool's identity — the model reads the docstring to decide *when* to call it.

**3. The ReAct pattern**  
Agents don't just answer — they *reason, act, then observe*. When I first inspected the raw message list, the empty `AIMessage` content with a `tool_calls` field was the "aha" moment — the agent is making a decision, not writing a response.

**4. Short-term memory with checkpointers**  
Stateless by default. Adding `InMemorySaver` + a `thread_id` is all it takes to give an agent conversational memory. The state is persisted as a snapshot — each new turn replays the history.

**5. Prompt engineering**  
The system prompt is the first lever for controlling agent behaviour. A well-structured prompt — with a clear role, numbered tasks, and output expectations — significantly reduces hallucination and keeps responses focused.

**6. Real-time tool integration**  
Models have a knowledge cutoff. Tavily solves this cleanly — it returns LLM-friendly, structured search results. The agent called it automatically, without any explicit instruction to do so, because the docstring described it accurately.

---

## Tech Stack

| Library | Version | Role |
|---------|---------|------|
| `langchain` | ≥ 1.3 | Agent framework, tool decorator, message types |
| `langgraph` | ≥ 1.0 | Agent state machine, `create_agent()`, `InMemorySaver` |
| `langchain-groq` | ≥ 1.1 | Groq provider integration |
| `tavily` | ≥ 0.7 | Real-time web search tool |
| `python-dotenv` | ≥ 1.2 | Load API keys from `.env` |

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as part of the [Introduction to LangChain — Python](https://academy.langchain.com/courses/foundation-introduction-to-langchain-python) course by [LangChain Academy](https://academy.langchain.com).*
