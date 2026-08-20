"""
The AI agent: a LangGraph ReAct-style agent that decides whether a tool is
needed, calls it, and folds the result back into a final answer.

This is deliberately separate from app/rag/chain.py (the plain grounded RAG
path). The RAG chain always retrieves and answers from documents; this agent
first *decides* what kind of request it's looking at — a legal question, a
calculation, a date question, or something needing no tool at all — which is
the actual difference between "an LLM call" and "an agent" per this
assignment's brief.

Uses langgraph.prebuilt.create_react_agent: the model is bound to the tool
schemas, and LangGraph handles the loop of call model -> run any requested
tools -> call model again with results -> repeat until the model returns a
plain answer with no further tool calls.

--- Conversation memory -----------------------------------------------------
create_react_agent's compiled graph state has a `messages` field that uses
LangGraph's `add_messages` reducer: every graph step *appends* to that list
instead of overwriting it. That's true with or without a checkpointer — the
checkpointer is what makes the list persist *between separate calls to
run_agent()* instead of starting empty every time.

Concretely: get_agent() now builds the graph with a SqliteSaver checkpointer
(see app/agent/checkpoint.py). Every run_agent() call passes a `thread_id` in
its config. Before the graph runs, LangGraph loads that thread's prior
`messages` list from SQLite and seeds the state with it; the new user message
is appended on top. As the graph moves through the agent node and any tool
nodes, the checkpointer writes the growing message list back to disk after
each step — so even mid-run (agent -> tool -> agent), history is never held
only in a Python variable. When the run finishes, the full updated
conversation is already persisted under that thread_id, ready for the next
turn or the next process restart.
"""

import logging
from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.agent.checkpoint import clear_thread, get_checkpointer
from app.agent.tools import TOOLS
from app.config import settings

logger = logging.getLogger("agent.graph")


@lru_cache(maxsize=1)
def get_agent():
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )
    checkpointer = get_checkpointer()
    return create_react_agent(llm, TOOLS, checkpointer=checkpointer)


def run_agent(question: str, thread_id: str) -> dict:
    """Run the agent on a question within a conversation thread.

    `thread_id` is the whole memory mechanism from the caller's point of
    view: pass the same thread_id again and the agent sees every prior
    turn in that conversation (loaded by the checkpointer before this
    invoke() call); pass a different thread_id and it's a brand new,
    fully isolated conversation, even if it's running in the same process
    a second later.

    Returns the answer plus a log of which tool(s) were used along the
    way (the assignment's 'log which tool was selected' bonus, surfaced
    back to the caller instead of only to server logs) and the thread_id
    the caller should reuse for the next turn.
    """
    if not thread_id:
        raise ValueError("run_agent() requires a non-empty thread_id")

    agent = get_agent()
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("thread_id=%s user_message=%r", thread_id, question)

    # Peek at what's already stored for this thread purely for logging /
    # to know where in result["messages"] the *new* turn starts, so
    # tools_used below only reflects this turn, not the whole history.
    prior_state = agent.get_state(config)
    prior_messages = prior_state.values.get("messages", []) if prior_state.values else []
    logger.info("thread_id=%s loaded_prior_messages=%d", thread_id, len(prior_messages))

    # Only the NEW human message is passed in -- the checkpointer supplies
    # everything before it. This is the key difference from the old,
    # memory-less version, which passed the full one-off message list on
    # every call and never loaded or saved anything.
    result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)

    tools_used = []
    for msg in result["messages"][len(prior_messages):]:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for call in calls:
                tools_used.append(call["name"])
                logger.info("thread_id=%s tool_selected=%s", thread_id, call["name"])

    final_message = result["messages"][-1]
    logger.info(
        "thread_id=%s response_ready total_messages=%d",
        thread_id,
        len(result["messages"]),
    )

    return {
        "answer": final_message.content,
        "tools_used": tools_used,
        "thread_id": thread_id,
    }


def reset_conversation(thread_id: str) -> None:
    """Clear all persisted memory for exactly one conversation thread.

    Every other thread_id's history is untouched -- see
    app/agent/checkpoint.py:clear_thread for why that's safe.
    """
    if not thread_id:
        raise ValueError("reset_conversation() requires a non-empty thread_id")

    clear_thread(thread_id)
    logger.info("thread_id=%s conversation_reset", thread_id)