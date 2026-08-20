"""
The AI agent: a hand-built LangGraph StateGraph that decides whether a tool
is needed, calls it, folds the result back into State, and — critically —
loops back through the agent node so it can decide whether ANOTHER tool is
needed before producing a final answer.

This replaces the earlier langgraph.prebuilt.create_react_agent version.
That prebuilt helper gave looping tool-calling "for free," but it hid the
State schema, nodes, and routing logic inside the library. Building it
explicitly here is what actually demonstrates (and lets us customize):

  - LangGraph State           -> app/agent/state.py::AgentState
  - Nodes                     -> agent_node, tool_node (below)
  - Edges                     -> agent_node -> tool_node -> agent_node
  - Conditional routing       -> route_after_agent()
  - Tool result handling      -> tool_node appends ToolMessage + updates state
  - Error handling            -> tool_node never lets an exception propagate
  - Conversation history      -> MemorySaver checkpointer, keyed by thread_id
  - Execution trace           -> build_execution_trace()

This is deliberately separate from app/rag/chain.py (the plain grounded RAG
path). The RAG chain always retrieves and answers from documents; this agent
first *decides* what kind of request it's looking at — a legal question, a
calculation, a date question, both in sequence, or something needing no tool
at all.
"""

import logging
import uuid
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.agent.state import AgentState
from app.agent.tools import TOOLS

logger = logging.getLogger("agent.graph")
logging.basicConfig(level=logging.INFO)

# Map tool name -> callable, so tool_node can dispatch without an if/elif
# chain that has to be updated every time a tool is added.
_TOOLS_BY_NAME = {t.name: t for t in TOOLS}

_SYSTEM_PROMPT = (
    "You are a legal research assistant for Pakistani law. You have access "
    "to three tools: search_legal_documents (search the indexed legal "
    "documents), calculate (basic arithmetic), and get_current_date. "
    "Use a tool whenever the question needs one — including calling "
    "search_legal_documents first and then calculate if a question asks "
    "you to look something up AND compute something from it. If no tool is "
    "needed (e.g. the user is asking you to clarify or simplify your own "
    "previous answer), respond directly. Never fabricate legal information "
    "that didn't come from search_legal_documents."
)


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file — see .env.example."
        )
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )
    return llm.bind_tools(TOOLS)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def agent_node(state: AgentState) -> dict:
    """The reasoning step. Sends the full message history (system prompt +
    everything accumulated so far, including any prior ToolMessages) to the
    LLM and gets back either a plain text answer or one or more tool calls.

    This node runs every time control returns to the agent — first on the
    user's raw question, and again after every tool_node execution — which
    is what makes multi-step behavior possible: the model sees its own
    previous tool results before deciding what to do next.
    """
    llm = _get_llm()
    messages = state["messages"]

    # Prepend the system prompt only if it isn't already the first message
    # (it won't be, on turn 2+, since checkpointed history already has it —
    # avoid duplicating it every turn).
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=_SYSTEM_PROMPT), *messages]

    response = llm.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Executes every tool call requested by the last AI message.

    Error handling lives at two layers on purpose:
      1. Each tool in app/agent/tools.py already catches its own exceptions
         and returns a readable string (e.g. "calculate tool failed: ...").
      2. This node ALSO wraps the call in try/except, as a safety net for
         any exception a tool doesn't catch itself (e.g. a RuntimeError from
         answer_question() if GROQ_API_KEY is missing). Either way, the
         result is always a ToolMessage — the graph never crashes because a
         tool failed.
    """
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []

    tool_messages: list[ToolMessage] = []
    used: list[str] = []
    last_error: str | None = None

    for call in tool_calls:
        name = call["name"]
        args = call.get("args", {})
        used.append(name)

        tool_fn = _TOOLS_BY_NAME.get(name)
        if tool_fn is None:
            content = f"Unknown tool requested: {name!r}"
            logger.error(content)
            last_error = content
        else:
            try:
                content = tool_fn.invoke(args)
                # The tools themselves also catch errors and return a
                # string like "<tool> tool failed: ...". Surface that as
                # a state-level error too, so the final answer can be
                # honest about a degraded result.
                if isinstance(content, str) and "tool failed" in content:
                    last_error = content
            except Exception as exc:  # noqa: BLE001
                content = f"{name} tool failed unexpectedly: {exc}"
                logger.exception("tool_node: %s raised", name)
                last_error = content

        tool_messages.append(
            ToolMessage(content=str(content), tool_call_id=call["id"], name=name)
        )

    return {
        "messages": tool_messages,
        "tools_used": used,
        "error": last_error,
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------
def route_after_agent(state: AgentState) -> Literal["tools", "end"]:
    """Decide what happens after the agent node runs.

    If the model's latest message requested one or more tool calls, go
    execute them. Otherwise the model produced a final answer — end the
    turn. This is what allows a genuinely variable-length path (0, 1, 2+
    tool calls) instead of a hard-coded fixed sequence.
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_graph():
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": END},
    )
    # After tools run, ALWAYS loop back to the agent so it can read the
    # result and decide whether another step is needed. This is the edge
    # that makes multi-step workflows real instead of simulated.
    builder.add_edge("tools", "agent")

    # In-memory checkpointer: each thread_id gets its own persisted message
    # history, so a second call with the same thread_id continues the same
    # conversation instead of starting fresh. Swap for a persistent
    # checkpointer (e.g. SqliteSaver) if history needs to survive a process
    # restart — in-memory is enough for a single-server demo/deployment.
    checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Execution trace — a human-readable path through the graph for this turn
# ---------------------------------------------------------------------------
def build_execution_trace(messages_before: int, all_messages: list) -> list[str]:
    """Turn the messages added during this turn into a step-by-step trace
    like ["AGENT", "search_legal_documents", "AGENT", "calculate", "AGENT"].
    `messages_before` is how many messages existed before this turn started
    (relevant on turn 2+, where prior history is already in state).
    """
    trace: list[str] = []
    for msg in all_messages[messages_before:]:
        if isinstance(msg, AIMessage):
            trace.append("AGENT")
        elif isinstance(msg, ToolMessage):
            trace.append(msg.name or "TOOL")
    return trace


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------
def run_agent(question: str, thread_id: str | None = None) -> dict:
    """Run the agent on a question, threading conversation history through
    `thread_id`, and return the final answer plus a transparent log of what
    happened.

    `thread_id=None` (the default) generates a fresh id, giving a stateless,
    one-off turn — pass the SAME thread_id on a follow-up call to let the
    agent see prior turns via the checkpointer.

    Returns:
        {
            "answer": str,
            "tools_used": list[str],       # tool names invoked this turn
            "execution_trace": list[str],  # e.g. ["AGENT", "calculate", "AGENT"]
            "thread_id": str,               # echo back so the caller can continue the thread
            "error": str | None,           # last tool failure this turn, if any
        }
    """
    if thread_id is None:
        thread_id = str(uuid.uuid4())

    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    # How much history already exists for this thread (0 on a fresh thread),
    # so build_execution_trace() only reports THIS turn's steps.
    prior_state = graph.get_state(config)
    messages_before = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    try:
        result = graph.invoke(
            {
                "messages": [("user", question)],
                "tools_used": [],
                "error": None,
            },
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        # Graph-level safety net: if something outside a tool call fails
        # (e.g. the LLM call itself, missing API key), surface it as a
        # readable error instead of a raw traceback reaching the API layer.
        logger.exception("run_agent: graph invocation failed")
        raise RuntimeError(f"Agent execution failed: {exc}") from exc

    all_messages = result["messages"]
    final_message = all_messages[-1]
    trace = build_execution_trace(messages_before, all_messages)

    # Derive tools_used from the trace (i.e. from the actual ToolMessages in
    # this turn's slice of `messages`) rather than trusting state["tools_used"]
    # directly. `messages` uses LangGraph's add_messages reducer, which
    # correctly APPENDS across multiple tool_node executions in one turn —
    # but AgentState.tools_used has no reducer, so on a genuine two-tool turn
    # (tools -> agent -> tools -> agent) the second tool_node call would
    # silently overwrite the first's contribution instead of accumulating.
    # Reading it back out of the message history sidesteps that entirely.
    tools_used = [step for step in trace if step != "AGENT"]

    return {
        "answer": final_message.content,
        "tools_used": tools_used,
        "execution_trace": trace,
        "thread_id": thread_id,
        "error": result.get("error"),
    }