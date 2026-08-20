"""
LangGraph State schema for the multi-step agent.

This replaces the implicit state that langgraph.prebuilt.create_react_agent
managed for us. Making it explicit is the whole point of this upgrade: every
node reads from and writes to this TypedDict, and LangGraph threads it
through the graph (and, via the checkpointer, across turns of the same
conversation) automatically.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    messages:
        The full conversation as a list of LangChain message objects
        (HumanMessage, AIMessage, ToolMessage, ...). `add_messages` is
        LangGraph's built-in reducer — instead of each node's return value
        *replacing* this field, new messages are *appended* to it. That's
        what makes both (a) multi-step tool loops within one turn and
        (b) multi-turn conversation history (via the checkpointer) work
        without any manual list-merging code.

    tools_used:
        Tool names invoked by the most recent tool_node execution. This has
        no reducer, so on a multi-step turn (two separate tool_node calls)
        it only reflects the LATEST batch, not the whole turn — that's fine,
        because it's an intermediate signal for the agent_node's own
        bookkeeping. The authoritative "which tools ran this turn" list the
        API returns is derived in run_agent() by scanning `messages` (which
        DOES accumulate correctly via add_messages), not from this field.

    error:
        The most recent tool failure message, if any, during this turn.
        `None` when every tool call (or no tool call) succeeded. The final
        response node can use this to decide whether to mention a
        degraded/partial answer.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    tools_used: list[str]
    error: str | None