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
"""

from functools import lru_cache

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from app.config import settings
from app.agent.tools import TOOLS


@lru_cache(maxsize=1)
def get_agent():
    llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )
    return create_react_agent(llm, TOOLS)


def run_agent(question: str) -> dict:
    """Run the agent on a question and return the answer plus a log of
    which tools were used along the way (the assignment's 'log which tool
    was selected' bonus, surfaced back to the caller instead of only to
    server logs).
    """
    agent = get_agent()
    result = agent.invoke({"messages": [("user", question)]})

    tools_used = []
    for msg in result["messages"]:
        calls = getattr(msg, "tool_calls", None)
        if calls:
            for call in calls:
                tools_used.append(call["name"])

    final_message = result["messages"][-1]

    return {
        "answer": final_message.content,
        "tools_used": tools_used,
    }
