"""
Tool definitions for the AI agent.

Each tool is a plain Python function wrapped with LangChain's @tool decorator,
which turns its signature + docstring into a schema the LLM can call. Two
things every tool here does on purpose, per the assignment's bonus criteria:

1. Logs which tool was invoked, with its input, before running.
2. Catches its own errors and returns a readable string instead of raising,
   so a failing tool degrades the agent's answer instead of crashing it.
"""

import ast
import logging
import operator
from datetime import datetime, timezone

from langchain_core.tools import tool

from app.rag.chain import answer_question

logger = logging.getLogger("agent.tools")
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Tool 1 — search the existing RAG knowledge base
# ---------------------------------------------------------------------------
@tool
def search_legal_documents(query: str) -> str:
    """Search the indexed Pakistani legal documents (Constitution, Penal Code,
    Contract Act, PECA) and return a grounded answer with sources. Use this
    for any question about Pakistani law, legal sections, rights, or
    penalties. Do not use this for math or date questions.
    """
    logger.info("tool_selected=search_legal_documents input=%r", query)
    try:
        result = answer_question(query)
        sources = ", ".join(
            f"{s['source']} ({s['chunk_id']})" for s in result["sources"]
        )
        return (
            f"{result['answer']}\n\n"
            f"Sources: {sources if sources else 'none found'}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("search_legal_documents failed")
        return f"search_legal_documents tool failed: {exc}"


# ---------------------------------------------------------------------------
# Tool 2 — calculator
# ---------------------------------------------------------------------------
# Deliberately NOT using eval()/exec() on model-provided input — that's a
# code-execution vulnerability. Instead we parse the expression into an AST
# and only allow numeric literals and basic arithmetic operators.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains an unsupported operation")


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Use this for numeric
    questions such as totaling a fine, converting a sentence length, or
    splitting a compensation amount — e.g. '50000 + 25000' or '3 * 12'.
    Only numbers and + - * / ** are supported, no variables or functions.
    """
    logger.info("tool_selected=calculate input=%r", expression)
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("calculate failed")
        return f"calculate tool failed: could not evaluate '{expression}' ({exc})"


# ---------------------------------------------------------------------------
# Tool 3 — current date
# ---------------------------------------------------------------------------
@tool
def get_current_date() -> str:
    """Return today's date. Useful for computing legal deadlines, such as
    the 24-hour requirement to produce an arrested person before a
    magistrate under Article 10 of the Constitution.
    """
    logger.info("tool_selected=get_current_date input=none")
    try:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d (UTC)")
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_current_date failed")
        return f"get_current_date tool failed: {exc}"


TOOLS = [search_legal_documents, calculate, get_current_date]
