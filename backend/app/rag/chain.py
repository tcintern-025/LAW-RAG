"""
The RAG chain: retrieve -> build grounded prompt -> call Groq -> package a
structured, sourced answer.

Kept as a plain function rather than a heavier LangChain LCEL/agent
abstraction on purpose — for a single-hop RAG flow like this, a straight
line function is easier to read, debug, and explain to a non-technical
client than a chain of piped operators would be.
"""

from functools import lru_cache

from langchain_groq import ChatGroq

from app.config import settings
from app.rag.prompts import SYSTEM_PROMPT_TEMPLATE, format_context
from app.retrieval.retriever import get_retriever


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not settings.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file — see .env.example."
        )
    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
    )


def answer_question(question: str) -> dict:
    retriever = get_retriever()
    chunks = retriever.invoke(question)

    if not chunks:
        return {
            "answer": (
                "The provided documents don't contain enough information to "
                "answer this question."
            ),
            "sources": [],
            "disclaimer": settings.DISCLAIMER,
            "has_sufficient_context": False,
        }

    context = format_context(chunks)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context, question=question)

    llm = get_llm()
    response = llm.invoke(prompt)
    answer_text = response.content

    # A lightweight, transparent signal (not a hidden model score) that the
    # model itself said it couldn't answer from context. Good enough for a
    # UI badge without pretending to be a calibrated confidence metric.
    insufficient_phrases = [
        "don't contain enough information",
        "do not contain enough information",
        "doesn't contain enough information",
        "does not contain enough information",
    ]
    has_sufficient_context = not any(
        phrase in answer_text.lower() for phrase in insufficient_phrases
    )

    sources = [
        {
            "source": chunk.metadata.get("source", "unknown"),
            "page": chunk.metadata.get("page"),
            "chunk_id": chunk.metadata.get("chunk_id"),
            "excerpt": chunk.page_content[:400],
        }
        for chunk in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "disclaimer": settings.DISCLAIMER,
        "has_sufficient_context": has_sufficient_context,
    }
