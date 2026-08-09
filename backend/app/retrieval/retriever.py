"""
Retriever construction.

Uses Maximal Marginal Relevance (MMR) instead of plain top-k similarity by
default. Plain similarity search on legal text tends to return several near-
duplicate chunks from the same Article/Section when a question is broadly
phrased, which wastes context budget and gives a one-sided source list. MMR
re-ranks for relevance *and* diversity, so a 5-result set is more likely to
span multiple relevant provisions instead of five paraphrases of one.
Set USE_MMR=false in .env to fall back to plain similarity search, which is
faster and fine for smaller, less repetitive document sets.
"""

from langchain_core.vectorstores import VectorStoreRetriever

from app.config import settings
from app.retrieval.vectorstore import get_vectorstore


def get_retriever() -> VectorStoreRetriever:
    store = get_vectorstore()

    if settings.USE_MMR:
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": settings.RETRIEVAL_TOP_K,
                "fetch_k": settings.RETRIEVAL_FETCH_K,
                "lambda_mult": 0.5,
            },
        )

    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.RETRIEVAL_TOP_K},
    )
