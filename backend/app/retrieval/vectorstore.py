"""
Vector store management.

Collections are namespaced by `settings.CLIENT_ID`. That's the whole trick
behind reselling this engine: point CLIENT_ID + DOCUMENTS_DIR at a new
client's documents, run the ingestion script, and their data lives in its
own Chroma collection — fully isolated from every other client — with zero
code changes.
"""

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    # Cached so the (fairly large) embedding model is only loaded into memory
    # once per process, not once per request.
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.CLIENT_ID,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.CHROMA_PERSIST_DIR),
    )


def collection_is_empty() -> bool:
    store = get_vectorstore()
    return store._collection.count() == 0
