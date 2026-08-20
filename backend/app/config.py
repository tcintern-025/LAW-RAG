"""
Central configuration for the RAG engine.

This is the ONE file you edit to re-skin this project for a different client or
industry (a different law firm, an insurance agency, an HR department, etc).
Everything else in the codebase reads from here — nothing else should hardcode
a document path, a model name, or brand copy.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # -------------------------------------------------------------------
    # Tenant / client identity — change these three when reselling this
    # engine to a new client. This becomes the ChromaDB collection name,
    # so each client's documents live in their own isolated collection.
    # -------------------------------------------------------------------
    CLIENT_ID: str = os.getenv("CLIENT_ID", "pakistan-law-demo")
    PRODUCT_NAME: str = os.getenv("PRODUCT_NAME", "Pakistan Law Assistant")
    PRODUCT_TAGLINE: str = os.getenv(
        "PRODUCT_TAGLINE",
        "Ask questions about Pakistani law and get grounded, sourced answers.",
    )

    # -------------------------------------------------------------------
    # Document source — point this at any client's document folder.
    # -------------------------------------------------------------------
    DOCUMENTS_DIR: Path = Path(
        os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "data" / "documents"))
    )
    CHROMA_PERSIST_DIR: Path = Path(
        os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))
    )

    # -------------------------------------------------------------------
    # Chunking. Legal text is dense and section-structured, so we use a
    # moderate chunk size with meaningful overlap so a clause split across
    # a chunk boundary is still retrievable from either side.
    # -------------------------------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # -------------------------------------------------------------------
    # Embeddings — Hugging Face, runs locally, no per-call API cost.
    # Swap MODEL for a multilingual one (e.g. intfloat/multilingual-e5-base)
    # if a client's documents aren't in English.
    # -------------------------------------------------------------------
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------------------------------------------------
    # LLM — Groq. Swap MODEL for any Groq-hosted model.
    # -------------------------------------------------------------------
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # -------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    RETRIEVAL_FETCH_K: int = int(os.getenv("RETRIEVAL_FETCH_K", "20"))
    USE_MMR: bool = os.getenv("USE_MMR", "true").lower() == "true"

    # -------------------------------------------------------------------
    # CORS — add your deployed frontend's URL here in production.
    # -------------------------------------------------------------------
    ALLOWED_ORIGINS: list[str] = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000"
    ).split(",")

    # -------------------------------------------------------------------
    # Branding / legal copy — this is what changes per industry.
    # See rag/prompts.py for how these are used.
    # -------------------------------------------------------------------
    DISCLAIMER: str = os.getenv(
        "DISCLAIMER",
        "This response is generated for general informational and educational "
        "purposes only. It is not legal advice and does not create a "
        "lawyer-client relationship. Consult a licensed attorney in your "
        "jurisdiction before acting on any legal matter.",
    )
    DOMAIN_LABEL: str = os.getenv("DOMAIN_LABEL", "Pakistani law")


settings = Settings()