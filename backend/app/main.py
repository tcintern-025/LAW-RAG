from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import AskRequest, AskResponse, HealthResponse, SourceDocument
from app.rag.chain import answer_question
from app.retrieval.vectorstore import get_vectorstore, collection_is_empty

app = FastAPI(
    title=settings.PRODUCT_NAME,
    description=settings.PRODUCT_TAGLINE,
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    store = get_vectorstore()
    return HealthResponse(
        status="ok",
        client_id=settings.CLIENT_ID,
        product_name=settings.PRODUCT_NAME,
        document_count=store._collection.count(),
    )


@app.get("/sources", response_model=list[SourceDocument])
def list_sources() -> list[SourceDocument]:
    """List the documents currently indexed for this client, with chunk counts."""
    store = get_vectorstore()
    records = store.get()
    counts = Counter(m.get("source", "unknown") for m in records["metadatas"])
    return [
        SourceDocument(name=name, chunk_count=count)
        for name, count in sorted(counts.items())
    ]


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if collection_is_empty():
        raise HTTPException(
            status_code=503,
            detail=(
                "No documents are indexed yet for this client. Run "
                "`python -m app.ingestion.build_index` first."
            ),
        )

    try:
        result = answer_question(request.question)
    except RuntimeError as exc:
        # e.g. missing GROQ_API_KEY — a config problem, not a client error.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AskResponse(**result)
