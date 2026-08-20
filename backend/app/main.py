import uuid
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    SourceDocument,
    AgentAskRequest,
    AgentAskResponse,
    ResetConversationRequest,
    ResetConversationResponse,
)
from app.rag.chain import answer_question
from app.agent.graph import run_agent, reset_conversation
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


@app.post("/agent/ask", response_model=AgentAskResponse)
def agent_ask(request: AgentAskRequest) -> AgentAskResponse:
    """Tool-calling agent endpoint. Unlike /ask, this doesn't always search
    the documents — it decides whether a tool is needed at all (legal
    search, calculator, or date), calls it if so, and returns the final
    answer along with which tool(s) it used.

    Conversation memory: pass the `thread_id` you got back from a previous
    call to continue that conversation (the agent will have the prior
    turns available when answering). Omit `thread_id` to start a fresh,
    isolated conversation — one is generated and returned to you.
    """
    thread_id = request.thread_id or str(uuid.uuid4())

    try:
        result = run_agent(request.question, thread_id=thread_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AgentAskResponse(**result)


@app.post("/agent/reset", response_model=ResetConversationResponse)
def agent_reset(request: ResetConversationRequest) -> ResetConversationResponse:
    """Clear one conversation's memory. Only the given thread_id is
    affected — every other conversation (this user's other tabs, or any
    other user's threads) is untouched.
    """
    reset_conversation(request.thread_id)
    return ResetConversationResponse(thread_id=request.thread_id, status="cleared")