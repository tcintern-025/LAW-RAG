from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=2000, description="The user's question."
    )


class SourceChunk(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    disclaimer: str
    has_sufficient_context: bool


class HealthResponse(BaseModel):
    status: str
    client_id: str
    product_name: str
    document_count: int


class AgentAskRequest(BaseModel):
    question: str = Field(
        ..., min_length=3, max_length=2000, description="The user's question."
    )
    # Optional. When provided, the agent's LangGraph checkpointer continues
    # that conversation's message history (so "explain that in simple
    # words" resolves against a prior turn). Omit it for a stateless,
    # one-off question — a fresh thread_id is generated server-side.
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation/session id. Reuse the same value across calls to "
            "give the agent memory of earlier turns in this conversation. "
            "Omit for a one-off, history-free question."
        ),
    )


class AgentAskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    # Ordered, human-readable path through the graph for this turn, e.g.
    # ["AGENT", "search_legal_documents", "AGENT", "calculate", "AGENT"].
    execution_trace: list[str]
    # The thread_id this turn was recorded under — echoed back so the
    # caller can pass it on the next request to continue the conversation.
    thread_id: str
    # Most recent tool failure this turn, if any (None on a clean run).
    error: str | None = None


class SourceDocument(BaseModel):
    name: str
    chunk_count: int