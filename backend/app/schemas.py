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
    thread_id: str | None = Field(
        None,
        description=(
            "Conversation to continue. Omit this on the first message of a "
            "new conversation — the server generates one and returns it in "
            "the response; send that same value back on every follow-up to "
            "keep the agent's memory of this conversation."
        ),
    )


class AgentAskResponse(BaseModel):
    answer: str
    tools_used: list[str]
    thread_id: str = Field(
        ..., description="Echo this back on the next request to continue this conversation."
    )


class ResetConversationRequest(BaseModel):
    thread_id: str = Field(
        ..., description="The conversation to clear. Only this thread's history is affected."
    )


class ResetConversationResponse(BaseModel):
    thread_id: str
    status: str


class SourceDocument(BaseModel):
    name: str
    chunk_count: int