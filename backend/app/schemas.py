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


class SourceDocument(BaseModel):
    name: str
    chunk_count: int
