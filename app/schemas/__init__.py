"""
Pydantic schemas for API requests and responses.
Structured output ensures every response includes cited_sections and confidence.
"""

from datetime import datetime

from pydantic import BaseModel, Field


# ── Request Schemas ──────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=1000, description="Legal question about BNS")
    session_id: str | None = Field(None, description="Session ID for conversation continuity")
    top_k: int = Field(5, ge=1, le=10, description="Number of context chunks to retrieve")


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(..., ge=-1, le=1, description="-1=bad, 0=neutral, 1=good")
    comment: str | None = None


# ── Response Schemas ─────────────────────────────────────────────────────────


class CitedSection(BaseModel):
    section_number: str
    section_title: str
    relevant_text: str
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    cited_sections: list[CitedSection]
    related_sections: list[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    message_id: str
    session_id: str
    model_used: str
    latency_ms: float
    cached: bool = False


class StreamChunk(BaseModel):
    """Schema for SSE streaming chunks."""
    type: str = Field(..., description="One of: token, metadata, done, error")
    content: str = ""
    metadata: dict | None = None


class SectionResponse(BaseModel):
    section_number: str
    section_title: str
    chapter: str
    full_text: str
    punishment: str | None = None
    related_sections: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    database: str
    vector_store: str
    primary_llm: str
    timestamp: datetime


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    message_id: str | None = None
    timestamp: datetime | None = None


class ConversationHistory(BaseModel):
    session_id: str
    messages: list[ChatMessage]
