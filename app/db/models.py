"""
SQLAlchemy models for PostgreSQL.
Tables: sections, conversations, messages, feedback, query_cache, query_logs.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── BNS Sections ─────────────────────────────────────────────────────────────


class Section(Base):
    """Stores the full BNS sections with metadata for keyword search."""

    __tablename__ = "sections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    section_number = Column(String(20), unique=True, nullable=False, index=True)
    section_title = Column(String(500), nullable=False)
    chapter_number = Column(String(20), nullable=False)
    chapter_title = Column(String(500), nullable=False)
    full_text = Column(Text, nullable=False)
    punishment = Column(Text, nullable=True)
    offence_category = Column(String(200), nullable=True)
    related_sections = Column(JSONB, default=list)

    # Full-text search vector
    search_vector = Column(TSVECTOR)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_sections_search_vector", "search_vector", postgresql_using="gin"),
        Index("ix_sections_chapter", "chapter_number"),
    )


# ── Conversations & Messages ─────────────────────────────────────────────────


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user | assistant
    content = Column(Text, nullable=False)
    cited_sections = Column(JSONB, default=list)
    model_used = Column(String(50), nullable=True)
    latency_ms = Column(Float, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    conversation = relationship("Conversation", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", uselist=False)


# ── Feedback ──────────────────────────────────────────────────────────────────


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.id"), unique=True, nullable=False)
    rating = Column(Integer, nullable=False)  # -1, 0, 1
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    message = relationship("Message", back_populates="feedback")


# ── Query Cache ───────────────────────────────────────────────────────────────


class QueryCache(Base):
    """SHA-256 hash of (query + config) → cached response. Avoids redundant LLM calls."""

    __tablename__ = "query_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cache_key = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256 hex
    query = Column(Text, nullable=False)
    response = Column(JSONB, nullable=False)
    hit_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)


# ── Query Logs (Observability) ────────────────────────────────────────────────


class QueryLog(Base):
    """Logs every query for observability: latency breakdown, tokens, model, errors."""

    __tablename__ = "query_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query = Column(Text, nullable=False)
    session_id = Column(String(100), nullable=True)

    # Latency breakdown
    total_latency_ms = Column(Float)
    embedding_latency_ms = Column(Float)
    retrieval_latency_ms = Column(Float)
    rerank_latency_ms = Column(Float)
    llm_latency_ms = Column(Float)

    # Model info
    model_used = Column(String(50))
    tokens_input = Column(Integer)
    tokens_output = Column(Integer)
    fallback_used = Column(Boolean, default=False)

    # Retrieval info
    chunks_retrieved = Column(Integer)
    top_similarity_score = Column(Float)
    cache_hit = Column(Boolean, default=False)

    # Error tracking
    error = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_query_logs_created", "created_at"),
    )
