"""
BNS Legal RAG — FastAPI Backend
================================
Endpoints:
  POST /api/query          — Main query endpoint (JSON response)
  POST /api/query/stream   — Streaming query endpoint (SSE)
  GET  /api/sections/{id}  — Get full section details
  POST /api/feedback       — Submit feedback on responses
  GET  /api/conversations/{session_id} — Get conversation history
  GET  /api/health         — Health check
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.config import get_settings
from app.core.logging import get_logger, log_latency, setup_logging
from app.db import Base, engine, get_db
from app.db.models import Conversation, Feedback, Message, QueryLog, Section
from app.llm.router import LLMRouter, build_context
from app.retrieval.hybrid import HybridRetriever, ResponseCache
from app.schemas import (
    ChatMessage,
    CitedSection,
    ConversationHistory,
    FeedbackRequest,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SectionResponse,
)

logger = get_logger(__name__)
settings = get_settings()

# ── Lifespan ──────────────────────────────────────────────────────────────────

retriever: HybridRetriever | None = None
llm_router: LLMRouter | None = None
cache: ResponseCache | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm_router, cache

    setup_logging()
    logger.info("Starting BNS Legal RAG API")

    # Create DB tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize components
    retriever = HybridRetriever()
    llm_router = LLMRouter()
    cache = ResponseCache()

    logger.info("All components initialized")
    yield

    # Cleanup
    await engine.dispose()
    logger.info("API shutdown complete")


# ── App Setup ─────────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="BNS Legal RAG API",
    description="AI-powered legal query platform for Bharatiya Nyaya Sanhita 2023",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Query Endpoint ────────────────────────────────────────────────────────────


@app.post("/api/query", response_model=QueryResponse)
@limiter.limit(settings.rate_limit)
async def query_bns(
    request: Request,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Main query endpoint. Retrieves relevant BNS sections and generates an answer.
    """
    start_total = time.perf_counter()
    session_id = body.session_id or str(uuid.uuid4())
    message_id = str(uuid.uuid4())
    log_entry = {"query": body.query, "session_id": session_id}

    try:
        # 1. Check cache
        cache_key = ResponseCache.generate_key(body.query, body.top_k)
        cached = await cache.get(cache_key, db)
        if cached:
            total_ms = round((time.perf_counter() - start_total) * 1000, 2)
            return QueryResponse(
                **cached,
                message_id=message_id,
                session_id=session_id,
                latency_ms=total_ms,
                cached=True,
            )

        # 2. Retrieve relevant chunks
        chunks, retrieval_latency = await retriever.retrieve(
            query=body.query,
            db=db,
            top_k=body.top_k,
        )

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No relevant BNS sections found for your query.",
            )

        # 3. Build context and generate
        context = build_context(chunks)

        # Get chat history for context
        chat_history = await _get_chat_history(session_id, db)

        llm_result = await llm_router.generate(
            query=body.query,
            context=context,
            chat_history=chat_history,
        )

        total_ms = round((time.perf_counter() - start_total) * 1000, 2)

        # 4. Build response
        cited_sections = [
            CitedSection(
                section_number=c.section_number,
                section_title=c.section_title,
                relevant_text=c.text[:500],
                similarity_score=c.similarity_score,
            )
            for c in chunks
        ]

        response = QueryResponse(
            answer=llm_result["answer"],
            cited_sections=cited_sections,
            related_sections=llm_result.get("related_sections", []),
            confidence_score=llm_result.get("confidence_score", 0.5),
            message_id=message_id,
            session_id=session_id,
            model_used=llm_result["model_used"],
            latency_ms=total_ms,
            cached=False,
        )

        # 5. Save to DB (conversation + cache + logs) — fire and forget
        await _save_conversation(session_id, body.query, response, db)
        await cache.set(
            cache_key,
            body.query,
            {
                "answer": response.answer,
                "cited_sections": [cs.model_dump() for cs in cited_sections],
                "related_sections": response.related_sections,
                "confidence_score": response.confidence_score,
                "model_used": response.model_used,
            },
            db,
        )
        await _log_query(body.query, session_id, llm_result, retrieval_latency, total_ms, db)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query failed", error=str(e), **log_entry)
        try:
            await db.rollback()
            await _log_query(body.query, session_id, {}, {}, 0, db, error=str(e))
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# ── Streaming Endpoint ────────────────────────────────────────────────────────


@app.post("/api/query/stream")
@limiter.limit(settings.rate_limit)
async def query_bns_stream(
    request: Request,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Streaming query endpoint using Server-Sent Events.
    Sends tokens as they're generated for lower perceived latency.
    """
    session_id = body.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            # Retrieve
            chunks, latency = await retriever.retrieve(
                query=body.query, db=db, top_k=body.top_k,
            )

            if not chunks:
                yield {"event": "error", "data": json.dumps({"error": "No relevant sections found"})}
                return

            # Send metadata first
            metadata = {
                "session_id": session_id,
                "chunks_retrieved": len(chunks),
                "sections": [
                    {"number": c.section_number, "title": c.section_title, "score": c.similarity_score}
                    for c in chunks
                ],
            }
            yield {"event": "metadata", "data": json.dumps(metadata)}

            # Stream LLM response
            context = build_context(chunks)
            chat_history = await _get_chat_history(session_id, db)

            async for token in llm_router.generate_stream(
                query=body.query, context=context, chat_history=chat_history,
            ):
                yield {"event": "token", "data": json.dumps({"token": token})}

            yield {"event": "done", "data": json.dumps({"status": "complete"})}

        except Exception as e:
            logger.error("Stream failed", error=str(e))
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())


# ── Section Detail Endpoint ───────────────────────────────────────────────────


@app.get("/api/sections/{section_number}", response_model=SectionResponse)
async def get_section(section_number: str, db: AsyncSession = Depends(get_db)):
    """Get full details of a specific BNS section."""
    result = await db.execute(
        select(Section).where(Section.section_number == section_number)
    )
    section = result.scalar_one_or_none()

    if not section:
        raise HTTPException(status_code=404, detail=f"Section {section_number} not found")

    return SectionResponse(
        section_number=section.section_number,
        section_title=section.section_title,
        chapter=f"Chapter {section.chapter_number} - {section.chapter_title}",
        full_text=section.full_text,
        punishment=section.punishment,
        related_sections=section.related_sections or [],
    )


# ── Feedback Endpoint ─────────────────────────────────────────────────────────


@app.post("/api/feedback")
async def submit_feedback(body: FeedbackRequest, db: AsyncSession = Depends(get_db)):
    """Submit feedback (👍/👎) on a response."""
    feedback = Feedback(
        message_id=uuid.UUID(body.message_id),
        rating=body.rating,
        comment=body.comment,
    )
    db.add(feedback)
    await db.commit()

    logger.info("Feedback received", message_id=body.message_id, rating=body.rating)
    return {"status": "ok", "message_id": body.message_id}


# ── Conversation History ──────────────────────────────────────────────────────


@app.get("/api/conversations/{session_id}", response_model=ConversationHistory)
async def get_conversation(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get conversation history for a session."""
    result = await db.execute(
        select(Conversation).where(Conversation.session_id == session_id)
    )
    conv = result.scalar_one_or_none()

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = [
        ChatMessage(
            role=msg.role,
            content=msg.content,
            message_id=str(msg.id),
            timestamp=msg.created_at,
        )
        for msg in conv.messages
    ]

    return ConversationHistory(session_id=session_id, messages=messages)


# ── Health Check ──────────────────────────────────────────────────────────────


@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint — checks DB, vector store, and LLM connectivity."""
    # Check DB
    db_status = "ok"
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    # Check ChromaDB
    vector_status = "ok"
    try:
        if retriever and retriever.semantic.collection:
            count = retriever.semantic.collection.count()
            vector_status = f"ok ({count} chunks)"
    except Exception:
        vector_status = "error"

    # Check LLM
    llm_status = "ok"
    if llm_router:
        if llm_router.primary:
            llm_status = f"ok (primary: groq, fallback: gemini)"
        elif llm_router.fallback:
            llm_status = "degraded (fallback only)"

    return HealthResponse(
        status="healthy" if all(s.startswith("ok") for s in [db_status, vector_status, llm_status]) else "degraded",
        database=db_status,
        vector_store=vector_status,
        primary_llm=llm_status,
        timestamp=datetime.now(timezone.utc),
    )


# ── Helper Functions ──────────────────────────────────────────────────────────


async def _get_chat_history(session_id: str, db: AsyncSession) -> list[dict]:
    """Get last 5 messages for conversation context."""
    result = await db.execute(
        select(Conversation).where(Conversation.session_id == session_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        return []

    return [
        {"role": msg.role, "content": msg.content}
        for msg in conv.messages[-5:]
    ]


async def _save_conversation(
    session_id: str,
    query: str,
    response: QueryResponse,
    db: AsyncSession,
) -> None:
    """Save user query and assistant response to conversation history."""
    # Get or create conversation
    result = await db.execute(
        select(Conversation).where(Conversation.session_id == session_id)
    )
    conv = result.scalar_one_or_none()

    if not conv:
        conv = Conversation(session_id=session_id)
        db.add(conv)
        await db.flush()

    # Add user message
    user_msg = Message(
        conversation_id=conv.id,
        role="user",
        content=query,
    )
    db.add(user_msg)

    # Add assistant message
    assistant_msg = Message(
        id=uuid.UUID(response.message_id),
        conversation_id=conv.id,
        role="assistant",
        content=response.answer,
        cited_sections=[cs.section_number for cs in response.cited_sections],
        model_used=response.model_used,
        latency_ms=response.latency_ms,
    )
    db.add(assistant_msg)
    await db.commit()


async def _log_query(
    query: str,
    session_id: str,
    llm_result: dict,
    retrieval_latency: dict,
    total_ms: float,
    db: AsyncSession,
    error: str | None = None,
) -> None:
    """Log query for observability."""
    log = QueryLog(
        query=query,
        session_id=session_id,
        total_latency_ms=total_ms,
        retrieval_latency_ms=retrieval_latency.get("semantic_ms", 0) + retrieval_latency.get("keyword_ms", 0),
        rerank_latency_ms=retrieval_latency.get("rerank_ms", 0),
        llm_latency_ms=llm_result.get("llm_latency_ms", 0),
        model_used=llm_result.get("model_used", ""),
        tokens_input=llm_result.get("tokens_input", 0),
        tokens_output=llm_result.get("tokens_output", 0),
        fallback_used=llm_result.get("fallback_used", False),
        chunks_retrieved=len(llm_result.get("cited_sections", [])),
        cache_hit=False,
        error=error,
    )
    db.add(log)
    await db.commit()
