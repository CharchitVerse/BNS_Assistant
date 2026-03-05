"""
Hybrid Retrieval Layer
======================
Combines:
1. Semantic search (ChromaDB cosine similarity)
2. Keyword search (Postgres full-text search with pg_trgm)
3. Cross-encoder reranking (ms-marco-MiniLM-L6-v2)

Fusion: Reciprocal Rank Fusion (RRF) with configurable weights.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger, log_latency

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RetrievedChunk:
    """A chunk retrieved from search with scores."""
    chunk_id: str
    text: str
    section_number: str
    section_title: str
    chapter: str
    similarity_score: float
    metadata: dict = field(default_factory=dict)


# ── Semantic Search (ChromaDB) ────────────────────────────────────────────────


class SemanticSearcher:
    """Vector similarity search using ChromaDB."""

    def __init__(self):
        self._model = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            self._collection = client.get_collection("bns_sections")
        return self._collection

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_metadata: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Perform semantic search on ChromaDB."""
        query_embedding = self.model.encode([query]).tolist()

        where_filter = None
        if filter_metadata:
            where_filter = {
                "$and": [
                    {k: {"$eq": v}} for k, v in filter_metadata.items()
                ]
            } if len(filter_metadata) > 1 else {
                k: {"$eq": v} for k, v in filter_metadata.items()
            }

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances; for cosine, similarity = 1 - distance
                distance = results["distances"][0][i]
                similarity = 1.0 - distance

                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    section_number=meta.get("section_number", ""),
                    section_title=meta.get("section_title", ""),
                    chapter=f"Chapter {meta.get('chapter_number', '')} - {meta.get('chapter_title', '')}",
                    similarity_score=round(similarity, 4),
                    metadata=meta,
                ))

        return chunks


# ── Keyword Search (Postgres FTS) ─────────────────────────────────────────────


class KeywordSearcher:
    """Full-text search using PostgreSQL tsvector + ts_rank."""

    async def search(
        self,
        query: str,
        db: AsyncSession,
        top_k: int = 20,
    ) -> list[RetrievedChunk]:
        """Perform keyword search using Postgres FTS."""
        # Convert query to tsquery format
        query_terms = " & ".join(query.split())

        sql = text("""
            SELECT 
                section_number,
                section_title,
                chapter_number,
                chapter_title,
                full_text,
                offence_category,
                punishment,
                related_sections,
                ts_rank(search_vector, to_tsquery('english', :query)) as rank
            FROM sections
            WHERE search_vector @@ to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """)

        result = await db.execute(sql, {"query": query_terms, "limit": top_k})
        rows = result.fetchall()

        chunks = []
        for row in rows:
            chunks.append(RetrievedChunk(
                chunk_id=f"pg_{row.section_number}",
                text=row.full_text,
                section_number=row.section_number,
                section_title=row.section_title,
                chapter=f"Chapter {row.chapter_number} - {row.chapter_title}",
                similarity_score=round(float(row.rank), 4),
                metadata={
                    "section_number": row.section_number,
                    "section_title": row.section_title,
                    "chapter_number": row.chapter_number,
                    "chapter_title": row.chapter_title,
                    "offence_category": row.offence_category,
                    "punishment": row.punishment,
                    "related_sections": row.related_sections or [],
                },
            ))

        return chunks


# ── Reranker ──────────────────────────────────────────────────────────────────


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Takes top-20 from hybrid search → returns top-5 with refined scores.
    """

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model", model=settings.reranker_model)
            self._model = CrossEncoder(settings.reranker_model)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank chunks using cross-encoder and return top-k."""
        if not chunks:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        # Update scores and sort
        for chunk, score in zip(chunks, scores):
            chunk.similarity_score = round(float(score), 4)

        reranked = sorted(chunks, key=lambda c: c.similarity_score, reverse=True)
        return reranked[:top_k]


# ── Hybrid Search + Reciprocal Rank Fusion ────────────────────────────────────


class HybridRetriever:
    """
    Combines semantic + keyword search using Reciprocal Rank Fusion (RRF),
    then reranks with cross-encoder.
    
    Flow: Query → [Semantic + Keyword] → RRF Fusion → Rerank → Top-K
    """

    def __init__(self):
        self.semantic = SemanticSearcher()
        self.keyword = KeywordSearcher()
        self.reranker = CrossEncoderReranker()

    async def retrieve(
        self,
        query: str,
        db: AsyncSession,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> tuple[list[RetrievedChunk], dict]:
        """
        Full hybrid retrieval pipeline.
        
        Returns:
            (reranked_chunks, latency_breakdown)
        """
        top_k = top_k or settings.top_k_rerank
        retrieval_k = settings.top_k_retrieval
        latency = {}

        # 1. Semantic search
        start = time.perf_counter()
        semantic_results = self.semantic.search(
            query, top_k=retrieval_k, filter_metadata=filter_metadata
        )
        latency["semantic_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # 2. Keyword search
        start = time.perf_counter()
        try:
            keyword_results = await self.keyword.search(query, db, top_k=retrieval_k)
        except Exception as e:
            logger.warning("Keyword search failed, using semantic only", error=str(e))
            keyword_results = []
        latency["keyword_ms"] = round((time.perf_counter() - start) * 1000, 2)

        # 3. Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
            semantic_weight=settings.hybrid_search_semantic_weight,
            keyword_weight=settings.hybrid_search_keyword_weight,
        )

        # 4. Rerank top candidates
        candidates = fused[:retrieval_k]
        start = time.perf_counter()
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        latency["rerank_ms"] = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "Retrieval complete",
            semantic_hits=len(semantic_results),
            keyword_hits=len(keyword_results),
            fused=len(fused),
            reranked=len(reranked),
            **latency,
        )

        return reranked, latency

    def _reciprocal_rank_fusion(
        self,
        semantic: list[RetrievedChunk],
        keyword: list[RetrievedChunk],
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        k: int = 60,
    ) -> list[RetrievedChunk]:
        """
        Combine two ranked lists using RRF.
        Score = weight * (1 / (k + rank))
        """
        scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        # Score semantic results
        for rank, chunk in enumerate(semantic):
            key = chunk.section_number or chunk.chunk_id
            scores[key] = scores.get(key, 0) + semantic_weight * (1.0 / (k + rank + 1))
            chunk_map[key] = chunk

        # Score keyword results
        for rank, chunk in enumerate(keyword):
            key = chunk.section_number or chunk.chunk_id
            scores[key] = scores.get(key, 0) + keyword_weight * (1.0 / (k + rank + 1))
            if key not in chunk_map:
                chunk_map[key] = chunk

        # Sort by fused score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        result = []
        for key in sorted_keys:
            chunk = chunk_map[key]
            chunk.similarity_score = round(scores[key], 4)
            result.append(chunk)

        return result


# ── Cache Layer ───────────────────────────────────────────────────────────────


class ResponseCache:
    """
    SHA-256 cache for query responses.
    Cache key = hash(normalized_query + top_k).
    Avoids redundant LLM calls for repeated/similar questions.
    """

    @staticmethod
    def generate_key(query: str, top_k: int = 5) -> str:
        normalized = query.strip().lower()
        raw = f"{normalized}|{top_k}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def get(self, cache_key: str, db: AsyncSession) -> dict | None:
        """Look up cached response."""
        sql = text("""
            UPDATE query_cache 
            SET hit_count = hit_count + 1 
            WHERE cache_key = :key 
              AND (expires_at IS NULL OR expires_at > NOW())
            RETURNING response
        """)
        result = await db.execute(sql, {"key": cache_key})
        row = result.fetchone()
        if row:
            logger.info("Cache hit", cache_key=cache_key[:16])
            return row.response
        return None

    async def set(
        self,
        cache_key: str,
        query: str,
        response: dict,
        db: AsyncSession,
        ttl_hours: int = 168,  # 1 week
    ) -> None:
        """Store response in cache."""
        sql = text("""
                   INSERT INTO query_cache (id, cache_key, query, response, expires_at)
                   VALUES (gen_random_uuid(), :key, :query, :response,
                           NOW() + :ttl * INTERVAL '1 hour') ON CONFLICT (cache_key) DO
                   UPDATE
                       SET response = :response, hit_count = 0, expires_at = NOW() + :ttl * INTERVAL '1 hour'
                   """)
        await db.execute(sql, {
            "key": cache_key,
            "query": query,
            "response": json.dumps(response),
            "ttl": ttl_hours,
        })
        await db.commit()
