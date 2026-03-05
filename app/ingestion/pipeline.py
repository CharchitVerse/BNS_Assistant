"""
BNS PDF Ingestion Pipeline
===========================

Pipeline: PDF → Extract → Section-Aware Chunking → Embed → Store in ChromaDB + Postgres

Usage:
    python -m app.ingestion.pipeline --pdf-path ./data/bns_english.pdf
"""

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.logging import get_logger, log_latency

logger = get_logger(__name__)
settings = get_settings()


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class BNSSection:
    """Represents a single BNS section extracted from the PDF."""
    section_number: str
    section_title: str
    chapter_number: str
    chapter_title: str
    full_text: str
    punishment: str | None = None
    offence_category: str | None = None
    related_sections: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A chunk ready for embedding and storage."""
    chunk_id: str
    text: str
    metadata: dict


# ── PDF Extractor ─────────────────────────────────────────────────────────────


class BNSExtractor:
    """
    Extracts structured sections from the BNS PDF.
    Preserves chapter hierarchy, illustrations, and explanations.
    """

    # Patterns to identify BNS structure
    CHAPTER_PATTERN = re.compile(
        r"CHAPTER\s+([IVXLC]+)\s*\n\s*(.+?)(?=\n)",
        re.IGNORECASE,
    )
    SECTION_PATTERN = re.compile(
        r"^(\d{1,3})\.\s*[\(\[]?1[\)\]]?\s*(.+?)(?=\n|$)",
        re.MULTILINE,
    )
    PUNISHMENT_PATTERN = re.compile(
        r"(?:shall be punished|punishable|imprisonment|fine|death|rigorous imprisonment|simple imprisonment)"
        r"(.+?)(?:\.|$)",
        re.IGNORECASE | re.DOTALL,
    )
    CROSS_REF_PATTERN = re.compile(
        r"(?:section|sections)\s+([\d,\s]+(?:and\s+\d+)?)",
        re.IGNORECASE,
    )

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        """Extract full text from PDF using PyMuPDF."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        with log_latency(logger, "pdf_extraction"):
            doc = fitz.open(str(pdf_path))
            full_text = ""
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc[page_num]
                full_text += page.get_text("text") + "\n"
            doc.close()

        logger.info("PDF extracted", pages=page_count, chars=len(full_text))
        return full_text

    def parse_sections(self, full_text: str) -> list[BNSSection]:
        """
        Parse the extracted text into structured BNS sections.
        Strategy: Split by section numbers and attach chapter metadata.
        """
        sections: list[BNSSection] = []

        # First pass: find all chapter boundaries
        chapters = {}
        for match in self.CHAPTER_PATTERN.finditer(full_text):
            chapters[match.start()] = {
                "number": match.group(1).strip(),
                "title": match.group(2).strip(),
            }

        chapter_positions = sorted(chapters.keys())

        # Second pass: split text by section numbers
        # BNS sections are numbered 1-358
        section_splits = list(
            re.finditer(r"\n(\d{1,3})\.\s+", full_text)
        )

        for i, match in enumerate(section_splits):
            sec_num = match.group(1)
            sec_num_int = int(sec_num)

            # Skip if not a valid BNS section number (1-358)
            if sec_num_int < 1 or sec_num_int > 358:
                continue

            # Get section text (until next section or end)
            start = match.start()
            end = section_splits[i + 1].start() if i + 1 < len(section_splits) else len(full_text)
            section_text = full_text[start:end].strip()

            # Determine which chapter this section belongs to
            chapter_num = "I"
            chapter_title = "PRELIMINARY"
            for pos in reversed(chapter_positions):
                if pos < start:
                    chapter_num = chapters[pos]["number"]
                    chapter_title = chapters[pos]["title"]
                    break

            # Extract section title (first line after section number)
            title_match = re.match(r"\d{1,3}\.\s*(?:[\(\[]?\d[\)\]]?\s*)?(.+?)(?:\.\s|\n)", section_text)
            section_title = title_match.group(1).strip() if title_match else f"Section {sec_num}"

            # Extract punishment info
            punishment = None
            punishment_match = self.PUNISHMENT_PATTERN.search(section_text)
            if punishment_match:
                punishment = punishment_match.group(0).strip()[:500]

            # Extract cross-referenced sections
            related = []
            for ref_match in self.CROSS_REF_PATTERN.finditer(section_text):
                ref_nums = re.findall(r"\d+", ref_match.group(1))
                related.extend(ref_nums)
            related = list(set(related) - {sec_num})

            # Determine offence category from chapter
            category = self._categorize_offence(chapter_title)

            sections.append(
                BNSSection(
                    section_number=sec_num,
                    section_title=section_title,
                    chapter_number=chapter_num,
                    chapter_title=chapter_title,
                    full_text=section_text,
                    punishment=punishment,
                    offence_category=category,
                    related_sections=related[:10],
                )
            )

        logger.info("Sections parsed", count=len(sections))
        return sections

    def _categorize_offence(self, chapter_title: str) -> str:
        """Map chapter titles to offence categories."""
        title_lower = chapter_title.lower()
        categories = {
            "preliminary": "General",
            "general exceptions": "Exceptions",
            "abetment": "Abetment & Conspiracy",
            "criminal conspiracy": "Abetment & Conspiracy",
            "offences against the state": "Against State",
            "offences against public tranquillity": "Public Tranquillity",
            "offences by public servants": "Public Servants",
            "offences against human body": "Against Human Body",
            "offences against woman": "Against Women",
            "property": "Property Offences",
            "marriage": "Marriage Offences",
            "cruelty": "Cruelty",
            "defamation": "Defamation",
            "intimidation": "Criminal Intimidation",
            "forgery": "Forgery",
        }
        for keyword, category in categories.items():
            if keyword in title_lower:
                return category
        return "Other"


# ── Chunker ───────────────────────────────────────────────────────────────────


class BNSChunker:
    """
    Section-aware chunking for legal text.
    
    Strategy:
    - Each BNS section becomes one chunk (most are <512 tokens).
    - Long sections (>512 tokens) are split with overlap, preserving metadata.
    - Illustrations and Explanations stay with their parent section.
    """

    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_sections(self, sections: list[BNSSection]) -> list[Chunk]:
        """Convert sections into chunks ready for embedding."""
        chunks: list[Chunk] = []

        for section in sections:
            text = section.full_text
            words = text.split()

            metadata = {
                "section_number": section.section_number,
                "section_title": section.section_title,
                "chapter_number": section.chapter_number,
                "chapter_title": section.chapter_title,
                "offence_category": section.offence_category or "Other",
                "punishment": section.punishment or "",
                "related_sections": section.related_sections if section.related_sections else ["none"],
            }

            if len(words) <= self.max_chunk_size:
                # Section fits in one chunk
                chunk_id = self._generate_chunk_id(section.section_number, 0, text)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata={**metadata, "chunk_index": 0, "total_chunks": 1},
                ))
            else:
                # Split long sections with overlap
                sub_chunks = self._split_with_overlap(words)
                for idx, sub_text in enumerate(sub_chunks):
                    chunk_id = self._generate_chunk_id(section.section_number, idx, sub_text)
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=sub_text,
                        metadata={
                            **metadata,
                            "chunk_index": idx,
                            "total_chunks": len(sub_chunks),
                        },
                    ))

        logger.info("Chunking complete", total_chunks=len(chunks))
        return chunks

    def _split_with_overlap(self, words: list[str]) -> list[str]:
        """Split word list into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.max_chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            start = end - self.overlap
        return chunks

    def _generate_chunk_id(self, section_number: str, chunk_index: int, text: str = "") -> str:
        """Deterministic chunk ID based on content."""
        raw = f"bns_s{section_number}_c{chunk_index}_{text[:50]}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Embedding & Storage ───────────────────────────────────────────────────────


class EmbeddingStore:
    """
    Generates embeddings locally and stores in ChromaDB.
    Runs on your MacBook — uses all-MiniLM-L6-v2 (384-dim, ~80MB RAM).
    """

    def __init__(self):
        self._model = None
        self._collection = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model", model=settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
        return self._model

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            self._collection = client.get_or_create_collection(
                name="bns_sections",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed_and_store(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """Embed all chunks and store in ChromaDB."""
        total = len(chunks)
        logger.info("Starting embedding", total_chunks=total, batch_size=batch_size)

        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]

            with log_latency(logger, f"embedding_batch_{i // batch_size}"):
                embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            logger.info(
                "Batch stored",
                batch=i // batch_size + 1,
                total_batches=(total + batch_size - 1) // batch_size,
            )

        logger.info("All embeddings stored in ChromaDB", total_chunks=total)


# ── Postgres Loader ───────────────────────────────────────────────────────────


class PostgresLoader:
    """Loads parsed sections into PostgreSQL for keyword search and metadata queries."""

    def load_sections(self, sections: list[BNSSection], db_url: str | None = None) -> None:
        """Load sections into Postgres using sync connection (for offline script)."""
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import Session as SyncSession

        from app.db.models import Base, Section

        url = db_url or settings.database_url_sync
        engine = create_engine(url, echo=False)

        # Create tables
        Base.metadata.create_all(engine)

        with SyncSession(engine) as session:
            for sec in sections:
                existing = session.query(Section).filter_by(section_number=sec.section_number).first()
                if existing:
                    # Update existing
                    existing.section_title = sec.section_title
                    existing.chapter_number = sec.chapter_number
                    existing.chapter_title = sec.chapter_title
                    existing.full_text = sec.full_text
                    existing.punishment = sec.punishment
                    existing.offence_category = sec.offence_category
                    existing.related_sections = sec.related_sections
                else:
                    db_section = Section(
                        section_number=sec.section_number,
                        section_title=sec.section_title,
                        chapter_number=sec.chapter_number,
                        chapter_title=sec.chapter_title,
                        full_text=sec.full_text,
                        punishment=sec.punishment,
                        offence_category=sec.offence_category,
                        related_sections=sec.related_sections,
                    )
                    session.add(db_section)

            session.commit()

            # Update full-text search vectors
            session.execute(text("""
                UPDATE sections 
                SET search_vector = to_tsvector('english', 
                    coalesce(section_number, '') || ' ' || 
                    coalesce(section_title, '') || ' ' || 
                    coalesce(full_text, '')
                )
            """))
            session.commit()

        logger.info("Sections loaded into Postgres", count=len(sections))


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_ingestion_pipeline(pdf_path: str) -> dict:
    """
    Full ingestion pipeline. Run on MacBook.
    
    Returns stats dict for logging.
    """
    from app.core.logging import setup_logging
    setup_logging()

    logger.info("=" * 60)
    logger.info("BNS Ingestion Pipeline Started", pdf=pdf_path)
    logger.info("=" * 60)

    # Step 1: Extract PDF
    extractor = BNSExtractor()
    full_text = extractor.extract_text_from_pdf(pdf_path)

    # Step 2: Parse into sections
    sections = extractor.parse_sections(full_text)
    logger.info("Parsed sections", count=len(sections))

    # Step 3: Chunk sections
    chunker = BNSChunker(max_chunk_size=512, overlap=50)
    chunks = chunker.chunk_sections(sections)

    # Step 4: Embed and store in ChromaDB
    store = EmbeddingStore()
    store.embed_and_store(chunks)

    # Step 5: Load into Postgres
    pg_loader = PostgresLoader()
    pg_loader.load_sections(sections)

    stats = {
        "pdf_chars": len(full_text),
        "sections_parsed": len(sections),
        "chunks_created": len(chunks),
        "embedding_model": settings.embedding_model,
        "chroma_dir": settings.chroma_persist_dir,
    }

    logger.info("Pipeline complete", **stats)
    return stats


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BNS PDF Ingestion Pipeline")
    parser.add_argument("--pdf-path", required=True, help="Path to BNS PDF file")
    args = parser.parse_args()

    stats = run_ingestion_pipeline(args.pdf_path)
    print(f"\n✅ Ingestion complete: {stats}")
