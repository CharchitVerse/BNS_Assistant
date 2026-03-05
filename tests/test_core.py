"""
BNS Legal RAG — Core Unit Tests
"""

import json
import pytest


class TestBNSExtractor:
    def test_categorize_offence(self):
        from app.ingestion.pipeline import BNSExtractor
        extractor = BNSExtractor()

        assert extractor._categorize_offence("OFFENCES AGAINST THE STATE") == "Against State"
        assert extractor._categorize_offence("OF OFFENCES AGAINST PROPERTY") == "Property Offences"
        assert extractor._categorize_offence("PRELIMINARY") == "General"
        assert extractor._categorize_offence("UNKNOWN CHAPTER") == "Other"

    def test_cross_reference_pattern(self):
        from app.ingestion.pipeline import BNSExtractor
        import re

        extractor = BNSExtractor()
        text = "as defined in section 2 and sections 103, 104 and 105"
        refs = []
        for match in extractor.CROSS_REF_PATTERN.finditer(text):
            nums = re.findall(r"\d+", match.group(1))
            refs.extend(nums)

        assert "2" in refs
        assert "103" in refs
        assert "105" in refs


class TestChunker:
    def test_short_section_single_chunk(self):
        from app.ingestion.pipeline import BNSChunker, BNSSection

        chunker = BNSChunker(max_chunk_size=512)
        section = BNSSection(
            section_number="1",
            section_title="Short title",
            chapter_number="I",
            chapter_title="PRELIMINARY",
            full_text="This is a short section with few words.",
        )

        chunks = chunker.chunk_sections([section])
        assert len(chunks) == 1
        assert chunks[0].metadata["section_number"] == "1"
        assert chunks[0].metadata["total_chunks"] == 1

    def test_long_section_splits(self):
        from app.ingestion.pipeline import BNSChunker, BNSSection

        chunker = BNSChunker(max_chunk_size=10, overlap=2)
        section = BNSSection(
            section_number="2",
            section_title="Definitions",
            chapter_number="I",
            chapter_title="PRELIMINARY",
            full_text=" ".join(["word"] * 50),
        )

        chunks = chunker.chunk_sections([section])
        assert len(chunks) > 1
        assert all(c.metadata["section_number"] == "2" for c in chunks)

    def test_chunk_id_deterministic(self):
        from app.ingestion.pipeline import BNSChunker

        chunker = BNSChunker()
        id1 = chunker._generate_chunk_id("42", 0)
        id2 = chunker._generate_chunk_id("42", 0)
        id3 = chunker._generate_chunk_id("42", 1)

        assert id1 == id2
        assert id1 != id3


class TestResponseCache:
    def test_cache_key_normalized(self):
        from app.retrieval.hybrid import ResponseCache

        key1 = ResponseCache.generate_key("What is murder?", 5)
        key2 = ResponseCache.generate_key("what is murder?", 5)
        key3 = ResponseCache.generate_key("What is theft?", 5)

        assert key1 == key2
        assert key1 != key3

    def test_cache_key_includes_config(self):
        from app.retrieval.hybrid import ResponseCache

        key1 = ResponseCache.generate_key("test", 5)
        key2 = ResponseCache.generate_key("test", 10)
        assert key1 != key2


class TestSchemas:
    def test_query_request_valid(self):
        from app.schemas import QueryRequest

        req = QueryRequest(query="What is murder under BNS?")
        assert req.top_k == 5
        assert req.session_id is None

    def test_query_request_too_short(self):
        from app.schemas import QueryRequest

        with pytest.raises(Exception):
            QueryRequest(query="ab")

    def test_feedback_rating_bounds(self):
        from app.schemas import FeedbackRequest

        fb = FeedbackRequest(message_id="test-id", rating=1)
        assert fb.rating == 1

        with pytest.raises(Exception):
            FeedbackRequest(message_id="test-id", rating=5)


class TestLLMParsing:
    def _get_parser(self):
        from app.llm.router import GroqLLM
        return GroqLLM.__new__(GroqLLM)

    def test_parse_valid_json(self):
        llm = self._get_parser()
        raw = json.dumps({
            "answer": "Murder is defined in Section 101.",
            "cited_sections": ["101"],
            "related_sections": ["100"],
            "confidence_score": 0.9,
        })
        parsed = llm._parse_response(raw)
        assert parsed["answer"] == "Murder is defined in Section 101."
        assert "101" in parsed["cited_sections"]
        assert parsed["confidence_score"] == 0.9

    def test_parse_markdown_wrapped_json(self):
        llm = self._get_parser()
        raw = '```json\n{"answer": "test", "cited_sections": [], "confidence_score": 0.8}\n```'
        parsed = llm._parse_response(raw)
        assert parsed["answer"] == "test"
        assert parsed["confidence_score"] == 0.8

    def test_parse_plain_text_fallback(self):
        llm = self._get_parser()
        raw = "This is just plain text, not JSON."
        parsed = llm._parse_response(raw)
        assert parsed["answer"] == raw
        assert parsed["confidence_score"] == 0.5

    def test_confidence_clamped(self):
        llm = self._get_parser()
        raw = json.dumps({"answer": "x", "confidence_score": 1.5})
        parsed = llm._parse_response(raw)
        assert parsed["confidence_score"] <= 1.0

        raw = json.dumps({"answer": "x", "confidence_score": -0.5})
        parsed = llm._parse_response(raw)
        assert parsed["confidence_score"] >= 0.0
