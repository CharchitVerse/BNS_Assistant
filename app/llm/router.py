"""
LLM Inference Layer
====================
Primary: Groq (Llama 3.1 70B) — free tier, ~300ms inference
Fallback: Google Gemini 1.5 Flash — triggers on timeout / 429 / 5xx

Features:
- Dual LLM routing with auto-failover
- Structured legal prompt engineering
- Token counting and cost tracking
- Streaming support via SSE
"""

import json
import time
from abc import ABC, abstractmethod
from typing import AsyncGenerator

from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.hybrid import RetrievedChunk

logger = get_logger(__name__)
settings = get_settings()


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal expert AI assistant specializing in the Bharatiya Nyaya Sanhita (BNS) 2023 — India's new criminal law code that replaced the Indian Penal Code (IPC).

Your role is to provide accurate, well-cited answers about BNS provisions, offences, punishments, and legal interpretations.

STRICT RULES:
1. ONLY answer based on the provided BNS sections. Never fabricate section numbers or legal provisions.
2. ALWAYS cite specific section numbers (e.g., "Section 103(1)") when referencing BNS provisions.
3. QUOTE relevant text from the provided sections to support your answer.
4. CLEARLY state the punishment/penalty when discussing offences.
5. If the question cannot be answered from the provided sections, say so explicitly.
6. FLAG any ambiguities or areas where legal interpretation may vary.
7. Add a disclaimer: "This is AI-generated legal information, not legal advice. Consult a qualified lawyer."

RESPONSE FORMAT:
- Start with a direct answer to the question
- Cite specific sections with quoted text
- State applicable punishments
- Mention related sections if relevant
- End with the disclaimer

You must respond in valid JSON format with these fields:
{
    "answer": "Your detailed answer with section citations",
    "cited_sections": ["103", "104", "105"],
    "related_sections": ["2", "6"],
    "confidence_score": 0.85
}

confidence_score should be:
- 0.9-1.0: Direct match, clear answer from provided text
- 0.7-0.9: Good match, answer requires some interpretation
- 0.5-0.7: Partial match, answer may be incomplete
- Below 0.5: Low confidence, sections may not fully address the question
"""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks for the LLM prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"--- SECTION {chunk.section_number}: {chunk.section_title} ---\n"
            f"Chapter: {chunk.chapter}\n"
            f"Text: {chunk.text}\n"
        )
    return "\n".join(context_parts)


def build_user_prompt(query: str, context: str, chat_history: list[dict] | None = None) -> str:
    """Build the user prompt with context and optional chat history."""
    history_str = ""
    if chat_history:
        history_parts = []
        for msg in chat_history[-5:]:  # Last 5 messages for context window
            history_parts.append(f"{msg['role'].upper()}: {msg['content']}")
        history_str = f"\n\nPREVIOUS CONVERSATION:\n" + "\n".join(history_parts) + "\n"

    return f"""Based on the following BNS sections, answer the user's question.

RELEVANT BNS SECTIONS:
{context}
{history_str}
USER QUESTION: {query}

Respond in the JSON format specified in your instructions."""


# ── Base LLM Interface ────────────────────────────────────────────────────────


class BaseLLM(ABC):
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """Generate a response. Returns dict with answer, tokens, model info."""
        ...

    @abstractmethod
    async def generate_stream(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream tokens one by one."""
        ...

    def _parse_response(self, raw_text: str) -> dict:
        """Parse LLM response, handling both JSON and plain text."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in raw_text:
                json_str = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                json_str = raw_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_text.strip()

            parsed = json.loads(json_str)
            return {
                "answer": parsed.get("answer", raw_text),
                "cited_sections": parsed.get("cited_sections", []),
                "related_sections": parsed.get("related_sections", []),
                "confidence_score": min(max(parsed.get("confidence_score", 0.5), 0.0), 1.0),
            }
        except (json.JSONDecodeError, IndexError):
            # Fallback: return raw text as answer
            logger.warning("Failed to parse LLM JSON response, using raw text")
            return {
                "answer": raw_text,
                "cited_sections": [],
                "related_sections": [],
                "confidence_score": 0.5,
            }


# ── Groq LLM (Primary) ───────────────────────────────────────────────────────


class GroqLLM(BaseLLM):
    """Groq API — Llama 3.1 70B. Free tier: 30 RPM, blazing fast (~300ms)."""

    def __init__(self):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = "llama-3.3-70b-versatile"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    async def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        user_prompt = build_user_prompt(query, context, chat_history)

        start = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=settings.max_output_tokens,
            temperature=0.1,  # Low temp for factual legal responses
            response_format={"type": "json_object"},
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        raw_text = response.choices[0].message.content or ""
        parsed = self._parse_response(raw_text)

        return {
            **parsed,
            "model_used": f"groq/{self.model}",
            "tokens_input": response.usage.prompt_tokens if response.usage else 0,
            "tokens_output": response.usage.completion_tokens if response.usage else 0,
            "llm_latency_ms": latency_ms,
        }

    async def generate_stream(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        user_prompt = build_user_prompt(query, context, chat_history)

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=settings.max_output_tokens,
            temperature=0.1,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ── Gemini LLM (Fallback) ────────────────────────────────────────────────────


class GeminiLLM(BaseLLM):
    """Google Gemini 1.5 Flash. Free tier: 60 RPM. Fallback for Groq failures."""

    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(
            "gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                max_output_tokens=settings.max_output_tokens,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
    async def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        user_prompt = build_user_prompt(query, context, chat_history)

        start = time.perf_counter()
        response = await self.model.generate_content_async(user_prompt)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        raw_text = response.text or ""
        parsed = self._parse_response(raw_text)

        # Gemini doesn't expose token counts in the same way
        token_count = response.usage_metadata
        return {
            **parsed,
            "model_used": "gemini/gemini-2.0-flash",
            "tokens_input": token_count.prompt_token_count if token_count else 0,
            "tokens_output": token_count.candidates_token_count if token_count else 0,
            "llm_latency_ms": latency_ms,
        }

    async def generate_stream(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        user_prompt = build_user_prompt(query, context, chat_history)

        response = await self.model.generate_content_async(
            user_prompt,
            stream=True,
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text


# ── LLM Router (Primary + Fallback) ──────────────────────────────────────────


class LLMRouter:
    """
    Routes to primary LLM (Groq) with auto-failover to fallback (Gemini).
    Triggers fallback on: timeout, 429 (rate limit), 5xx errors.
    """

    def __init__(self):
        self.primary: BaseLLM | None = None
        self.fallback: BaseLLM | None = None
        self._init_providers()

    def _init_providers(self):
        if settings.groq_api_key:
            self.primary = GroqLLM()
            logger.info("Primary LLM initialized", provider="Groq")
        if settings.gemini_api_key:
            self.fallback = GeminiLLM()
            logger.info("Fallback LLM initialized", provider="Gemini")

        if not self.primary and not self.fallback:
            raise RuntimeError("No LLM providers configured. Set GROQ_API_KEY or GEMINI_API_KEY.")

    async def generate(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> dict:
        """Generate with primary, failover to fallback on error."""
        fallback_used = False

        if self.primary:
            try:
                result = await self.primary.generate(query, context, chat_history)
                result["fallback_used"] = False
                return result
            except Exception as e:
                logger.warning(
                    "Primary LLM failed, switching to fallback",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                fallback_used = True

        if self.fallback:
            try:
                result = await self.fallback.generate(query, context, chat_history)
                result["fallback_used"] = fallback_used
                return result
            except Exception as e:
                logger.error("Fallback LLM also failed", error=str(e))
                raise

        raise RuntimeError("All LLM providers failed")

    async def generate_stream(
        self,
        query: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream with primary, failover to fallback."""
        if self.primary:
            try:
                async for token in self.primary.generate_stream(query, context, chat_history):
                    yield token
                return
            except Exception as e:
                logger.warning("Primary LLM stream failed, switching to fallback", error=str(e))

        if self.fallback:
            async for token in self.fallback.generate_stream(query, context, chat_history):
                yield token
