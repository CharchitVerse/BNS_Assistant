"""
Structured logging with structlog.
Every request logs: query, latency_ms, chunks_retrieved, llm_model, tokens_used.
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Any

import structlog

from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if not settings.is_production
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def log_latency(logger: structlog.BoundLogger, operation: str):
    """Context manager to track and log operation latency."""
    start = time.perf_counter()
    result: dict[str, Any] = {}
    try:
        yield result
    finally:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        result["latency_ms"] = elapsed_ms
        logger.info(f"{operation} completed", latency_ms=elapsed_ms, **{k: v for k, v in result.items() if k != "latency_ms"})
