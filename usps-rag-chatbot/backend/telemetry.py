"""Structured logging. In production wire this to OpenTelemetry / CloudWatch.

Every chat request writes a single JSON line capturing the question, retrieved
chunk IDs, model, and latency. No user PII is stored beyond what the caller
submits; tune redaction in `redact()` per your compliance posture.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict

logger = logging.getLogger("usps_rag")


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def redact(text: str) -> str:
    """Placeholder redaction. Override to mask PII (SSN, CC, etc.) per policy."""
    return text


def audit(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, ensure_ascii=False))


@contextmanager
def timer():
    t0 = time.perf_counter()
    elapsed = {"ms": 0}
    try:
        yield elapsed
    finally:
        elapsed["ms"] = int((time.perf_counter() - t0) * 1000)
