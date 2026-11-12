"""Shared logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Structured single-line JSON output (machine friendly for log scraping)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key in ("request_id", "paper_id", "session_id"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure root logging once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn duplicates access logs; keep the application logger quiet enough.
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # When stdout is a file (the usual deployment) Python block-buffers it, so
    # the log file stays empty until the buffer fills.  That makes a running
    # service impossible to observe - which is exactly when the log matters.
    # Line buffering keeps the file current; it costs nothing on a terminal.
    try:
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    except (AttributeError, ValueError):  # pragma: no cover - exotic stdout
        pass

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Convenience accessor used across the code base."""
    return logging.getLogger(name)
