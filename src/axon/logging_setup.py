"""Structured logging setup for Axon — idempotent configure_logging + request_id contextvar."""

from __future__ import annotations

import contextvars
import logging
from typing import Any

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdFilter(logging.Filter):
    """Inject the current request_id contextvar value into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call multiple times from different entry points.

    Sets up a single StreamHandler on the root logger that includes a
    ``rid=<request_id>`` field in every log line.  Subsequent calls are
    no-ops so entry points (CLI, API lifespan, MCP server) can each call
    this without duplicating handlers.
    """
    global _configured
    if _configured:
        return
    _configured = True
    fmt = "%(asctime)s [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s"
    handler = logging.StreamHandler()
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def set_request_id(rid: str) -> contextvars.Token[str]:
    """Set the request_id contextvar and return the reset token."""
    return _request_id_var.set(rid)


def reset_request_id(token: contextvars.Token[str]) -> None:
    """Reset the request_id contextvar using the token returned by set_request_id."""
    _request_id_var.reset(token)


def get_request_id() -> str:
    """Return the current request_id (defaults to '-' when not set)."""
    return _request_id_var.get()


__all__: list[Any] = [
    "RequestIdFilter",
    "configure_logging",
    "set_request_id",
    "reset_request_id",
    "get_request_id",
]
