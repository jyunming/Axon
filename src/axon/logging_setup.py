"""Structured logging setup for Axon — idempotent configure_logging + request_id contextvar."""

from __future__ import annotations

import contextvars
import logging
import threading

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_AXON_HANDLER_NAME = "_axon_request_id_handler"
_configure_lock = threading.Lock()


class RequestIdFilter(logging.Filter):
    """Inject the current request_id contextvar value into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get()
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent: safe to call multiple times from different entry points.

    Adds a single named StreamHandler that includes a ``rid=<request_id>``
    field in every log line.  If the handler is already present (same name),
    updates its formatter and level but does not add a second copy.  Does NOT
    clear handlers installed by the embedding application or other libraries.
    """
    fmt = "%(asctime)s [%(levelname)s] [rid=%(request_id)s] %(name)s: %(message)s"
    formatter = logging.Formatter(fmt)
    root = logging.getLogger()
    root.setLevel(level)

    with _configure_lock:
        for handler in root.handlers:
            if handler.get_name() == _AXON_HANDLER_NAME:
                handler.setFormatter(formatter)
                return

        handler = logging.StreamHandler()
        handler.set_name(_AXON_HANDLER_NAME)
        handler.addFilter(RequestIdFilter())
        handler.setFormatter(formatter)
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


__all__: list[str] = [
    "RequestIdFilter",
    "configure_logging",
    "set_request_id",
    "reset_request_id",
    "get_request_id",
]
