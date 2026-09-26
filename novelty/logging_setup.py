"""Logging configuration for the entry points (CLI, server, scripts).

Library modules only create loggers (``logging.getLogger(__name__)``); they never configure
handlers. The entry points call ``configure_logging`` once.

    NOVELTY_LOG_LEVEL   DEBUG | INFO (default) | WARNING | ERROR
    NOVELTY_LOG_FILE    optional path; log lines are appended there as well as to stderr

Every line carries a request id (``[r000012]``) when one is set, so the lines belonging to one
web request can be grepped together. At INFO each scored input produces two lines (the input
and the result); DEBUG adds the full per-signal calculation.
"""

from __future__ import annotations

import logging
import os
import sys
from contextvars import ContextVar

request_id: ContextVar[str] = ContextVar("request_id", default="-")

_FORMAT = "%(asctime)s %(levelname)-7s [%(rid)s] %(name)s: %(message)s"
_OWNED = "_novelty_handler"  # marks handlers we installed, so reconfiguring replaces them


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.rid = request_id.get()
        return True


def _level(value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    name = (value or os.environ.get("NOVELTY_LOG_LEVEL") or "INFO").upper()
    level = logging.getLevelName(name)
    return level if isinstance(level, int) else logging.INFO


def configure_logging(level: str | int | None = None, log_file: str | None = None) -> logging.Logger:
    """Set up the ``novelty`` logger. Safe to call more than once (handlers are replaced)."""
    logger = logging.getLogger("novelty")
    logger.setLevel(_level(level))
    logger.propagate = False
    for h in [h for h in logger.handlers if getattr(h, _OWNED, False)]:
        logger.removeHandler(h)
        h.close()

    # A Windows console codepage cannot encode every character users type; never let a log
    # line with, say, Hindi text raise inside the logging machinery.
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except (ValueError, OSError):
            pass

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    path = log_file or os.environ.get("NOVELTY_LOG_FILE")
    if path:
        try:
            handlers.append(logging.FileHandler(path, encoding="utf-8"))
        except OSError as e:
            print(f"warning: cannot open log file {path!r} ({e}); logging to stderr only", file=sys.stderr)
    for h in handlers:
        h.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))
        h.addFilter(_RequestIdFilter())
        setattr(h, _OWNED, True)
        logger.addHandler(h)
    return logger


def preview(text: str, limit: int = 60) -> str:
    """One-line, length-limited rendering of user text for log messages."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."
