"""
Logging support: a shared `agenticwf` logger tree plus automatic
contextual-field propagation, via `get_logger()` wrapping stdlib
`logging`, using a `ContextVar`-backed context object merged into every
record via a `logging.Filter`.

Configuration follows this codebase's own convention (`common.config`'s
ini-style `config_get`/`config_has_option`), not pydantic settings or
env-var-only toggles, to stay consistent with `orm.base.session`.
"""

import logging
import logging.handlers
from contextvars import ContextVar

from agenticwf.agent.core.common.config import config_get, config_has_option

LOG_SECTION = 'logging'
ROOT_LOGGER_NAME = 'agenticwf'

_configured = False
_log_context = ContextVar('agenticwf_log_context', default=None)


def _config_option(option, default):
    if config_has_option(LOG_SECTION, option):
        return config_get(LOG_SECTION, option)
    return default


class LogContextFilter(logging.Filter):
    """Merges the current `log_context()` fields onto every record that passes through it."""

    def filter(self, record):
        context = _log_context.get()
        if context:
            for key, value in context.items():
                setattr(record, key, value)
        record.context = context or {}
        return True


class ContextFormatter(logging.Formatter):
    """Plain-text formatter that appends any contextual fields as key=value pairs."""

    def format(self, record):
        base = super().format(record)
        if getattr(record, 'context', None):
            extra = ' '.join('%s=%s' % (k, v) for k, v in record.context.items())
            return '%s [%s]' % (base, extra)
        return base


def setup_logging():
    """
    Configure the shared `agenticwf` root logger once (idempotent).
    Reads from the "logging" config section:
      - level: e.g. "DEBUG"/"INFO"/"WARNING" (default "INFO")
      - file: optional path for a rotating file handler (10MB x 3 backups)
    Always logs to the console; a file handler is added only if configured.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = _config_option('level', 'INFO').upper()
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    context_filter = LogContextFilter()
    formatter = ContextFormatter('%(asctime)s %(levelname)s [%(name)s] %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    logger.addHandler(console_handler)

    log_file = _config_option('file', None)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(context_filter)
        logger.addHandler(file_handler)


def get_logger(name):
    """
    Return a logger scoped under the shared `agenticwf` root, configuring
    it on first use. Accepts either a short name ("telemetry") or a full
    module `__name__` (e.g. "agenticwf.runtime.execution_engine") --
    an "agenticwf." prefix on `name` is stripped so callers can always
    just pass `__name__`, the normal Python convention, without doubling up.
    """
    setup_logging()
    prefix = ROOT_LOGGER_NAME + '.'
    if name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(prefix):
        name = name[len(prefix):]
    return logging.getLogger(ROOT_LOGGER_NAME).getChild(name)


def set_log_context(**fields):
    """Merge fields into the current context; every subsequent log record on this thread carries them."""
    current = dict(_log_context.get() or {})
    current.update(fields)
    _log_context.set(current)
    return current


def get_log_context():
    """Return a copy of the current context fields."""
    return dict(_log_context.get() or {})


def clear_log_context():
    """Reset the current context to empty."""
    _log_context.set(None)


class log_context:
    """Context manager: merge fields in on entry, restore the prior context on exit."""

    def __init__(self, **fields):
        self.fields = fields
        self._token = None

    def __enter__(self):
        previous = dict(_log_context.get() or {})
        merged = dict(previous)
        merged.update(self.fields)
        self._token = _log_context.set(merged)
        return merged

    def __exit__(self, exc_type, exc_value, traceback):
        _log_context.reset(self._token)
