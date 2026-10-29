"""
Secrets in text the runtime passes on: tool results going to the model, notes going to long-term
memory.

TODO: a real secret scanner; these only catch the obvious cases.
"""

from __future__ import annotations

import re

SECRET_PATTERNS = [
    re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/-]{20,}=*"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)((?:api[_-]?key|secret|password|token)\s*[:=]\s*)\S+"),
]


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
    return text


def contains_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)
