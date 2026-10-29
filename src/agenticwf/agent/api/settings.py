"""Agent API settings, read from environment variables.

The variables that concern the auth integration use the same names as authsvc in the
agenticwf-auth container, so one secret file serves both.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from agenticwf.auth.config import read_secret

TRUE = {"1", "true", "yes", "on"}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # HMAC key shared with authsvc; verifies X-Auth-Context / X-Auth-Context-Sig.
    context_secret: bytes
    # Require that a signed context was issued for this exact method and path. Turn off
    # only if a proxy rewrites paths between authsvc's /verify and this service.
    bind_request: bool = True
    # Mirror of policy.yaml `required_scopes` (account approval), checked again here as
    # defence in depth. Empty disables the in-process check.
    required_scopes: frozenset[str] = frozenset({"app:access"})
    # Upper bound for one Server-Sent Events connection (s); streams also end when the
    # token behind the AuthContext expires, so clients reconnect through /verify.
    max_stream_seconds: int = 3600
    # How long POST /sessions/{sid}/messages waits for the reply (s) before answering
    # "running"; the reply still arrives as a live update and in the transcript.
    reply_timeout: float = 120.0
    # Where the trace view links a trace id, with `{trace_id}` in it, e.g.
    # http://localhost:16686/trace/{trace_id} (Jaeger). Empty: trace ids without links.
    trace_url: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        secret = read_secret("AUTH_CONTEXT_SECRET")
        if not secret:
            raise RuntimeError("AUTH_CONTEXT_SECRET or AUTH_CONTEXT_SECRET_FILE is required")
        if len(secret) < 32:
            raise RuntimeError("AUTH_CONTEXT_SECRET must be at least 32 characters")
        return cls(
            context_secret=secret.encode(),
            bind_request=_env("AUTH_BIND_REQUEST", "true").lower() in TRUE,
            required_scopes=frozenset(_env("AGENT_REQUIRED_SCOPES", "app:access").replace(",", " ").split()),
            max_stream_seconds=int(_env("AGENT_MAX_STREAM_SECONDS", "3600")),
            reply_timeout=float(_env("AGENT_REPLY_TIMEOUT", "120")),
            trace_url=_env("AGENT_TRACE_URL", ""),
        )
