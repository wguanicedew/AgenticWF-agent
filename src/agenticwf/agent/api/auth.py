"""AuthContext checks for agent endpoints.

authsvc's route policy (policy.yaml) is the first gate; these FastAPI dependencies repeat
the checks inside the process so a misconfigured policy cannot expose agent data on its own.
Data-level rules (which session or task a principal may touch) are in ``owner_key``.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from agenticwf.auth.client import current_auth_context, require_auth
from agenticwf.auth.context import AuthContext

__all__ = ["AuthContext", "current_auth_context", "require_auth", "require", "owner_key", "is_approved"]


def is_approved(ctx: AuthContext, request: Request) -> bool:
    return ctx.has_scopes(*request.app.state.settings.required_scopes)


def require(*scopes: str, allow_machine: bool = True, allow_pending: bool = False):
    """FastAPI dependency: an approved principal holding all ``scopes``.

    ``allow_machine=False`` rejects machine-to-machine (client credentials) tokens;
    ``allow_pending=True`` skips the account-approval scopes.
    """

    def dependency(request: Request, ctx: AuthContext = Depends(require_auth)) -> AuthContext:
        if not allow_pending and not is_approved(ctx, request):
            raise HTTPException(status_code=403, detail="approval_pending")
        if not allow_machine and ctx.is_machine:
            raise HTTPException(status_code=403, detail="machine tokens are not allowed here")
        missing = set(scopes) - ctx.scopes
        if missing:
            raise HTTPException(status_code=403, detail=f"missing scopes: {' '.join(sorted(missing))}")
        return ctx

    return dependency


def owner_key(ctx: AuthContext) -> tuple[str, str | None]:
    """Who owns the sessions and tasks this principal creates and may read.

    The organization is part of the key, so a user's organization-scoped token and their
    personal token see different data (tenant isolation).
    """
    return (ctx.subject, ctx.organization_id)
