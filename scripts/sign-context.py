#!/usr/bin/env python3
"""Print curl header options carrying a signed AuthContext, for local development without
Caddy and authsvc. The signature uses the same code as authsvc's /verify.

    export AUTH_CONTEXT_SECRET=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')
    uvicorn --factory agenticwf.agent.api.app:create_app --port 8000 &
    curl $(scripts/sign-context.py POST /api/v1/sessions) -H 'Content-Type: application/json' \
         -d '{}' http://127.0.0.1:8000/api/v1/sessions

The context is bound to METHOD and PATH (no query string) and valid for --ttl seconds.
"""

import argparse
import time

from agenticwf.auth.config import read_secret
from agenticwf.auth.context import AuthContext, encode_context


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("method")
    p.add_argument("path")
    p.add_argument("--subject", default="user_dev")
    p.add_argument("--org", default=None, help="organization id")
    p.add_argument("--scopes", default="app:access agent:use")
    p.add_argument("--machine", action="store_true", help="machine-to-machine token")
    p.add_argument("--ttl", type=int, default=60)
    args = p.parse_args()

    secret = read_secret("AUTH_CONTEXT_SECRET")
    if not secret:
        p.error("set AUTH_CONTEXT_SECRET or AUTH_CONTEXT_SECRET_FILE")
    ctx = AuthContext(
        subject=args.subject,
        user_id=None if args.machine else args.subject,
        client_id=args.subject if args.machine else "dev",
        organization_id=args.org,
        scopes=frozenset(args.scopes.split()),
        expires_at=int(time.time()) + 3600,
    )
    payload, sig = encode_context(ctx, secret.encode(), args.ttl, args.method, args.path)
    print(f"-H X-Auth-Context:{payload} -H X-Auth-Context-Sig:{sig}")


if __name__ == "__main__":
    main()
