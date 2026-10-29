#!/usr/bin/env bash
# Container health: PostgreSQL accepts connections and the agent API is up.
set -euo pipefail
# As `postgres`: Docker runs the check as root, and a role "root" does not exist (peer
# authentication maps the OS user to the role), so PostgreSQL would log a FATAL every time.
runuser -u postgres -- pg_isready -q -h /run/postgresql -d postgres
curl -fsS -o /dev/null "http://127.0.0.1:${PORT:-8000}/healthz"
