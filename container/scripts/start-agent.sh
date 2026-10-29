#!/usr/bin/env bash
# supervisord program: AgenticWF agent API (runs as user `agent`).
set -euo pipefail

for _ in $(seq 1 60); do
    pg_isready -q -h /run/postgresql -d "${AGENT_DB_NAME:-agent}" && break
    sleep 1
done

# Create missing tables and bring Alembic to head; safe on every start. Retried with backoff
# (about 5 minutes): an external database that is briefly down at boot must not exhaust
# supervisord's start retries.
delay=2
for attempt in $(seq 1 12); do
    /opt/agent/venv/bin/python -m agenticwf.agent.api.db init && break
    if [[ $attempt -eq 12 ]]; then
        echo "[start-agent] the database is still unavailable; giving up" >&2
        exit 1
    fi
    echo "[start-agent] database not ready (attempt $attempt); retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$(( delay < 30 ? delay * 2 : 30 ))
done

if [[ "${AGENT_AUTH_MODE:-authsvc}" == local ]]; then
    # The personal agent (docs/local.md): the same API behind its own token front, with this
    # container's PostgreSQL and agent.cfg. The access link is printed to the logs.
    exec /opt/agent/venv/bin/agenticwf-agent --home "$AGENTICWF_AGENT_HOME" serve \
        --config "$AGENTICWF_CONFIG" \
        --host "$LISTEN_HOST" --port "$PORT" \
        --public-url "${AGENT_PUBLIC_URL:-http://localhost:8765}" \
        --log-level "$LOG_LEVEL"
fi

# One worker: live updates are fanned out in process (src/agenticwf/agent/api/events.py), and background tasks
# run in the API process. Client addresses come from Caddy, the only peer.
exec /opt/agent/venv/bin/uvicorn --factory agenticwf.agent.api.app:create_app \
    --host "$LISTEN_HOST" --port "$PORT" \
    --workers 1 \
    --log-level "$LOG_LEVEL" \
    --no-server-header \
    --no-proxy-headers \
    --timeout-graceful-shutdown 5
