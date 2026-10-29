#!/usr/bin/env bash
# supervisord program: a worker process beside the API (runs as user `agent`). The API process
# creates and migrates the schema; a worker waits until it is at this code's Alembic head (on an
# existing database the tables are there before new columns are, e.g. agent_jobs.trace).
set -euo pipefail

for _ in $(seq 1 120); do
    /opt/agent/venv/bin/python -c "from agenticwf.agent.core.orm.base.migrate import at_head; \
at_head() or exit(1)" 2>/dev/null && break
    sleep 2
done
exec /opt/agent/venv/bin/agenticwf-agent-worker
