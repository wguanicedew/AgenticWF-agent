#!/usr/bin/env bash
# Container entrypoint: prepare persistent state, then hand over to supervisord.
#
#  1. create /var/lib/agenticwf-agent/{pgdata,secrets}
#  2. authsvc mode: check the AuthContext secret shared with authsvc (or store it from the
#     environment). Local mode: prepare the personal agent's token directory instead
#  3. initialise PostgreSQL on first boot; ensure the `agent` role + database
#  4. exec supervisord (postgres, agent)
#
# AGENT_AUTH_MODE selects who authenticates requests (docs/deployment.md, docs/local.md):
#   authsvc  (default) behind Caddy + authsvc, which sign every request's AuthContext
#   local    a personal agent: no Caddy, no authsvc; a personal access token instead
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/var/lib/agenticwf-agent}
PGDATA=${PGDATA:-$DATA_ROOT/pgdata}
SECRETS_DIR=$DATA_ROOT/secrets
AGENT_DB_NAME=${AGENT_DB_NAME:-agent}
AGENT_AUTH_MODE=${AGENT_AUTH_MODE:-authsvc}

log() { echo "[entrypoint] $*" >&2; }

case "$AGENT_AUTH_MODE" in
    authsvc|local) ;;
    *) log "ERROR: AGENT_AUTH_MODE must be 'authsvc' or 'local', not '$AGENT_AUTH_MODE'."; exit 1 ;;
esac

install -d -m 0755 "$DATA_ROOT"
# 0711: the agent can open its own secret file but cannot list the directory.
install -d -m 0711 -o root -g root "$SECRETS_DIR"
install -d -m 0775 -o postgres -g postgres /run/postgresql
# Users' own task types and plugins ([context] sandbox_root), written by the agent.
install -d -m 0750 -o agent -g agent "$DATA_ROOT/sandboxes"
# Optional log file ([logging] file in agent.cfg), written by the agent.
install -d -m 0750 -o agent -g agent "$DATA_ROOT/logs"

# Key that encrypts secrets the agent stores in its database (model API keys managers add,
# docs/agent-models.md). Created once; it never enters the database. Back it up with the volume:
# without it, stored API keys cannot be decrypted.
export AGENT_SECRETS_KEY_FILE=${AGENT_SECRETS_KEY_FILE:-$SECRETS_DIR/secrets_key}
if [[ ! -s "$AGENT_SECRETS_KEY_FILE" ]]; then
    /opt/agent/venv/bin/python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' \
        > "$AGENT_SECRETS_KEY_FILE"
    chown agent:agent "$AGENT_SECRETS_KEY_FILE"
    chmod 0400 "$AGENT_SECRETS_KEY_FILE"
    log "created the secrets key $AGENT_SECRETS_KEY_FILE"
    if [[ -s "$PGDATA/PG_VERSION" || -n "${AGENT_RESTORED_DATABASE:-}" ]]; then
        log "WARNING: a NEW secrets key was created for an existing database: API keys and MCP headers"
        log "WARNING: stored before cannot be decrypted with it. Restore the old key into"
        log "WARNING: $AGENT_SECRETS_KEY_FILE (docs/deployment.md, Backup) and restart, or enter them again."
    fi
fi

if [[ "$AGENT_AUTH_MODE" == local ]]; then
    # The personal agent keeps its access token and signing key on the volume, readable by
    # the agent only; `agenticwf-agent serve` creates them on first start.
    AGENTICWF_AGENT_HOME=${AGENTICWF_AGENT_HOME:-$DATA_ROOT/local}
    install -d -m 0700 -o agent -g agent "$AGENTICWF_AGENT_HOME"
    export AGENTICWF_AGENT_HOME
    unset AUTH_CONTEXT_SECRET
else
    # AuthContext secret: the same HMAC key as authsvc in the agenticwf-auth container.
    # Preferred: a read-only file mount at $AUTH_CONTEXT_SECRET_FILE. If AUTH_CONTEXT_SECRET
    # is set instead, keep it in a file so it does not stay in the services' environment.
    if [[ -n "${AUTH_CONTEXT_SECRET:-}" ]]; then
        printf '%s' "$AUTH_CONTEXT_SECRET" > "$SECRETS_DIR/auth_context_secret"
        chown agent:agent "$SECRETS_DIR/auth_context_secret"
        chmod 0400 "$SECRETS_DIR/auth_context_secret"
        unset AUTH_CONTEXT_SECRET
        export AUTH_CONTEXT_SECRET_FILE=$SECRETS_DIR/auth_context_secret
    fi
    export AUTH_CONTEXT_SECRET_FILE=${AUTH_CONTEXT_SECRET_FILE:-/run/secrets/auth_context_secret}
    if ! runuser -u agent -- test -r "$AUTH_CONTEXT_SECRET_FILE"; then
        log "ERROR: $AUTH_CONTEXT_SECRET_FILE is missing or not readable by uid $(id -u agent)."
        log "Copy authsvc's secret from the agenticwf-auth container and mount it read-only, see docs/deployment.md §2."
        log "To run the agent on its own instead (no Caddy, no authsvc), set AGENT_AUTH_MODE=local, see docs/local.md."
        exit 1
    fi
fi

if [[ ! -s "$PGDATA/PG_VERSION" ]]; then
    log "initialising PostgreSQL cluster in $PGDATA"
    install -d -m 0700 -o postgres -g postgres "$PGDATA"
    runuser -u postgres -- initdb -D "$PGDATA" \
        --encoding=UTF8 --locale=C.UTF-8 \
        --auth-local=peer --auth-host=reject >/dev/null
    cat >> "$PGDATA/postgresql.conf" <<EOF

# --- agenticwf-agent ---
# Unix socket only: no TCP port that could clash with the auth container's PostgreSQL
# under host networking, and nothing reachable from outside the container.
listen_addresses = ''
unix_socket_directories = '/run/postgresql'
logging_collector = off
EOF
fi

# On every boot, make sure the role and database exist. The agent connects over the Unix
# socket with peer authentication (OS user `agent` = role `agent`), so there is no password.
runuser -u postgres -- pg_ctl -D "$PGDATA" -w start >/dev/null
trap 'runuser -u postgres -- pg_ctl -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true' EXIT
runuser -u postgres -- psql -v ON_ERROR_STOP=1 -q -X -v db="$AGENT_DB_NAME" <<'SQL'
SELECT 'CREATE ROLE agent LOGIN' WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent') \gexec
ALTER ROLE agent NOSUPERUSER NOCREATEDB NOCREATEROLE;
SELECT format('CREATE DATABASE %I OWNER agent ENCODING %L', :'db', 'UTF8')
 WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = :'db') \gexec
SELECT format('REVOKE ALL ON DATABASE %I FROM PUBLIC', :'db') \gexec
SQL
runuser -u postgres -- pg_ctl -D "$PGDATA" -m fast -w stop >/dev/null
trap - EXIT
log "PostgreSQL ready"

# Values consumed by supervisord's %(ENV_...)s expansions and the programs.
export PGDATA DATA_ROOT AGENT_AUTH_MODE
if [[ "$AGENT_AUTH_MODE" == authsvc ]]; then
    export AUTH_CONTEXT_SECRET_FILE
fi
# Address the agent listens on. authsvc mode: 127.0.0.1 for `docker run --network host` (keeps
# it off the host's public interfaces), 0.0.0.0 for bridge networking with -p 127.0.0.1:8000:8000.
# Local mode: 0.0.0.0, published by `-p 127.0.0.1:8765:8000` (the host side decides who reaches it).
if [[ "$AGENT_AUTH_MODE" == local ]]; then
    export LISTEN_HOST=${LISTEN_HOST:-0.0.0.0}
else
    export LISTEN_HOST=${LISTEN_HOST:-127.0.0.1}
fi
export PORT=${PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-info}

# Worker processes beside the API process (docs/deployment.md, Worker processes): 0 (default) runs
# every job in the API process. More: they share the jobs through the database, and the API
# process and the workers all switch to [harness] shared.
AGENT_WORKER_PROCESSES=${AGENT_WORKER_PROCESSES:-0}
if ! [[ "$AGENT_WORKER_PROCESSES" =~ ^[0-9]+$ ]]; then
    log "ERROR: AGENT_WORKER_PROCESSES must be a number, not '$AGENT_WORKER_PROCESSES'."; exit 1
fi
if (( AGENT_WORKER_PROCESSES > 0 )); then
    if [[ "$AGENT_AUTH_MODE" == local ]]; then
        log "ERROR: worker processes are for the shared deployment, not the personal agent (AGENT_AUTH_MODE=local)."; exit 1
    fi
    export AGENT_HARNESS_SHARED=true AGENT_WORKERS_AUTOSTART=true AGENT_WORKER_PROCESSES_SLOTS=$AGENT_WORKER_PROCESSES
else
    export AGENT_WORKERS_AUTOSTART=false AGENT_WORKER_PROCESSES_SLOTS=1   # supervisord needs numprocs >= 1
fi

exec "$@"
