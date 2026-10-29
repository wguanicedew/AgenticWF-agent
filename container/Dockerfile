# syntax=docker/dockerfile:1
#
# AgenticWF agent container (AlmaLinux 9), next to the agenticwf-auth container
#
#   supervisord
#   ├── postgres   Unix socket /run/postgresql only (no TCP), database `agent`
#   └── agent      uvicorn agenticwf.agent.api.app:create_app on ${LISTEN_HOST}:8000 (Caddy's upstream),
#                  verifies authsvc's signed X-Auth-Context
#                  or, with AGENT_AUTH_MODE=local, the personal agent (agenticwf-agent serve):
#                  no Caddy or authsvc; a personal access token (docs/local.md)
#
# The `agenticwf-auth-server` package (AuthContext verification) comes from the AgenticWF-auth repository
# through a named build context, so this repository does not vendor it. Build from the
# repository root with either a local checkout or a git ref:
#
#   docker build --network=host -f container/Dockerfile \
#     --build-context authsvc=../AgenticWF-auth/authsvc -t agenticwf-agent .
#
#   docker build --network=host -f container/Dockerfile \
#     --build-context authsvc='https://github.com/InnovateAILab/AgenticWF-auth.git#v0.1.0:authsvc' \
#     -t agenticwf-agent .
#
# Use the same AgenticWF-auth version as the running agenticwf-auth image.

FROM almalinux:9

ARG POSTGRES_STREAM=16
ARG PYTHON=python3.12

LABEL org.opencontainers.image.title="agenticwf-agent" \
      org.opencontainers.image.description="AgenticWF agent API + PostgreSQL on AlmaLinux 9, behind Caddy and the AgenticWF auth service" \
      org.opencontainers.image.licenses="Apache-2.0"

# Base packages: PostgreSQL from its AppStream module stream, supervisor from EPEL.
RUN dnf -y install epel-release \
 && dnf -y module enable "postgresql:${POSTGRES_STREAM}" \
 && dnf -y install --setopt=install_weak_deps=False \
        postgresql-server postgresql \
        "${PYTHON}" "${PYTHON}-pip" \
        supervisor \
        procps-ng \
 && dnf clean all && rm -rf /var/cache/dnf

# The agent runs as uid 1000: the host-side copy of the AuthContext secret must be
# readable by it. PostgreSQL peer authentication maps this OS user to the `agent` role.
RUN useradd --uid 1000 --user-group --create-home --home-dir /home/agent --shell /sbin/nologin agent

# Agent and authsvc in one virtualenv; code stays root-owned.
COPY --from=authsvc . /opt/agent/src/authsvc
COPY pyproject.toml /opt/agent/src/agent/
COPY src /opt/agent/src/agent/src
RUN "${PYTHON}" -m venv /opt/agent/venv \
 && /opt/agent/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/agent/venv/bin/pip install --no-cache-dir /opt/agent/src/authsvc \
 && /opt/agent/venv/bin/pip install --no-cache-dir "/opt/agent/src/agent[llm,otel]"

COPY container/supervisord.conf /etc/supervisord.conf
COPY container/supervisord.d/ /etc/supervisord.d/
COPY container/config/agent.cfg /etc/agenticwf/agent.cfg
# System task types and tools of the runtime ([context] system_dir): the packaged defaults.
COPY src/agenticwf/agent/runtime/defaults/ /etc/agenticwf/agent/
RUN mkdir -p /etc/agenticwf/agent/tools
COPY container/scripts/ /usr/local/bin/
RUN chmod 0755 /usr/local/bin/*.sh

# AGENT_AUTH_MODE (authsvc | local) is not set here: the entrypoint defaults it to authsvc.
ENV DATA_ROOT=/var/lib/agenticwf-agent \
    PGDATA=/var/lib/agenticwf-agent/pgdata \
    AGENTICWF_CONFIG=/etc/agenticwf/agent.cfg \
    AUTH_CONTEXT_SECRET_FILE=/run/secrets/auth_context_secret \
    AGENTICWF_AGENT_HOME=/var/lib/agenticwf-agent/local \
    AGENT_CONTAINER=1 \
    PATH=/opt/agent/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

# Persist the database.
VOLUME ["/var/lib/agenticwf-agent"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD /usr/local/bin/healthcheck.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisord.conf"]
