# AgenticWF-agent

The agent backend of [AgenticWF](https://github.com/wguanicedew/AgenticWF). It runs in its
own container behind Caddy and the AgenticWF auth service
([AgenticWF-auth](https://github.com/wguanicedew/AgenticWF-auth)), and serves the chat page,
the REST API and live updates on `127.0.0.1:8000`. Like the auth container, the image is
AlmaLinux 9 with supervisord running PostgreSQL (the agent's database) and the agent API.

```
CLI / scripts ── Bearer ──┐
Browser chat  ── cookie ──┼─► Caddy ─► authsvc /verify ─► agenticwf-agent :8000
Live updates  ── cookie ──┘   (host)   (agenticwf-auth)   verifies the signed AuthContext
```

authsvc authenticates every request and checks the route policy. It then signs an
`AuthContext` (who the caller is, their scopes, roles and organization) that the agent verifies
with a shared secret. The agent uses it to check scopes again and to keep each user's sessions
and tasks separate.

It also runs on its own as a **personal agent** on your machine, with token authentication
built in instead of Caddy and authsvc, either directly with SQLite or in the same image with
its PostgreSQL:

```bash
pip install ../AgenticWF-auth/authsvc && pip install -e .
agenticwf-agent serve --open          # http://localhost:8765/?token=…

docker run -d -e AGENT_AUTH_MODE=local -p 127.0.0.1:8765:8000 \
  -v agenticwf-agent-local:/var/lib/agenticwf-agent agenticwf-agent
docker logs <container> | grep open:  # the sign-in link
```

Two pip packages, built into `dist/` by `scripts/build-packages.sh [server|client]`:

| Package | Source | Contains |
|---|---|---|
| `agenticwf-agent-server` | the repository root | the service (`agenticwf.agent.api`, `core`, `runtime`), `agenticwf-agent`, `agenticwf-agent-worker`; what the container installs |
| `agenticwf-agent-client` | [client/](client) | `AgentClient` (`agenticwf.agent.client`) and the `agenticwf agent` command |

The client needs only httpx. `agenticwf agent` is a subcommand of `agenticwf` (agenticwf-auth-client,
AgenticWF-auth `cli/`), which also gets the tokens:

```bash
pip install ../AgenticWF-auth/cli ./client
agenticwf login && agenticwf agent chat "hello"          # behind Caddy (AGENTICWF_API_URL, …)
export AGENTICWF_AGENT_URL=http://127.0.0.1:8765 AGENTICWF_AGENT_TOKEN_FILE=~/.agenticwf-agent/token
agenticwf agent task "summarise run 42" --wait           # the personal agent, with its own token
```

| Path | Content |
|---|---|
| [src/agenticwf/agent/api/](src/agenticwf/agent/api) | FastAPI application (`agenticwf.agent.api.app:create_app`), the chat page, the personal agent (`agenticwf.agent.api.local`) |
| [src/agenticwf/agent/runtime/](src/agenticwf/agent/runtime) | The agent runtime: threads, harness, tools, policy, context; default task type in `runtime/defaults` |
| [src/agenticwf/agent/core/](src/agenticwf/agent/core) | Shared code: configuration, logging, exceptions, ORM base and tables |
| [client/](client) | The `agenticwf-agent-client` package: `AgentClient` and `agenticwf agent` |
| [examples/](examples) | Example `AGENTS.md`, `WORKFLOW.md`, task types and tool definitions |
| [container/](container) | AlmaLinux 9 Dockerfile, supervisord config, scripts, default `agent.cfg`, env file example |
| [deploy/policy.agent.yaml](deploy/policy.agent.yaml) | authsvc route rules for the agent's endpoints |
| [scripts/sign-context.py](scripts/sign-context.py) | Sign AuthContext headers for local testing |
| [.github/workflows/build-container.yml](.github/workflows/build-container.yml) | Tests, then builds and pushes the image on every new tag |

Quick start (development):

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install ../AgenticWF-auth/authsvc && pip install -e '.[test]' -e client
pytest -q                     # SQLite; AGENT_TEST_DB_URL=postgresql+psycopg2://… for PostgreSQL
```
