# agenticwf-agent-client

The client of the AgenticWF agent: the Python `AgentClient` and the `agenticwf agent` command
(with [agenticwf-auth-client](https://pypi.org/project/agenticwf-auth-client/), installed with it).

```bash
pip install agenticwf-agent-client
agenticwf agent chat "hello"                      # a new chat; prints the reply
agenticwf agent chat "and then?" --session SID
agenticwf agent task "summarise run 42" --wait    # a background task
agenticwf agent approve CALL --session SID        # a tool call that waits for you
```

Only the agent's URL is needed: the agent tells how to sign in, and behind the AgenticWF server
(Caddy, Logto) the server publishes the Logto settings. Sign in once with `agenticwf login`:

```bash
export AGENTICWF_AGENT_URL=https://host.example.org/app
agenticwf login
```

Scripts and CI use a personal access token instead (`AGENTICWF_TOKEN=pat_…`), or a machine
client (`AGENTICWF_CLIENT_ID` + `AGENTICWF_CLIENT_SECRET`). `AGENTICWF_AUTH_URL`,
`AGENTICWF_API_URL`, `AGENTICWF_CLIENT_ID` and `AGENTICWF_RESOURCE`, when set, take precedence
over what the server publishes (an administrator gets them with
`docker exec agenticwf cat /var/lib/agenticwf/logto/clients.env`).

The personal agent (`agenticwf-agent serve`) has its own token instead. On the same machine the
URL is enough (the token is read from `~/.agenticwf-agent/token`, or `$AGENTICWF_AGENT_HOME`);
otherwise use the sign-in link it printed as the URL, or give the token:

```bash
export AGENTICWF_AGENT_URL=http://127.0.0.1:8765
export AGENTICWF_AGENT_URL='http://127.0.0.1:8765/?token=…'   # or: the sign-in link
export AGENTICWF_AGENT_TOKEN=…                                # or: the token (or _TOKEN_FILE)
```

From Python (`AgentClient.from_env()` finds the agent and its token as the command does):

```python
from agenticwf.agent.client import AgentClient

with AgentClient.from_env() as agent:
    sid = agent.create_session("demo")["id"]
    print(agent.send(sid, "hello")["reply"])
    print(agent.wait_task(agent.submit_task("summarise run 42")["task_id"])["result"])
```
