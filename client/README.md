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

Behind the AgenticWF server (Caddy, Logto) the commands sign in with `agenticwf login`; set the
connection once (an administrator gets the values with
`docker exec agenticwf cat /var/lib/agenticwf/logto/clients.env`):

```bash
export AGENTICWF_AUTH_URL=https://host.example.org
export AGENTICWF_API_URL=https://host.example.org/app
export AGENTICWF_CLIENT_ID=<Logto native app id>
export AGENTICWF_RESOURCE=<API resource indicator>
agenticwf login
```

Scripts and CI use a personal access token instead (`AGENTICWF_TOKEN=pat_…`), or a machine
client (`AGENTICWF_CLIENT_ID` + `AGENTICWF_CLIENT_SECRET`).

The personal agent (`agenticwf-agent serve`) has its own token instead:

```bash
export AGENTICWF_AGENT_URL=http://127.0.0.1:8765
export AGENTICWF_AGENT_TOKEN_FILE=~/.agenticwf-agent/token
```

From Python (`AgentClient.from_env()` finds the agent and its token as the command does):

```python
from agenticwf.agent.client import AgentClient

with AgentClient.from_env() as agent:
    sid = agent.create_session("demo")["id"]
    print(agent.send(sid, "hello")["reply"])
    print(agent.wait_task(agent.submit_task("summarise run 42")["task_id"])["result"])
```
