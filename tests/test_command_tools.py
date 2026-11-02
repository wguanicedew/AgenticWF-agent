"""Command tools of system plugins: a program run in the environment the plugin's setup leaves
behind (agenticwf.agent.runtime.plugins.command); no cvmfs here: a setup that puts a fake client on PATH."""

import asyncio
import os
import sys
import time

import pytest

from agenticwf.agent.runtime.plugins.command import ShellSetup, capture
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.tools.spec import CallContext, HandlerKind

from tests.test_runtime import user

CLIENT = f"""#!{sys.executable}
import json, os, sys, time
args = json.load(sys.stdin)
op = sys.argv[1]
if op == "echo":
    print(json.dumps({{"content": f"{{args['value']}} via {{os.environ['FAKE_CLIENT']}}", "summary": "echoed",
                      "facts": {{"files": [{{"path": "did:" + args['value']}}]}},
                      "env": sorted(os.environ)}}))
elif op == "env":
    print("\\n".join(f"{{k}}={{v}}" for k, v in sorted(os.environ.items())))
elif op == "token":
    path = os.environ["BEARER_TOKEN_FILE"]
    print(json.dumps({{"content": open(path).read() + " " + oct(os.stat(path).st_mode & 0o777) + " " + path}}))
elif op == "refuse":
    print(json.dumps({{"error": "no such dataset", "code": "not_found"}}))
elif op == "fail":
    print("traceback: something broke", file=sys.stderr)
    sys.exit(2)
elif op == "hang":
    time.sleep(60)
"""


def tool(name, op, timeout=10):
    return f"""---
name: {name}
summary: The {name} tool of the fake client.
input_schema:
  type: object
  properties:
    value: {{type: string}}
timeout: {timeout}
handler:
  kind: command
  argv: [fake_client, {op}]
---
Runs fake_client {op}.
"""


def make_plugin(tmp_path, setup=None):
    root = tmp_path / "plugins" / "atlas"
    (root / "tools").mkdir(parents=True)
    (root / "bin").mkdir()
    client = root / "bin" / "fake_client"
    client.write_text(CLIENT)
    client.chmod(0o755)
    setup = setup or 'export PATH="$PWD/bin:$PATH"\nexport FAKE_CLIENT=cvmfs-1.0\necho "Setting up the client" '
    (root / "plugin.yaml").write_text("environment:\n  setup: |\n" + "".join(f"    {line}\n" for line in setup.splitlines())
                                      + "  refresh: 3600\n  pass: [RUCIO_ACCOUNT]\n")
    for name, op, timeout in (("echo", "echo", 10), ("env", "env", 10), ("token", "token", 10),
                              ("refuse", "refuse", 10), ("fail", "fail", 10), ("hang", "hang", 2)):
        (root / "tools" / f"{name}.md").write_text(tool(name, op, timeout))
    return root.parent


def call_tool(registry, name, arguments=None, seconds=10):
    spec = registry.lookup(name, ("user_a", None))
    context = CallContext(task_id="t1", call_id="c1", auth=user(), deadline=time.time() + seconds)
    return asyncio.run(registry.handler(spec)(context, arguments or {"value": "x"}))


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SECRETS_KEY", "never-to-be-seen")
    monkeypatch.setenv("RUCIO_ACCOUNT", "wguan")
    plugins = PluginManager([make_plugin(tmp_path)])
    registry = ToolRegistry(plugins=plugins)
    assert registry.load() == [] and plugins.load() == []
    return registry


def test_a_command_runs_in_the_setups_environment(registry):
    assert registry.lookup("atlas__echo", ("user_a", None)).handler.kind is HandlerKind.COMMAND
    result = call_tool(registry, "atlas__echo", {"value": "data18"})
    assert not result.is_error and result.content == "data18 via cvmfs-1.0" and result.summary == "echoed"
    assert result.facts == {"files": [{"path": "did:data18"}]}
    env = dict(line.split("=", 1) for line in call_tool(registry, "atlas__env").content.splitlines())
    assert env["FAKE_CLIENT"] == "cvmfs-1.0" and env["RUCIO_ACCOUNT"] == "wguan"   # passed on purpose
    assert "AGENT_SECRETS_KEY" not in env                                      # the agent's own: never
    assert env["AGENT_TASK_ID"] == "t1" and env["AGENT_USER"] == "user_a" and not os.path.exists(env["AGENT_CALL_DIR"])
    status = registry.plugins.get("atlas").to_dict()["environment"]
    assert status["state"] == "ok" and status["error"] == ""


def test_errors_reach_the_model(registry):
    refused = call_tool(registry, "atlas__refuse")
    assert refused.is_error and refused.error_code == "not_found" and refused.content == "no such dataset"
    failed = call_tool(registry, "atlas__fail")
    assert failed.is_error and failed.error_code == "command_failed" and "something broke" in failed.content


def test_a_call_past_its_time_is_stopped(registry):
    started = time.monotonic()
    result = call_tool(registry, "atlas__hang", seconds=2)
    assert result.is_error and result.error_code == "timeout" and time.monotonic() - started < 5


def test_the_users_credentials_go_to_their_call_only(tmp_path):
    plugins = PluginManager([make_plugin(tmp_path)], credentials=lambda auth, plugin: {
        "BEARER_TOKEN_FILE": f"token-of-{auth.subject}-for-{plugin}".encode()})
    registry = ToolRegistry(plugins=plugins)
    registry.load()
    content, mode, path = call_tool(registry, "atlas__token").content.split(" ")
    assert content == "token-of-user_a-for-atlas" and mode == "0o600" and not os.path.exists(path)


def test_a_failing_setup_is_reported(tmp_path):
    plugins = PluginManager([make_plugin(tmp_path, setup="echo cvmfs is not mounted >&2\nfalse")])
    registry = ToolRegistry(plugins=plugins)
    registry.load()
    result = call_tool(registry, "atlas__echo")
    assert result.is_error and result.error_code == "environment_unavailable" and "cvmfs is not mounted" in result.content
    status = plugins.get("atlas").to_dict()["environment"]
    assert status["state"] == "failed" and "exited with 1" in status["error"]


def test_capture_drops_bashs_own_variables(tmp_path):
    env = capture(ShellSetup("export ONE=1\nfunction helper { :; }"), tmp_path)
    assert env["ONE"] == "1" and "SHLVL" not in env and "__status" not in env and env["PATH"]
    with pytest.raises(ValueError):
        ShellSetup.parse({"refresh": 10})


def test_users_plugins_cannot_run_commands(tmp_path):
    sandbox = tmp_path / "sandbox"
    make_plugin(sandbox)
    plugins = PluginManager(user_sandbox=lambda owner: sandbox)
    registry = ToolRegistry(plugins=plugins)
    registry.load()
    assert registry.lookup("atlas__echo", ("user_a", None)) is None
    errors = plugins.get("atlas", ("user_a", None)).errors
    assert any("environment ignored" in e for e in errors)
    assert any("command" in e for e in errors if "echo.md" in e)
