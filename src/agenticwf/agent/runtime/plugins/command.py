"""
Command tools: a system plugin's tool that runs a program, in the environment a setup script
leaves behind (e.g. `setupATLAS; lsetup rucio panda` on cvmfs), one process per call.

    plugin.yaml
      environment:                  system plugins only (operators write it; it runs a shell)
        setup: |                    sourced once in a clean bash; its environment is captured
          export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
          source $ATLAS_LOCAL_ROOT_BASE/user/atlasLocalSetup.sh --quiet
          lsetup -q rucio panda
        refresh: 3600               seconds before it is sourced again (new releases on cvmfs)
        pass: [RUCIO_ACCOUNT]       the agent's variables the setup and the tools may see; no others
        timeout: 120                seconds the setup may take

    tools/list_rules.md
      handler:
        kind: command
        argv: [python3, "${plugin}/bin/rucio_tool.py", list_rules]   ${plugin}: the plugin's directory;
                                                                     the program is looked up on the
                                                                     setup's PATH

The setup runs in `bash --noprofile --norc` started with `env -i` and a minimal environment, so
the agent's own secrets (model API keys, AGENT_SECRETS_KEY) never reach it or the tools. Aliases
and shell functions are not captured: name programs, not `setupATLAS` or `lsetup`, in `argv`. A
setup that fails is reported on the plugin and by its tools, and tried again after a minute.

A call: the tool's arguments as JSON on stdin; on stdout either a JSON object

    {"content": "…", "summary": "…", "facts": {…}}          or  {"error": "…", "code": "…"}

or plain text (the content). A non-zero exit is an error with the end of stderr. The process runs
in its own process group and the whole group is killed when the call's timeout (the tool's, or an
admin's on the Tools tab) is reached or the task is cancelled. Each call gets a private directory,
AGENT_CALL_DIR, removed afterwards, and AGENT_TASK_ID, AGENT_CALL_ID, AGENT_USER. A
CredentialProvider (none yet: docs/todo.md, item 2) may add files there for the task's user, a
token or a proxy, with the variable that points at each (BEARER_TOKEN_FILE, X509_USER_PROXY).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.tools.spec import CallContext, Handler, ToolResult, ToolSpec

LOG = get_logger(__name__)

BASE_PATH = "/usr/local/bin:/usr/bin:/bin"
DROPPED = frozenset({"_", "SHLVL", "PWD", "OLDPWD", "__status"})   # bash's own, not the setup's
RETRY_FAILED = 60.0                                               # seconds before a failed setup is tried again
STDERR_TAIL = 2000

# (the task's user, the plugin's name) -> {variable: file content}: written to the call's directory,
# the variable set to the file's path
CredentialProvider = Callable[[AuthContext, str], dict[str, bytes]]


class SetupError(Exception):
    pass


@dataclass(frozen=True)
class ShellSetup:
    """`environment:` in plugin.yaml."""

    script: str
    refresh: float = 3600.0
    passthrough: tuple[str, ...] = ()
    timeout: float = 120.0

    @classmethod
    def parse(cls, value: Any) -> "ShellSetup":
        if not isinstance(value, dict) or not isinstance(value.get("setup"), str) or not value["setup"].strip():
            raise ValueError("environment needs `setup`: the shell lines to source")
        passthrough = value.get("pass") or []
        if not isinstance(passthrough, list) or not all(isinstance(v, str) for v in passthrough):
            raise ValueError("environment.pass must be a list of variable names")
        try:
            refresh, timeout = float(value.get("refresh", 3600)), float(value.get("timeout", 120))
        except (TypeError, ValueError) as exc:
            raise ValueError("environment.refresh and timeout are seconds") from exc
        if refresh <= 0 or timeout <= 0:
            raise ValueError("environment.refresh and timeout must be positive")
        return cls(value["setup"].strip(), refresh, tuple(passthrough), timeout)


def clean_environment(passthrough: tuple[str, ...] = ()) -> dict[str, str]:
    """What a setup (or a plugin without one) starts from: nothing of the agent's but these."""
    home = os.environ.get("HOME") or tempfile.gettempdir()
    env = {"HOME": home, "PATH": BASE_PATH, "LANG": os.environ.get("LANG") or "C.UTF-8",
           "TMPDIR": tempfile.gettempdir()}
    env.update({k: os.environ[k] for k in passthrough if k in os.environ})
    return env


class ShellEnvironment:
    """A plugin's captured environment: its setup sourced once, again every `refresh` seconds."""

    def __init__(self, setup: ShellSetup | None, root: Path):
        self.setup = setup
        self.root = root
        self._env: dict[str, str] | None = None
        self._at = 0.0
        self.error = ""
        self._lock = threading.Lock()

    def environment(self) -> dict[str, str]:
        """The environment for a call; it blocks while the setup runs (call it in a thread).
        SetupError: the setup failed (now, or less than a minute ago)."""
        if self.setup is None:
            return clean_environment()
        with self._lock:
            now = time.monotonic()
            if self._env is not None and now - self._at < self.setup.refresh:
                return dict(self._env)
            if self.error and now - self._at < RETRY_FAILED:
                raise SetupError(self.error)
            try:
                self._env, self.error = capture(self.setup, self.root), ""
            except SetupError as exc:
                self._env, self.error = None, str(exc)
                LOG.error("plugin %s: environment setup failed: %s", self.root.name, exc)
                raise
            finally:
                self._at = time.monotonic()
            return dict(self._env)

    def status(self) -> dict[str, Any]:
        """For the admin page: whether the setup ran, when, and its error. Never the variables."""
        if self.setup is None:
            return {}
        with self._lock:
            state = "ok" if self._env is not None else "failed" if self.error else "not run yet"
            ago = round(time.monotonic() - self._at) if self._at else None
            return {"state": state, "seconds_ago": ago, "error": self.error, "refresh": self.setup.refresh}


def capture(setup: ShellSetup, root: Path) -> dict[str, str]:
    """Source the setup in a clean bash and return the environment it leaves. Its own output goes
    to stderr (kept for the error); the environment comes back on file descriptor 3."""
    script = ("exec 3>&1 1>&2\n{\n" + setup.script + "\n}\n__status=$?\n"
              "[ $__status -eq 0 ] || exit $__status\nenv -0 >&3\n")
    try:
        done = subprocess.run(["bash", "--noprofile", "--norc", "-c", script], env=clean_environment(setup.passthrough),
                              cwd=root, capture_output=True, timeout=setup.timeout, start_new_session=True)
    except subprocess.TimeoutExpired as exc:
        raise SetupError(f"the setup did not finish within {setup.timeout:g} s") from exc
    except OSError as exc:
        raise SetupError(f"cannot run bash: {exc}") from exc
    if done.returncode != 0:
        tail = done.stderr.decode(errors="replace").strip()[-STDERR_TAIL:]
        raise SetupError(f"the setup exited with {done.returncode}" + (f": {tail}" if tail else ""))
    env = {}
    for item in done.stdout.decode(errors="replace").split("\0"):
        key, sep, value = item.partition("=")
        if sep and key and key not in DROPPED:
            env[key] = value
    if not env.get("PATH"):
        raise SetupError("the setup left no PATH")
    return env


def command_handler(spec: ToolSpec, plugin: str, root: Path, shell: ShellEnvironment,
                    credentials: CredentialProvider | None = None) -> Handler:
    """The handler of a `kind: command` tool of a system plugin."""
    argv = [str(a).replace("${plugin}", str(root)) for a in spec.handler.settings.get("argv") or ()]

    async def handler(call: CallContext, arguments: dict) -> ToolResult:
        if not argv:
            return ToolResult.error("tool_unavailable", f"{spec.name} has no argv")
        try:
            env = await asyncio.to_thread(shell.environment)
        except SetupError as exc:
            return ToolResult.error("environment_unavailable", f"{spec.name}: the plugin's environment setup failed: {exc}")
        program = shutil.which(argv[0], path=env.get("PATH")) if "/" not in argv[0] else argv[0]
        if program is None:
            return ToolResult.error("tool_unavailable", f"{spec.name}: {argv[0]!r} is not on the setup's PATH")
        directory = tempfile.mkdtemp(prefix="agent-call-")   # 0700
        try:
            env.update(AGENT_CALL_DIR=directory, AGENT_TASK_ID=call.task_id, AGENT_CALL_ID=call.call_id,
                       AGENT_USER=call.auth.subject or "")
            if credentials is not None:
                for variable, content in credentials(call.auth, plugin).items():
                    path = Path(directory) / variable.lower()
                    path.write_bytes(content)
                    path.chmod(0o600)
                    env[variable] = str(path)
            return await _run([program, *argv[1:]], json.dumps(arguments).encode(), env, root, _within(call, spec.timeout),
                              spec.name)
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    return handler


def _within(call: CallContext, fallback: float) -> float:
    """The process's time: a little less than the call's, so it is killed here, not abandoned."""
    if not call.deadline:
        return fallback
    remaining = call.deadline - time.time()
    return max(1.0, remaining - min(5.0, remaining / 10))


async def _run(argv: list[str], stdin: bytes, env: dict[str, str], cwd: Path, timeout: float, name: str) -> ToolResult:
    try:
        process = await asyncio.create_subprocess_exec(*argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE, env=env, cwd=cwd, start_new_session=True)
    except OSError as exc:
        return ToolResult.error("tool_unavailable", f"{name}: cannot start {argv[0]}: {exc}")
    try:
        out, err = await asyncio.wait_for(process.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        return ToolResult.error("timeout", f"{name} did not finish within {timeout:g} s and was stopped")
    finally:   # timed out, or the call was cancelled: stop the program and everything it started
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
    text = out.decode(errors="replace").strip()
    if process.returncode != 0:
        tail = err.decode(errors="replace").strip()[-STDERR_TAIL:] or text[-STDERR_TAIL:]
        return ToolResult.error("command_failed", f"{name} exited with {process.returncode}" + (f": {tail}" if tail else ""))
    return _result(text, name)


def _result(text: str, name: str) -> ToolResult:
    try:
        data = json.loads(text) if text.startswith("{") else None
    except ValueError:
        data = None
    if not isinstance(data, dict):
        return ToolResult(content=text, summary=f"{name}: {len(text)} chars")
    if data.get("error"):
        return ToolResult.error(str(data.get("code") or "command_error"), str(data["error"]))
    content = data.get("content", "")
    facts = data.get("facts") if isinstance(data.get("facts"), dict) else {}
    return ToolResult(content=content if isinstance(content, str) else json.dumps(content, indent=1),
                      facts=facts, summary=str(data.get("summary") or f"{name}: done")[:200])
