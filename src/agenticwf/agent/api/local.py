"""The personal agent: the same service on localhost, with its own authentication (docs/local.md).

    agenticwf-agent serve            http://127.0.0.1:8765/?token=…   (opens once; then a cookie)
    agenticwf-agent token            print the access token (for curl / scripts: Bearer)

With AGENT_LOCAL_HINT (a phrase you chose, at least 8 characters), the sign-in page also shows the
token to whoever gives it; wrong answers are throttled (after 5 in a row, one try a minute).

Two ways to run it: directly (SQLite, everything in the home directory), or in the container
with AGENT_AUTH_MODE=local (the container's PostgreSQL and agent.cfg, `serve --config`; the
home directory then only holds the token and the key).

In a deployment, Caddy and authsvc authenticate every request and sign the AuthContext the
agent trusts. Here `LocalAuth` stands in for both: it checks the personal access token (a
Bearer header, or an HttpOnly cookie set from the link printed at start), rejects cross-origin
writes, and signs the AuthContext for the local user with a secret only this process knows. The
application behind it is the unchanged API, with the same AuthContextMiddleware.

Everything lives in one directory (default ~/.agenticwf-agent, or $AGENTICWF_AGENT_HOME):

    agent.cfg     core configuration: SQLite database, task types, model ([llm]), local scopes
    token         the personal access token (0600)
    secret        the AuthContext signing key (0600)
    agent.db      sessions, transcripts, the runtime's records
    system/       task types (task-types/<type>/AGENTS.md, WORKFLOW.md) and tools (tools/*.md)
    sandboxes/    per-user task types and plugins
"""

from __future__ import annotations

import argparse
import getpass
import hmac
import html
import ipaddress
import json
import os
import secrets
import shutil
import sys
import threading
import time
import webbrowser
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode

DEFAULT_PORT = 8765
# `admin`: the personal agent's user is also its manager (adds models, docs/agent-models.md).
DEFAULT_SCOPES = ("app:access agent:use admin code:read code:write knowledge:read dataset:read dataset:write")
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CONTEXT_TTL = 8 * 3600   # how long one signed AuthContext lives (also ends the live-update stream)
HINT_MIN = 8             # a shorter AGENT_LOCAL_HINT is ignored
HINT_TRIES, HINT_WAIT = 5, 60   # wrong hints in a row, then one try per HINT_WAIT seconds

CONFIG_TEMPLATE = """\
# Personal agent (agenticwf-agent serve): core configuration, read at start.
# See docs/local.md; the sections are the same as the container's agent.cfg.

[database]
default = sqlite:///{home}/agent.db?check_same_thread=false

[logging]
level = INFO
# file = {home}/agent.log

[context]
# Task types (task-types/<type>/AGENTS.md, WORKFLOW.md, task.yaml) and tools (tools/*.md).
system_dir = {home}/system
sandbox_root = {home}/sandboxes

[tools]
# Plugins: task types, tools and MCP servers, one directory each (docs/plugins.md).
# plugin_dirs = {home}/plugins
# top_k = 15             # tools offered at the start, closest to the request first
# top_k_skills = 5       # skills offered besides them (docs/skills.md)

[harness]
# split = false
# plan = never
# max_steps = 50
# conversation_messages = 20    # a chat's earlier messages are summarised past this many,
# conversation_tokens = 8000    # or past this many tokens; the newest few stay as they are
# conversation_keep = 4

[memory]
# Long-term memory, per person (docs/agent-structure.md, Long-term memory).
# consolidate_failed = false    # also keep what failed and cancelled tasks noted (it may be wrong)
# extract = llm                 # llm: a model call picks what lasts; notes: keep working-memory notes as they are
# model =                       # the model for that call; empty: the task's
# max_per_task = 5

[llm.routing]
# Choose models by what each call needs and costs (docs/model-routing.md). Models take part when
# they declare a tier and a context_window ([llm.model.<id>] tier = small|medium|large, ...).
# mode = auto                     # auto | pinned (always the named model, else the default)
# step = medium                   # the least tier per purpose: step, plan, split, reply, compact,
# split = small                   # compact_conversation, consolidate, final_report, cron_check, cron_report
# context_headroom = 0.25
# max_escalations = 1
# task_budget =                   # a task's cost limit (cost units)
# feedback = suggest              # suggest | adjust

[llm.purpose]
# Pin a purpose to a model id: compact_conversation = gemma

[llm.budgets]
# default = 20 per day            # per owner, in [llm] cost_unit; organization.<org> = 2000 per month
# max_tier = large                # max_tier.<subject> = medium

[cron]
# Scheduled checks, the issue board and shift reports (docs/cron.md). Crons are defined in
# <system dir>/crons/, plugins' crons/, users' sandboxes and on the Crons page.
# enabled = true                # run the scheduler in this process
# tick = 30                     # seconds between scheduler passes
# system_scopes =               # caps the scopes system and plugin crons may request; empty: no cap
# max_scopes =                  # caps a user's cron's scopes (otherwise the owner's, when they enabled it)
# deny_tools = ggus__create* jira__create*   # never offered to a cron's tasks (crons do not file tickets)

[embeddings]
# Match tools, skills and memories by meaning as well as by words ("changelog" finds release-notes).
# Empty provider: words only (BM25). See docs/agent-runtime.md, Selecting tools for a task.
# provider = voyage             # voyage | openai | ollama | vllm (any OpenAI-compatible /embeddings)
# model = voyage-3.5
# base_url =                    # the provider's default; required for vllm
# api_key_env = VOYAGE_API_KEY  # the provider's default
# weight = 0.6                  # share of meaning in the score; words the rest
# min_similarity = 0.3          # how close a candidate sharing no word must be

[llm]
# The default model id. `echo` (below) answers with the request; add real models here or, while
# the agent runs, from the API (POST /api/v1/admin/models). See docs/agent-models.md.
default = echo
# fallback = sonnet          # tried when the default's provider fails (rate limit, outage)
# Where models you add on the admin page may point (their base_url), besides the providers' own
# APIs. They use the key you give them, never this machine's environment.
allowed_hosts = localhost, 127.0.0.1

[llm.model.echo]
provider = echo
model = echo
label = Echo (testing)

# [llm.model.opus]
# provider = anthropic
# model = claude-opus-5-5
# label = Claude Opus 5.5
# effort = high
# fallbacks = default        # re-run declined requests on Anthropic's recommended fallback
#
# [llm.model.local]
# provider = ollama
# model = qwen3:32b
#
# [llm.provider.anthropic]
# api_key_env = ANTHROPIC_API_KEY

[local]
# Scopes the local user holds; tools declare the scopes they need.
scopes = {scopes}
# name = Your Name
"""


# ================================================================================================
# The auth front
# ================================================================================================
class LocalUser:
    def __init__(self, subject: str, name: str, scopes: frozenset[str]):
        self.subject, self.name, self.scopes = subject, name, scopes


class LocalAuth:
    """ASGI front for one local user: the token in, a signed AuthContext out."""

    def __init__(self, app: Any, token: str, secret: bytes, user: LocalUser):
        self.app = app
        self.token = token
        self.secret = secret
        self.user = user
        self._hint_failures, self._hint_last = 0, 0.0

    def cookie_name(self, scope: dict) -> str:
        port = (scope.get("server") or ("", DEFAULT_PORT))[1]
        return f"agenticwf_local_{port}"   # cookies ignore the port: one per instance

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            if scope["type"] == "websocket":   # no websocket routes; never pass one unauthenticated
                await send({"type": "websocket.close", "code": 1008})
                return
            await self.app(scope, receive, send)
            return

        # Never trust identity headers from the client: this front is the only signer.
        headers = [(k, v) for k, v in scope["headers"] if not k.lower().startswith(b"x-auth-")]
        scope = dict(scope, headers=headers)
        method, path = scope["method"], scope["path"]
        if path == "/healthz":
            await self.app(scope, receive, send)
            return
        if path == "/favicon.ico":   # browsers ask for it on every page, signed in or not
            await _respond(send, 204, b"")
            return
        if path.startswith("/auth/"):
            await self._auth_route(scope, receive, send)
            return

        via = self._credential(scope)
        query = parse_qs(scope.get("query_string", b"").decode())
        if via is None and method == "GET" and self._valid(query.get("token", [""])[0]):
            # The link printed at start: keep the token in a cookie, drop it from the address bar.
            rest = {k: v for k, v in query.items() if k != "token"}
            target = path + ("?" + urlencode(rest, doseq=True) if rest else "")
            await _respond(send, 303, b"", headers=[(b"location", target.encode()), self._set_cookie(scope)])
            return
        if via is None:
            if method == "GET" and not path.startswith("/api/"):
                await _respond(send, 302, b"", headers=[(b"location", b"/auth/login?return_to=" + quote(path).encode())])
            else:
                await _json(send, 401, {"error": "invalid_token", "error_description": "missing or wrong access token"})
            return
        if via == "cookie" and method not in SAFE_METHODS and not _same_origin(scope):
            await _json(send, 403, {"error": "csrf", "error_description": "cross-origin request with the session cookie"})
            return

        await self.app(self._signed(scope, via), receive, send)

    # -- credentials ----------------------------------------------------------------------------
    def _valid(self, candidate: str | None) -> bool:
        return bool(candidate) and hmac.compare_digest(candidate.encode(), self.token.encode())

    def _credential(self, scope: dict) -> str | None:
        """"bearer" or "cookie" when the request carries the right token, else None."""
        headers = _headers(scope)
        auth = headers.get("authorization", "")
        if auth[:7].lower() == "bearer " and self._valid(auth[7:].strip()):
            return "bearer"
        value = _cookies(headers.get("cookie", "")).get(self.cookie_name(scope))
        return "cookie" if self._valid(value) else None

    def _hint_answer(self, answer: str) -> str:
        """"ok", "wrong" or "wait" (too many wrong answers in a row: one try per HINT_WAIT)."""
        hint, now = _hint(), time.monotonic()
        if self._hint_failures >= HINT_TRIES and now < self._hint_last + HINT_WAIT:
            return "wait"
        if hint and hmac.compare_digest(answer.encode(), hint.encode()):
            self._hint_failures = 0
            return "ok"
        self._hint_failures, self._hint_last = self._hint_failures + 1, now
        return "wrong"

    def _set_cookie(self, scope: dict, clear: bool = False) -> tuple[bytes, bytes]:
        secure = "; Secure" if scope.get("scheme") == "https" else ""
        value, age = ("", 0) if clear else (self.token, 30 * 86400)
        return (b"set-cookie", f"{self.cookie_name(scope)}={value}; Path=/; Max-Age={age}; HttpOnly; "
                               f"SameSite=Strict{secure}".encode())

    def _signed(self, scope: dict, via: str) -> dict:
        from agenticwf.auth.context import AuthContext, encode_context

        ctx = AuthContext(subject=self.user.subject, user_id=self.user.subject, scopes=self.user.scopes,
                          expires_at=int(time.time()) + CONTEXT_TTL,
                          auth_method="session" if via == "cookie" else "bearer")
        payload, signature = encode_context(ctx, self.secret, 60, scope["method"], scope["path"])
        headers = [*scope["headers"], (b"x-auth-context", payload.encode()), (b"x-auth-context-sig", signature.encode())]
        return dict(scope, headers=headers)

    # -- /auth/* (what authsvc serves in a deployment) ------------------------------------------
    async def _auth_route(self, scope: dict, receive: Any, send: Any) -> None:
        method, path = scope["method"], scope["path"]
        query = parse_qs(scope.get("query_string", b"").decode())
        return_to = _safe_return_to(query.get("return_to", ["/"])[0])
        if path == "/auth/session":
            if self._credential(scope):
                body = {"authenticated": True, "approved": True,
                        "user": {"id": self.user.subject, "name": self.user.name}, "scopes": sorted(self.user.scopes)}
            else:
                body = {"authenticated": False, "login_url": "/auth/login"}
            await _json(send, 200, body)
            return
        if path == "/auth/login" and method == "GET":
            await _html(send, 200, _login_page(return_to))
            return
        if path == "/auth/login" and method == "POST":
            if not _same_origin(scope):
                await _json(send, 403, {"error": "csrf"})
                return
            form = parse_qs((await _body(receive)).decode())
            if not self._valid(form.get("token", [""])[0].strip()):
                await _html(send, 401, _login_page(_safe_return_to(form.get("return_to", ["/"])[0]), error="token"))
                return
            target = _safe_return_to(form.get("return_to", ["/"])[0])
            await _respond(send, 303, b"", headers=[(b"location", target.encode()), self._set_cookie(scope)])
            return
        if path == "/auth/hint" and method == "POST":
            # The right hint signs this browser in and shows the token (for scripts and other browsers).
            if not _same_origin(scope):
                await _json(send, 403, {"error": "csrf"})
                return
            form = parse_qs((await _body(receive)).decode())
            target = _safe_return_to(form.get("return_to", ["/"])[0])
            result = self._hint_answer(form.get("hint", [""])[0].strip())
            if result == "ok":
                await _html(send, 200, _token_page(self.token, target), [self._set_cookie(scope)])
            else:
                await _html(send, 429 if result == "wait" else 401, _login_page(target, error=result))
            return
        if path == "/auth/logout" and method == "POST":
            if not _same_origin(scope):
                await _json(send, 403, {"error": "csrf"})
                return
            accept = _headers(scope).get("accept", "")
            if "application/json" in accept:
                await _json(send, 200, {"logout_url": "/auth/login"}, extra=[self._set_cookie(scope, clear=True)])
            else:
                await _respond(send, 303, b"", headers=[(b"location", b"/auth/login"), self._set_cookie(scope, clear=True)])
            return
        await _json(send, 404, {"error": "not_found"})


def _cookies(header: str) -> dict[str, str]:
    """Name → value from a Cookie header. Lenient on purpose: the browser sends every cookie of
    the host (all ports: other local apps too), and http.cookies.SimpleCookie drops the whole
    header when any one of them has a character it does not accept (JSON, spaces, quotes)."""
    found: dict[str, str] = {}
    for part in header.split(";"):
        name, sep, value = part.strip().partition("=")
        if sep and name and name not in found:
            found[name] = value.strip().strip('"')
    return found


def _headers(scope: dict) -> dict[str, str]:
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}


def _same_origin(scope: dict) -> bool:
    """A write with the cookie must come from this page: Origin (or Sec-Fetch-Site) says so."""
    headers = _headers(scope)
    origin = headers.get("origin")
    if origin is None:
        return headers.get("sec-fetch-site", "same-origin") in ("same-origin", "none")
    return origin == f"{scope.get('scheme', 'http')}://{headers.get('host', '')}"


def _safe_return_to(value: str) -> str:
    """A path on this site only. Control characters and spaces are refused too: browsers drop tabs
    and newlines, so "/\t/evil.example" would become "//evil.example" (as authsvc's check does)."""
    if any(ord(c) < 0x21 or ord(c) == 0x7f for c in value):
        return "/"
    return value if value.startswith("/") and not value.startswith("//") and "\\" not in value else "/"


async def _body(receive: Any) -> bytes:
    chunks, more = [], True
    while more:
        message = await receive()
        chunks.append(message.get("body", b""))
        more = message.get("more_body", False)
    return b"".join(chunks)


async def _respond(send: Any, status: int, body: bytes, headers: list[tuple[bytes, bytes]] | None = None,
                   content_type: bytes = b"text/plain; charset=utf-8") -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", content_type), (b"content-length", str(len(body)).encode()),
                            (b"cache-control", b"no-store"), *(headers or [])]})
    await send({"type": "http.response.body", "body": body})


async def _json(send: Any, status: int, body: dict, extra: list | None = None) -> None:
    await _respond(send, status, json.dumps(body).encode(), extra, b"application/json")


async def _html(send: Any, status: int, page: str, extra: list | None = None) -> None:
    await _respond(send, status, page.encode(), [(b"x-frame-options", b"DENY"), *(extra or [])],
                   b"text/html; charset=utf-8")


def _hint() -> str:
    """AGENT_LOCAL_HINT (the container passes AGENTICWF_LOCAL_HINT): whoever gives it on the sign-in
    page is shown the token. Off when unset or shorter than HINT_MIN."""
    value = os.environ.get("AGENT_LOCAL_HINT", "").strip()
    return value if len(value) >= HINT_MIN else ""


def _token_help() -> str:
    """How to get the link and the token: AGENT_LOCAL_TOKEN_HELP when set (the container sets its
    `docker exec` commands; one per line, `#` lines explain), else this command's own."""
    lines = [line.strip() for line in os.environ.get("AGENT_LOCAL_TOKEN_HELP", "").splitlines() if line.strip()] \
        or ["# On this machine (the sign-in link was printed when it started):", "agenticwf-agent token"]
    return "\n".join(f'<p class="h">{html.escape(line.lstrip("# "))}</p>' if line.startswith("#")
                     else f"<pre>{html.escape(line)}</pre>" for line in lines)


PAGE_STYLE = """body{font:15px/1.5 system-ui,sans-serif;display:grid;place-items:center;min-height:100vh;margin:0}
main{width:min(560px,92vw)} h1{font-size:22px;margin:0 0 6px} p{color:#777;margin:0 0 18px}
p.h{margin:0 0 4px;font-size:13px} pre{margin:0 0 12px;padding:8px 10px;border:1px solid #8884;border-radius:8px;
font-size:12.5px;white-space:pre-wrap;word-break:break-all;user-select:all}
input{width:100%;box-sizing:border-box;padding:10px 12px;border:1px solid #8886;border-radius:10px;font:inherit}
button,a.btn{display:block;box-sizing:border-box;margin-top:12px;width:100%;padding:10px;border:0;border-radius:10px;
background:#2f6fed;color:#fff;font:inherit;text-align:center;text-decoration:none;cursor:pointer}
button.alt{background:none;border:1px solid #8886;color:inherit} form+form,details{margin-top:28px}
summary{cursor:pointer;color:#777;margin-bottom:10px} code{font-size:13px} .err{color:#c62828}"""


def _page(body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>AgenticWF · sign in</title>
<meta name="color-scheme" content="light dark"><style>{PAGE_STYLE}</style></head><body><main>
{body}</main></body></html>"""


def _login_page(return_to: str, error: str | None = None) -> str:
    """error: "token" (a wrong token), "wrong" (a wrong hint) or "wait" (too many wrong hints)."""
    back = f'<input type="hidden" name="return_to" value="{html.escape(return_to)}">'
    hinted = error in ("wrong", "wait")
    token_form = f"""<form method="post" action="/auth/login">
{'<p class="err">That token is not right.</p>' if error == "token" else ""}<input type="password" name="token" \
placeholder="Access token"{"" if hinted else " autofocus"} autocomplete="current-password">
{back}<button>Sign in</button></form>"""
    if not _hint():
        return _page(f"""<h1>AgenticWF</h1>
<p>Personal agent. Open its sign-in link, or paste its access token. To get them:</p>
{_token_help()}
{token_form}""")
    message = {"wrong": "That hint is not right.",
               "wait": "Too many wrong hints: try again in a minute."}.get(error, "")
    return _page(f"""<h1>AgenticWF</h1>
<p>Personal agent. Paste its access token, or give your hint to see it.</p>
{token_form}
<form method="post" action="/auth/hint"><p class="h">Lost the token? Your hint shows it.</p>
{f'<p class="err">{message} Get the token from the container instead (below).</p>' if message else ""}\
<input type="password" name="hint" placeholder="Hint"{" autofocus" if hinted else ""} autocomplete="off">
{back}<button class="alt">Show the token</button></form>
<details{" open" if hinted else ""}><summary>Or get the sign-in link and the token from the container</summary>
{_token_help()}</details>""")


def _token_page(token: str, return_to: str) -> str:
    return _page(f"""<h1>AgenticWF</h1>
<p>Signed in. Your access token, for other browsers and scripts (<code>Authorization: Bearer …</code>):</p>
<pre>{html.escape(token)}</pre>
<a class="btn" href="{html.escape(return_to)}">Continue</a>""")


# ================================================================================================
# Home directory, configuration, the command
# ================================================================================================
def default_home() -> Path:
    return Path(os.environ.get("AGENTICWF_AGENT_HOME") or Path.home() / ".agenticwf-agent").expanduser().resolve()


def _secret_file(path: Path, make: callable, renew: bool = False) -> str:
    if path.exists() and not renew:
        return path.read_text().strip()
    value = make()
    path.write_text(value + "\n")
    path.chmod(0o600)
    return value


def prepare_home(home: Path, renew_token: bool = False, config: Path | None = None) -> tuple[Path, str, bytes]:
    """Create the home directory on first use. Returns (agent.cfg, token, secret).

    With `config` (an existing configuration, e.g. the container's), the home directory only
    keeps the token and the key: database, task types and tools are whatever `config` says."""
    home.mkdir(parents=True, exist_ok=True)
    home.chmod(0o700)
    if config is not None:
        token = _secret_file(home / "token", lambda: secrets.token_urlsafe(32), renew=renew_token)
        secret = _secret_file(home / "secret", lambda: secrets.token_urlsafe(48)).encode()
        _secrets_key(home)
        return config, token, secret
    config = home / "agent.cfg"
    if not config.exists():
        escaped = str(home).replace("%", "%%")   # configparser interpolation
        config.write_text(CONFIG_TEMPLATE.format(home=escaped, scopes=DEFAULT_SCOPES))
    system = home / "system"
    if not (system / "task-types" / "default").is_dir():
        with as_file(files("agenticwf.agent.runtime") / "defaults") as defaults:
            shutil.copytree(defaults, system, dirs_exist_ok=True)
    (system / "tools").mkdir(parents=True, exist_ok=True)
    (home / "sandboxes").mkdir(exist_ok=True)
    token = _secret_file(home / "token", lambda: secrets.token_urlsafe(32), renew=renew_token)
    secret = _secret_file(home / "secret", lambda: secrets.token_urlsafe(48)).encode()
    _secrets_key(home)
    return config, token, secret


def _secrets_key(home: Path) -> None:
    """The key for API keys stored in the database: from the environment (the container sets
    it), else <home>/secrets_key, created once."""
    if os.environ.get("AGENT_SECRETS_KEY") or os.environ.get("AGENT_SECRETS_KEY_FILE"):
        return
    from agenticwf.agent.runtime.secrets import SecretBox

    _secret_file(home / "secrets_key", lambda: SecretBox.generate_key().decode())
    os.environ["AGENT_SECRETS_KEY_FILE"] = str(home / "secrets_key")


def local_user() -> LocalUser:
    """The local user: AGENT_LOCAL_USER / _NAME / _SCOPES, else the [local] section of the
    configuration (already loaded), else the login name and the default scopes."""
    from agenticwf.agent.core.common.config import config_get, config_has_option

    def opt(name: str, default: str) -> str:
        env = os.environ.get(f"AGENT_LOCAL_{name.upper()}")
        if env:
            return env
        return config_get("local", name) if config_has_option("local", name) else default

    login = opt("user", getpass.getuser())
    scopes = frozenset(opt("scopes", DEFAULT_SCOPES).replace(",", " ").split()) | {"app:access", "agent:use"}
    return LocalUser(subject=f"local:{login}", name=opt("name", login), scopes=scopes)


def build_app(token: str, secret: bytes) -> LocalAuth:
    """The API behind LocalAuth. The configuration must be loaded (AGENTICWF_CONFIG set) first."""
    from agenticwf.agent.api.app import create_app
    from agenticwf.agent.api.settings import Settings

    settings = Settings(context_secret=secret, bind_request=True, required_scopes=frozenset({"app:access"}),
                        max_stream_seconds=CONTEXT_TTL, trace_url=os.environ.get("AGENT_TRACE_URL", ""))
    return LocalAuth(create_app(settings), token, secret, local_user())


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="agenticwf-agent", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--home", type=Path, default=None, help="data directory (default ~/.agenticwf-agent)")
    sub = parser.add_subparsers(dest="command")
    serve = sub.add_parser("serve", help="run the personal agent (default)")
    serve.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1: this machine only)")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument("--open", action="store_true", help="open the chat in the browser")
    serve.add_argument("--new-token", action="store_true", help="replace the access token (signs out every browser)")
    serve.add_argument("--config", type=Path, default=None,
                       help="use this agent.cfg (e.g. the container's, with PostgreSQL) instead of the home's")
    serve.add_argument("--public-url", default=os.environ.get("AGENT_PUBLIC_URL"),
                       help="the address browsers use, for the printed link (e.g. http://localhost:8765 "
                            "when a container publishes the port)")
    serve.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "warning"))
    sub.add_parser("token", help="print the access token")
    args = parser.parse_args(argv)
    command = args.command or "serve"

    home = (args.home.expanduser().resolve() if args.home else default_home())
    given = getattr(args, "config", None)
    config, token, secret = prepare_home(home, renew_token=getattr(args, "new_token", False),
                                         config=given.expanduser().resolve() if given else None)
    if command == "token":
        print(token)
        return

    # agenticwf.agent.core.common.config is read on first import: point it at this home before importing the app.
    os.environ["AGENTICWF_CONFIG"] = str(config)
    host, port = getattr(args, "host", "127.0.0.1"), getattr(args, "port", DEFAULT_PORT)
    if not _is_loopback(host) and not os.environ.get("AGENT_CONTAINER"):   # in a container, -p decides
        print(f"warning: listening on {host}, reachable from the network over plain HTTP. Anyone with the "
              "token gets your agent; put it behind TLS (e.g. Caddy) or keep the default 127.0.0.1.", file=sys.stderr)

    import uvicorn

    from agenticwf.agent.api.db import init_db

    init_db()
    app = build_app(token, secret)
    shown = "localhost" if host in ("127.0.0.1", "::1", "0.0.0.0", "::") else host
    base = (args.public_url or f"http://{shown}:{port}").rstrip("/")
    url = f"{base}/?token={token}"
    print(f"\n  AgenticWF personal agent\n  open:   {url}\n  data:   {home}\n"
          f"  API:    curl -H \"Authorization: Bearer $(agenticwf-agent token)\" {base}/api/v1/me\n", flush=True)
    if getattr(args, "open", False):
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    # Open live-update streams never block a shutdown: requests get 5 s, then the agent stops its workers.
    uvicorn.run(app, host=host, port=port, log_level=args.log_level, proxy_headers=False, server_header=False,
                timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
