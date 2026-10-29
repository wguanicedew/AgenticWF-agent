"""The personal agent (agenticwf.agent.api.local): the token front that stands in for Caddy + authsvc."""

import getpass
import stat

import pytest
from fastapi.testclient import TestClient

from agenticwf.auth.context import encode_context

from agenticwf.agent.api.local import build_app, prepare_home
from tests.conftest import user

TOKEN = "t" * 43
SECRET = b"k" * 48
ORIGIN = {"Origin": "http://testserver"}


@pytest.fixture
def local():
    with TestClient(build_app(TOKEN, SECRET), follow_redirects=False) as client:
        yield client


def test_without_token(local):
    assert local.get("/healthz").status_code == 200
    page = local.get("/")
    assert page.status_code == 302 and page.headers["location"] == "/auth/login?return_to=/"
    assert local.get("/api/v1/me").json()["error"] == "invalid_token"
    assert local.get("/api/v1/me", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert local.get("/auth/session").json() == {"authenticated": False, "login_url": "/auth/login"}
    assert "Access token" in local.get("/auth/login").text


def test_link_with_token_sets_cookie(local):
    r = local.get(f"/?token={TOKEN}")
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith("agenticwf_local_80=") and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert local.get("/").status_code == 200                      # the cookie now
    session = local.get("/auth/session").json()
    assert session["authenticated"] and session["user"]["id"] == f"local:{getpass.getuser()}"
    me = local.get("/api/v1/me").json()
    assert me["auth_method"] == "session" and "agent:use" in me["scopes"]


def test_bearer_token(local):
    me = local.get("/api/v1/me", headers={"Authorization": f"Bearer {TOKEN}"}).json()
    assert me["subject"] == f"local:{getpass.getuser()}" and me["auth_method"] == "bearer"


def test_client_identity_headers_are_dropped(local):
    payload, sig = encode_context(user(subject="mallory"), SECRET, 60, "GET", "/api/v1/me")   # even with the real key
    r = local.get("/api/v1/me", headers={"X-Auth-Context": payload, "X-Auth-Context-Sig": sig})
    assert r.status_code == 401
    r = local.get("/api/v1/me", headers={"X-Auth-Context": payload, "X-Auth-Context-Sig": sig,
                                         "Authorization": f"Bearer {TOKEN}"})
    assert r.json()["subject"] != "mallory"


def test_cookie_writes_need_same_origin(local):
    local.get(f"/?token={TOKEN}")
    assert local.post("/api/v1/sessions", json={}, headers={"Origin": "https://evil.example"}).status_code == 403
    sid = local.post("/api/v1/sessions", json={}, headers=ORIGIN).json()["id"]
    r = local.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hi"}, headers=ORIGIN)
    assert r.json()["reply"] == "echo: hi"
    # Bearer requests are not cookie-borne: no Origin needed (curl, scripts).
    assert local.post("/api/v1/sessions", json={}, headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 201


def test_login_form_and_logout(local):
    bad = local.post("/auth/login", data={"token": "nope", "return_to": "/"}, headers=ORIGIN)
    assert bad.status_code == 401 and "not right" in bad.text
    assert local.post("/auth/login", data={"token": TOKEN}, headers={"Origin": "https://evil.example"}).status_code == 403
    ok = local.post("/auth/login", data={"token": TOKEN, "return_to": "//evil.example/x"}, headers=ORIGIN)
    assert ok.status_code == 303 and ok.headers["location"] == "/"          # no open redirect
    assert local.get("/auth/session").json()["authenticated"]
    out = local.post("/auth/logout", headers=ORIGIN)
    assert out.status_code == 303 and "Max-Age=0" in out.headers["set-cookie"]
    local.cookies.clear()
    assert not local.get("/auth/session").json()["authenticated"]


def test_prepare_home(tmp_path):
    home = tmp_path / "home"
    config, token, secret = prepare_home(home)
    assert (home / "system" / "task-types" / "default" / "AGENTS.md").is_file() and (home / "system" / "tools").is_dir()
    assert f"sqlite:///{home}/agent.db" in config.read_text() and "provider = echo" in config.read_text()
    for name in ("token", "secret"):
        assert stat.S_IMODE((home / name).stat().st_mode) == 0o600
    assert stat.S_IMODE(home.stat().st_mode) == 0o700
    assert prepare_home(home)[1:] == (token, secret)                         # stable across starts
    assert prepare_home(home, renew_token=True)[1] != token


def test_prepare_home_with_a_given_config(tmp_path):
    """The container's local mode: its own agent.cfg (PostgreSQL); the home keeps the token and key only."""
    given = tmp_path / "agent.cfg"
    given.write_text("[database]\ndefault = postgresql+psycopg2://agent@/agent?host=/run/postgresql\n")
    config, token, secret = prepare_home(tmp_path / "home", config=given)
    assert config == given and token and secret
    assert sorted(p.name for p in (tmp_path / "home").iterdir()) == ["secret", "token"]


def test_local_user_from_environment(monkeypatch):
    from agenticwf.agent.api.local import local_user

    monkeypatch.setenv("AGENT_LOCAL_USER", "wguan")
    monkeypatch.setenv("AGENT_LOCAL_NAME", "Wen Guan")
    monkeypatch.setenv("AGENT_LOCAL_SCOPES", "code:read")
    u = local_user()
    assert u.subject == "local:wguan" and u.name == "Wen Guan"
    assert u.scopes == {"code:read", "app:access", "agent:use"}   # the agent's own scopes are always there


def test_cookie_among_other_local_apps_cookies(local):
    """Browsers send the cookies of every app on localhost (cookies ignore the port)."""
    others = 'grafana_session=x; ajs_user_id="u"; prefs={"theme": "dark"}; t=a b'
    r = local.get("/", headers={"Cookie": f"{others}; agenticwf_local_80={TOKEN}"})
    assert r.status_code == 200
    assert local.get("/", headers={"Cookie": f"{others}; agenticwf_local_80=wrong"}).status_code == 302


def test_favicon_is_not_a_login_redirect(local):
    assert local.get("/favicon.ico").status_code == 204


def test_login_page_shows_how_to_get_the_token(local, monkeypatch):
    assert "agenticwf-agent token" in local.get("/auth/login").text              # run on the host
    monkeypatch.setenv("AGENT_LOCAL_TOKEN_HELP", "# In the container's host:\ndocker exec abc <x> agenticwf-links agent\n")
    page = local.get("/auth/login").text
    assert '<p class="h">In the container&#x27;s host:</p>' in page
    assert "<pre>docker exec abc &lt;x&gt; agenticwf-links agent</pre>" in page and "agenticwf-agent token" not in page


def test_hint_shows_the_token(local, monkeypatch):
    assert 'action="/auth/hint"' not in local.get("/auth/login").text             # no hint: no hint form
    monkeypatch.setenv("AGENT_LOCAL_HINT", "short")                                 # too short: off
    assert 'action="/auth/hint"' not in local.get("/auth/login").text
    monkeypatch.setenv("AGENT_LOCAL_HINT", "blue whale 42")
    page = local.get("/auth/login").text
    assert 'action="/auth/hint"' in page and "<details>" in page                   # the commands, folded
    assert local.post("/auth/hint", data={"hint": "blue whale 42"},
                      headers={"Origin": "https://evil.example"}).status_code == 403
    wrong = local.post("/auth/hint", data={"hint": "red whale"}, headers=ORIGIN)
    assert wrong.status_code == 401 and "hint is not right" in wrong.text and "<details open>" in wrong.text
    assert TOKEN not in wrong.text and "set-cookie" not in wrong.headers
    right = local.post("/auth/hint", data={"hint": " blue whale 42 ", "return_to": "/chat"}, headers=ORIGIN)
    assert right.status_code == 200 and f"<pre>{TOKEN}</pre>" in right.text and 'href="/chat"' in right.text
    assert local.get("/api/v1/me").status_code == 200                              # and signed in


def test_wrong_hints_are_throttled(local, monkeypatch):
    import agenticwf.agent.api.local as module

    monkeypatch.setenv("AGENT_LOCAL_HINT", "blue whale 42")
    for _ in range(module.HINT_TRIES):
        assert local.post("/auth/hint", data={"hint": "guess"}, headers=ORIGIN).status_code == 401
    held = local.post("/auth/hint", data={"hint": "blue whale 42"}, headers=ORIGIN)   # even the right one waits
    assert held.status_code == 429 and TOKEN not in held.text
    monkeypatch.setattr(module, "HINT_WAIT", 0)
    assert local.post("/auth/hint", data={"hint": "blue whale 42"}, headers=ORIGIN).status_code == 200
