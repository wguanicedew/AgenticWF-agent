"""The service accepts only AuthContexts signed by authsvc for this exact request."""

import pytest

from agenticwf.agent.api.settings import Settings
from tests.conftest import SECRET, machine, signed, user


def test_health_is_public(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_no_context_is_401(client):
    assert client.get("/api/v1/me").status_code == 401
    assert client.get("/api/v1/sessions").status_code == 401


def test_me(as_user):
    body = as_user(user()).get("/api/v1/me").json()
    assert body["subject"] == "user_alice" and body["approved"] is True and body["is_machine"] is False


def test_forged_signature_rejected(client):
    headers = signed(user(), "GET", "/api/v1/me", secret=b"x" * 48)
    r = client.get("/api/v1/me", headers=headers)
    assert r.status_code == 401 and r.json()["error"] == "invalid_auth_context"


def test_context_bound_to_path_and_method(client):
    assert client.get("/api/v1/sessions", headers=signed(user(), "GET", "/api/v1/me")).status_code == 401
    assert client.post("/api/v1/sessions", json={}, headers=signed(user(), "GET", "/api/v1/sessions")).status_code == 401


def test_expired_context_rejected(client):
    headers = signed(user(), "GET", "/api/v1/me", ttl=-1)
    assert client.get("/api/v1/me", headers=headers).status_code == 401


def test_pending_account(as_user):
    pending = as_user(user(scopes=()))
    assert pending.get("/api/v1/me").json()["approved"] is False      # allowed before approval
    r = pending.get("/api/v1/sessions")
    assert r.status_code == 403 and r.json()["detail"] == "approval_pending"


def test_missing_scope(as_user):
    r = as_user(user(scopes=("app:access",))).post("/api/v1/sessions", json={})
    assert r.status_code == 403 and "agent:use" in r.json()["detail"]


def test_machine_tokens_cannot_chat_but_can_submit_tasks(as_user):
    m2m = as_user(machine())
    assert m2m.post("/api/v1/sessions", json={}).status_code == 403
    assert m2m.post("/api/v1/tasks", json={"prompt": "x"}).status_code == 202


def test_settings_require_strong_secret(monkeypatch):
    monkeypatch.delenv("AUTH_CONTEXT_SECRET", raising=False)
    monkeypatch.delenv("AUTH_CONTEXT_SECRET_FILE", raising=False)
    with pytest.raises(RuntimeError):
        Settings.from_env()
    monkeypatch.setenv("AUTH_CONTEXT_SECRET", "short")
    with pytest.raises(RuntimeError):
        Settings.from_env()


def test_settings_from_secret_file(monkeypatch, tmp_path):
    secret_file = tmp_path / "auth_context_secret"
    secret_file.write_text(SECRET.decode() + "\n")
    monkeypatch.delenv("AUTH_CONTEXT_SECRET", raising=False)
    monkeypatch.setenv("AUTH_CONTEXT_SECRET_FILE", str(secret_file))
    monkeypatch.setenv("AGENT_REQUIRED_SCOPES", "app:access, beta")
    settings = Settings.from_env()
    assert settings.context_secret == SECRET and settings.required_scopes == {"app:access", "beta"}


def test_unbound_mode_accepts_rewritten_paths(as_user):
    from fastapi.testclient import TestClient

    from agenticwf.agent.api.app import create_app

    app = create_app(Settings(context_secret=SECRET, bind_request=False))
    with TestClient(app) as c:
        headers = signed(user(), "GET", "/rewritten/me")
        assert c.get("/api/v1/me", headers=headers).status_code == 200
        assert c.get("/api/v1/me", headers=signed(user(), "GET", "/x", ttl=-1)).status_code == 401
