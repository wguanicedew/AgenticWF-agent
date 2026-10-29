import os
import tempfile
import time
from pathlib import Path

# agenticwf.agent.core.common.config is read at import time (agenticwf.agent.core.orm, logging), so write the test
# configuration before importing anything from src/agenticwf/agent/api/ or src/agenticwf/agent/core/. The database is SQLite unless
# AGENT_TEST_DB_URL points at PostgreSQL, e.g.
#   AGENT_TEST_DB_URL=postgresql+psycopg2://postgres:pw@127.0.0.1:5432/postgres pytest -q
_TMP = Path(tempfile.mkdtemp(prefix="agent-tests-"))
_DB_URL = os.environ.get("AGENT_TEST_DB_URL") or f"sqlite:///{_TMP / 'agent.db'}?check_same_thread=false"
_SYSTEM_DIR = Path(__file__).resolve().parents[1] / "src" / "agenticwf" / "agent" / "runtime" / "defaults"   # the task types that ship
(_TMP / "agent.cfg").write_text(
    "[database]\ndefault = %s\n\n[logging]\nlevel = WARNING\n\n[context]\nsystem_dir = %s\n\n"
    "[llm]\nprovider = echo\n" % (_DB_URL.replace("%", "%%"), _SYSTEM_DIR))
os.environ["AGENTICWF_CONFIG"] = str(_TMP / "agent.cfg")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agenticwf.auth.context import AuthContext, encode_context  # noqa: E402

from agenticwf.agent.api.app import create_app  # noqa: E402
from agenticwf.agent.api.settings import Settings  # noqa: E402

SECRET = b"s" * 48


def user(subject="user_alice", scopes=("app:access", "agent:use"), org=None, ttl=3600, client_id=None):
    return AuthContext(subject=subject, user_id=None if client_id == subject else subject, client_id=client_id,
                       organization_id=org, scopes=frozenset(scopes), expires_at=int(time.time()) + ttl)


def machine(app_id="app_m2m", scopes=("app:access", "agent:use")):
    return user(subject=app_id, scopes=scopes, client_id=app_id)


def signed(ctx, method, path, secret=SECRET, ttl=60, sign_path=None):
    """The headers Caddy copies from authsvc's /verify response."""
    payload, sig = encode_context(ctx, secret, ttl, method, sign_path or path)
    return {"X-Auth-Context": payload, "X-Auth-Context-Sig": sig}


class Client:
    """TestClient that signs every request for ``ctx``, like Caddy + authsvc would."""

    def __init__(self, client: TestClient, ctx):
        self.client, self.ctx = client, ctx

    def request(self, method, path, **kwargs):
        headers = signed(self.ctx, method, path.split("?", 1)[0]) if self.ctx else {}
        return self.client.request(method, path, headers=headers, **kwargs)

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, **kw):
        return self.request("POST", path, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)

    def patch(self, path, **kw):
        return self.request("PATCH", path, **kw)

    def put(self, path, **kw):
        return self.request("PUT", path, **kw)


@pytest.fixture(scope="session", autouse=True)
def database():
    from agenticwf.agent.api.db import init_db

    init_db()


@pytest.fixture(autouse=True)
def clean_tables(database):
    from agenticwf.agent.core.orm.base.session import BASE, get_engine

    with get_engine().begin() as conn:
        for table in reversed(BASE.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def app():
    return create_app(Settings(context_secret=SECRET))


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


@pytest.fixture
def as_user(client):
    return lambda ctx: Client(client, ctx)
