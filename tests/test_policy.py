"""deploy/policy.agent.yaml, evaluated by authsvc's own policy engine."""

import re
from pathlib import Path

import pytest

from tests.conftest import machine, user

POLICY = Path(__file__).resolve().parents[1] / "deploy" / "policy.agent.yaml"


@pytest.fixture(scope="module")
def policy():
    from agenticwf.auth.authz import Policy

    return Policy.load(POLICY)


def test_every_route_has_a_rule(app, policy):
    # authsvc denies routes no rule matches ("no policy rule grants access to this route").
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith(("/api/", "/admin", "/crons")):
            continue
        concrete = re.sub(r"\{[^}]+\}", "x", path)
        for method in getattr(route, "methods", None) or ("GET",):
            assert policy.match(method, concrete) is not None, f"no policy rule for {method} {path}"


def test_admin_page_reads_and_people_only_writes(policy):
    for method, path in [("GET", "/api/v1/instructions"), ("GET", "/api/v1/code/repos"),
                         ("PUT", "/api/v1/models/default"), ("PUT", "/api/v1/skills/s/files/SKILL.md")]:
        assert policy.authorize(user(), method, path).allowed, (method, path)
    assert not policy.authorize(machine(), "PUT", "/api/v1/models/default").allowed
    assert not policy.authorize(machine(), "POST", "/api/v1/code/repos").allowed
