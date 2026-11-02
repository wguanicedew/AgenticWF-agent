"""Claude Code and Codex plugins, skills and tool definitions, read as they are written for those hosts."""

import asyncio
import json
import sys

import pytest

from agenticwf.agent.runtime.mcp.endpoint import EndpointError, parse_endpoint
from agenticwf.agent.runtime.mcp.manager import McpManager, McpSettings
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.skills.catalog import SkillCatalog
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.llm.base import LLMResponse, ToolCallRequest
from agenticwf.agent.runtime.policy.engine import DecisionKind, PolicyConfig, PolicyEngine, PolicyRule
from agenticwf.agent.runtime.records import Intent
from agenticwf.agent.runtime.skills.skill import Skill, parse_aliases, translate_tools
from agenticwf.agent.runtime.types import RuntimeState
from agenticwf.agent.runtime.tools.loader import load_plugin_tools
from agenticwf.agent.runtime.tools.spec import ToolResult, ToolSource

from tests.test_plugins import MCP_SERVER
from tests.test_runtime import make_system_dir
from tests.test_skills import person, run, write_skill

ALICE = ("user_alice", None)
ALICE_CERN, BOB = ("user_alice", "cern"), ("user_bob", "cern")


def claude_plugin(root):
    """code-review, as a Claude Code plugin: plugin.json, a skill, commands, an MCP server, agents and hooks."""
    plugin = root / "code-review"
    (plugin / ".claude-plugin").mkdir(parents=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(json.dumps(
        {"name": "code-review", "version": "2.1.0", "description": "Review pull requests",
         "author": {"name": "Ops", "email": "ops@example.org"}, "hooks": "./hooks/hooks.json"}))
    write_skill(plugin / "skills", "review-pr", "Review a pull request for bugs.",
                body="Review $ARGUMENTS.\nRead ${CLAUDE_SKILL_DIR}/checklist.md and "
                     "${CLAUDE_PLUGIN_ROOT}/skills/review-pr/style.md first.",
                extra="allowed-tools: Read, Grep, Bash(git diff:*), mcp__plugin_code-review_github__get_pull\n",
                files={"checklist.md": "- tests", "style.md": "- short"})
    write_skill(plugin / "skills", "release", "Cut a release.", extra="disable-model-invocation: true\n")
    (plugin / "commands" / "git").mkdir(parents=True)
    (plugin / "commands" / "summarize.md").write_text(
        "---\ndescription: Summarize the change\nallowed-tools: [Read]\n---\nSummarize $ARGUMENTS.\n")
    (plugin / "commands" / "git" / "commit.md").write_text("# Write a commit message\n\nUse the staged diff.\n")
    (plugin / "commands" / "review-pr.md").write_text("Same name as the skill.\n")
    (plugin / "server.py").write_text(MCP_SERVER)
    (plugin / ".mcp.json").write_text(json.dumps(   # without the mcpServers wrapper
        {"github": {"command": sys.executable, "args": ["${CLAUDE_PLUGIN_ROOT}/server.py"]}}))
    (plugin / "agents").mkdir()
    (plugin / "agents" / "reviewer.md").write_text("---\nname: reviewer\n---\nReview.\n")
    return plugin


def codex_plugin(root):
    """notes, as a Codex plugin: its skills and MCP servers where plugin.json says."""
    plugin = root / "notes"
    (plugin / ".codex-plugin").mkdir(parents=True)
    (plugin / ".codex-plugin" / "plugin.json").write_text(json.dumps(
        {"name": "notes", "version": "0.3.0", "description": "Meeting notes",
         "skills": "./codex-skills/", "mcpServers": "./servers.json", "apps": "./.app.json"}))
    skill = write_skill(plugin / "codex-skills", "take-notes", "Take meeting notes.")
    (skill / "agents").mkdir()
    (skill / "agents" / "openai.yaml").write_text("interface:\n  display_name: Notes\npolicy:\n  allow_implicit_invocation: false\n")
    write_skill(plugin / "codex-skills", "summarize", "Summarize notes.", extra="allowed-tools: [apply_patch, shell]\n")
    (plugin / "servers.json").write_text(json.dumps({"mcpServers": {"store": {"command": sys.executable, "args": ["server.py"]}}}))
    (plugin / "server.py").write_text(MCP_SERVER)
    return plugin


@pytest.fixture
def plugins(tmp_path):
    root = tmp_path / "plugins"
    claude_plugin(root)
    codex_plugin(root)
    manager = PluginManager([root], mcp=McpManager(McpSettings(connect_timeout=10)))
    yield manager
    manager.mcp.close()


def test_a_claude_code_plugin_loads_as_written(plugins):
    errors = plugins.load()
    plugin = plugins.get("code_review")
    assert plugin.manifest.version == "2.1.0" and plugin.manifest.author == "Ops"
    assert any("hooks ignored: hooks are not supported" in e for e in errors)
    assert any("agents/ ignored: subagents are not supported" in e for e in errors)
    assert any("'review-pr' is already provided" in e for e in errors)          # skill first, command reported

    assert plugin.mcp_endpoints[0].expansion.status == "ok"                    # ${CLAUDE_PLUGIN_ROOT}
    assert {"code_review__echo", "code_review__fail_hard"} <= {s.name for s in plugins.tools()}

    skills = {k.name: k for k in plugin.skills}
    assert set(skills) == {"review-pr", "release", "summarize", "git-commit"}
    review = skills["review-pr"]
    assert review.id == "system:code_review.review-pr" and review.tool_name == "skill__code_review_review_pr"
    assert review.tools == () and review.approved == ("Read", "Grep", "Bash(git diff:*)",
                                                      "mcp__plugin_code-review_github__get_pull")   # as written
    assert "Read checklist.md and style.md first." in review.instructions
    loaded = review.activation({"purpose": "PR #42"})
    assert "Review PR #42." in loaded.content and "checklist.md, style.md" in loaded.content
    assert loaded.facts["skill"]["tools"] == []                                  # allowed-tools never narrow
    assert loaded.facts["skill"]["approved"] == ["read_file", "search_code", "run_command", "code_review__get_pull"]
    assert skills["git-commit"].description == "Write a commit message" and skills["git-commit"].approved == ()
    assert skills["summarize"].approved == ("Read",) and skills["summarize"].files() == []
    assert not skills["release"].model_invocable


def test_a_codex_plugin_loads_as_written(plugins):
    plugins.load()
    plugin = plugins.get("notes")
    assert plugin.manifest.version == "0.3.0"
    assert any("apps ignored" in e for e in plugin.errors)
    assert [e.endpoint.name for e in plugin.mcp_endpoints] == ["store"]
    assert "notes__echo" in {s.name for s in plugins.tools()}
    skills = {k.name: k for k in plugin.skills}
    assert not skills["take-notes"].model_invocable                             # agents/openai.yaml policy
    assert skills["summarize"].activation().facts["skill"]["approved"] == ["edit_file", "run_command"]


def test_skills_only_people_invoke_are_listed_but_not_offered(plugins):
    catalog = SkillCatalog(plugins=plugins)
    listed = {k.qualified for k in catalog.skills(ALICE)}
    offered = {s.name for s in catalog.tools_for(ALICE)}
    assert {"code_review.release", "notes.take-notes"} <= listed
    assert "skill__code_review_release" not in offered and "skill__notes_take_notes" not in offered
    assert "skill__code_review_review_pr" in offered


def test_skills_in_claude_and_codex_directories_of_a_sandbox(tmp_path):
    sandbox = tmp_path / "sandboxes" / "user_alice"
    write_skill(sandbox / "skills", "notes", "Mine.")
    write_skill(sandbox / ".claude" / "skills", "commit", "Write commit messages.")
    write_skill(sandbox / ".agents" / "skills", "deploy", "Deploy the app.")
    write_skill(sandbox / ".codex" / "skills", "notes", "A second notes.")
    catalog = SkillCatalog(user_sandbox=lambda owner: tmp_path / "sandboxes" / owner[0])
    assert {k.name for k in catalog.skills(ALICE)} == {"notes", "commit", "deploy"}
    assert next(k for k in catalog.skills(ALICE) if k.name == "notes").description == "Mine."


def test_tool_names_of_other_hosts():
    assert translate_tools(("Read", "Edit", "Write", "WebFetch", "Bash(npm test:*)", "shell")) == (
        "read_file", "edit_file", "web_fetch", "run_command")
    assert translate_tools(("mcp__github__list_issues", "mcp__my-server__*")) == ("github__list_issues", "my_server__*")
    assert translate_tools(("mcp__plugin_lint_rules__check",), plugin="lint") == ("lint__check",)
    assert translate_tools(("read_thing", "github__*")) == ("read_thing", "github__*")
    assert translate_tools(("Task", "Skill")) == ()                              # nothing left: no narrowing


def test_tool_definitions_written_for_other_hosts(tmp_path):
    plugin = tmp_path / "my-tools"
    (plugin / "tools").mkdir(parents=True)
    (plugin / "tools" / "lookup.md").write_text(
        "---\nname: lookup\ndescription: Look a word up. Returns its meaning.\n"
        "parameters:\n  type: object\n  properties:\n    word: {type: string}\n"
        "handler:\n  kind: http\n  url: https://example.org/lookup\n---\n")
    specs, errors = load_plugin_tools(plugin, ToolSource("user_plugin", owner=ALICE), "my_tools")
    assert errors == []
    assert specs[0].name == "my_tools__lookup" and specs[0].summary == "Look a word up."
    assert specs[0].description == "Look a word up. Returns its meaning."
    assert specs[0].input_schema["properties"] == {"word": {"type": "string"}}


def test_mcp_settings_with_defaults_and_the_plugin_root(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_TEST_UNSET", raising=False)
    endpoint = parse_endpoint("srv", {"command": "${CLAUDE_PLUGIN_ROOT}/run", "args": ["${AGENT_TEST_UNSET:-fallback}"],
                                      "env": {"EMPTY": "${AGENT_TEST_UNSET:-}"}}, source="plugin", plugin="p", root=tmp_path)
    assert endpoint.command == f"{tmp_path}/run" and endpoint.args == ("fallback",) and endpoint.env == {"EMPTY": ""}
    with pytest.raises(EndpointError, match="AGENT_TEST_UNSET is not set"):
        parse_endpoint("srv", {"command": "${AGENT_TEST_UNSET}"}, source="plugin", plugin="p", root=tmp_path)
    user = parse_endpoint("srv", {"url": "https://mcp.example.org/${CLAUDE_PLUGIN_ROOT}"}, source="user_plugin",
                          plugin="p", root=tmp_path, owner=ALICE)
    assert str(tmp_path) in user.url                                             # the root is not the environment


def test_tool_aliases_from_the_configuration():
    aliases = parse_aliases("Bash: shell_exec, Read: read_file kb_read\nTask: dispatch,\nWebSearch:")
    assert translate_tools(("Bash(ls)", "Read", "Task", "WebSearch", "Edit"), aliases=aliases) == (
        "shell_exec", "read_file", "kb_read", "dispatch", "edit_file")             # defaults kept, Edit
    with pytest.raises(ValueError):
        parse_aliases("Bash run_command")


# --- pre-approval -------------------------------------------------------------------------------

def build(system, catalog, script, policy=None):
    registry = ToolRegistry(system_dir=system, skills=catalog)
    registry.load()
    for name in ("read_thing", "write_thing", "delete_thing"):
        registry.register_handler(f"test.{name}", lambda ctx, args, n=name: ToolResult(content=f"{n} done"))
    return AgentHarness(HarnessServices(llm=ScriptedLLM(script), task_types=TaskTypeResolver(system),
                                        tool_dispatcher=ToolDispatcher(registry, policy or PolicyEngine()),
                                        skills=catalog))


@pytest.fixture
def approving(tmp_path):
    """delete_thing asks (make_system_dir); a system skill and Alice's own skill pre-approve it."""
    system = make_system_dir(tmp_path)
    write_skill(system / "skills", "cleanup", "Clean up old things.", extra="allowed-tools: [delete_thing]\n")
    sandboxes = tmp_path / "sandboxes"
    write_skill(sandboxes / "user_alice" / "skills", "my-cleanup", "Alice cleans up.",
                extra="allowed-tools: delete_thing\n")
    catalog = SkillCatalog(system, user_sandbox=lambda owner: sandboxes / owner[0])
    return system, catalog


def delete_after(skill_tool):
    return [LLMResponse("", [ToolCallRequest("c1", skill_tool, {})]),
            LLMResponse("", [ToolCallRequest("c2", "delete_thing", {"value": "x"})]), LLMResponse("done")]


@pytest.mark.parametrize("skill_tool", ["skill__cleanup", "skill__my_cleanup"])
def test_a_skill_pre_approves_its_allowed_tools(approving, skill_tool):
    system, catalog = approving
    h = build(system, catalog, delete_after(skill_tool))
    task = run(h, h.create_task(person(ALICE), "clean up"))
    assert task.state is RuntimeState.COMPLETED
    assert {c.id: c.result.content for c in task.calls}["c2"] == "delete_thing done"   # no approval asked
    assert "delete_thing" in {t["name"] for t in h.services.llm.requests[1].tools}       # offered, not narrowed to


def test_without_the_skill_or_with_pre_approval_off_it_asks(approving):
    system, catalog = approving
    h = build(system, catalog, [LLMResponse("", [ToolCallRequest("c1", "delete_thing", {"value": "x"})])])
    assert run(h, h.create_task(person(ALICE), "clean up")).state is RuntimeState.WAITING_APPROVAL
    h = build(system, catalog, delete_after("skill__cleanup"), PolicyEngine(PolicyConfig(skill_preapproval=False)))
    assert run(h, h.create_task(person(ALICE), "clean up")).state is RuntimeState.WAITING_APPROVAL


def test_someone_elses_skill_pre_approves_nothing():
    shared = Skill(name="cleanup", description="Clean.", instructions="…", source="database", owner=BOB,
                   approved=("delete_thing",), visibility="organization")
    catalog = SkillCatalog()
    assert catalog.activate(shared, BOB).facts["skill"]["approved"] == ["delete_thing"]
    assert catalog.activate(shared, ALICE_CERN).facts["skill"]["approved"] == []


# --- /skill -------------------------------------------------------------------------------------

def intake(h, text, owner=ALICE):
    async def main():
        intent = Intent(thread_id="t", owner=owner, auth=person(owner), text=text)
        h.store.save_intent(intent)
        return await h.intake(intent.id)
    return asyncio.run(main())


def test_a_person_invokes_a_skill_with_a_slash(tmp_path):
    system = make_system_dir(tmp_path)
    write_skill(system / "skills", "release", "Cut a release.", body="Release version $ARGUMENTS.",
                extra="disable-model-invocation: true\nallowed-tools: [delete_thing]\n")
    catalog = SkillCatalog(system)
    h = build(system, catalog, [LLMResponse("", [ToolCallRequest("c1", "delete_thing", {"value": "x"})]),
                                      LLMResponse("Released.")])
    [task] = intake(h, "/release 2.0")
    assert task.input == "2.0" and task.invoked_skill == {"id": "system:release", "purpose": "2.0"}
    assert task.active_skill["id"] == "system:release"
    task = run(h, task)
    assert task.state is RuntimeState.COMPLETED                                  # pre-approved by the skill
    first = h.services.llm.requests[0]
    assert any('<skill name="release">\nRelease version 2.0.' in m.content for m in first.messages if m.role == "user")
    assert not any(t["name"].startswith("skill__") for t in first.tools)


def test_a_slash_that_names_no_skill_is_an_ordinary_request(tmp_path):
    system = make_system_dir(tmp_path)
    h = build(system, SkillCatalog(system), [])
    [task] = intake(h, "/etc/hosts has a typo")
    assert task.input == "/etc/hosts has a typo" and task.invoked_skill is None


def test_a_denied_skill_is_not_loaded_by_a_slash(tmp_path):
    system = make_system_dir(tmp_path)
    write_skill(system / "skills", "release", "Cut a release.")
    catalog = SkillCatalog(system)
    h = build(system, catalog, [], PolicyEngine(PolicyConfig(rules=(PolicyRule("skill__release", DecisionKind.DENY),))))
    [task] = intake(h, "/release 2.0")
    assert task.active_skill is None
