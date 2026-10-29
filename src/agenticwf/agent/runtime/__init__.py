"""
Agent runtime (skeleton). Design: docs/agent-runtime.md and docs/agent-structure.md.

    harness/   AgentHarness: lifecycle state machine, loop, limits      harness.harness
    llm/       LLM: provider-neutral, stateless model client             llm.base
    tools/     ToolRegistry, ToolDispatcher, tool .md loader             tools.registry, tools.dispatcher
    policy/    PolicyEngine: deterministic allow / ask / deny            policy.engine
    context/   AgentContext, working memory, task types                  context.agent_context, context.task_types

Import from the submodules; package __init__ files import nothing, which keeps the runtime free
of import cycles.
"""
