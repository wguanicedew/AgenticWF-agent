# Core tools

The tools the runtime itself provides: the description the model reads is the body, `summary`
the short form, `input_schema` the arguments. Edit the wording here; argument names are read by
code, so change them only together with it.

| File | What | Offered |
|---|---|---|
| `update_plan.md`, `update_working_memory.md`, `ask_user.md`, `search_tools.md` | Control tools (`agenticwf.agent.runtime.tools.control`, handled by `AgentHarness._control`) | Always (pinned, in this order) |
| `complete_turn.md` | Control tool | In the turns of a multi-turn plan |
| `read_skill_file.md` | Control tool | While a skill with files runs |
| `report_issue.md`, `report_check.md` | Control tools (`agenticwf.agent.runtime.cron`) | In a cron's check tasks |
| `memory_search.md` | `agenticwf.agent.runtime.context.memory` | When long-term memory is configured |
| `skill.md` | The template every skill is offered with, `skill__<name>` (`agenticwf.agent.runtime.skills.skill`); `${description}` is the skill's own description | For each skill the task may use |

These are packaged with the runtime and are not read from `<system dir>`: unlike `prompts/`,
operators cannot override them.
