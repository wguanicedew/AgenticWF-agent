# Prompts

The instructions the runtime gives the model around the agent loop (docs/agent-flow.md, Prompts).
Edit a copy to change how the system behaves; a file left out falls back to the packaged one.

| File | Used for | Per task type |
|---|---|---|
| `cron_check.md` | The request of a cron's check task: its window, key fields, open issues, suppressions (docs/cron.md) | no |
| `cron_report.md` | The summary of a cron's report, when the cron's period gives no template of its own | no |
| `preamble.md` | The runtime's rules, first in every task's system prompt (above `AGENTS.md`) | no (they apply whatever a type says) |
| `split.md` | Intake: split a request into typed tasks (`[harness] split = true`) | no (runs before a type is chosen) |
| `reply.md` | Combine several tasks' results into the reply | no (one reply per request) |
| `plan.md` | Plan a task into turns (`plan: always`); `${max_items}` is the limit | yes |
| `combine.md` | Combine a multi-turn task's turn results into its answer | yes |
| `final_report.md` | The last, tool-free step at `max_steps` | yes |
| `compact.md` | Summarise old steps when the context grows large | yes |
| `consolidate.md` | Pick what a finished task leaves in long-term memory (`[memory] extract = llm`); `${max_items}` is the limit | yes |
| `select_tools.md` | Pick a task's tools from their names and summaries, with a large catalogue (`[tools] select_model_above`); `${max_items}` is the limit | no |
| `compact_conversation.md` | Summarise the older part of a chat when it grows too long (`[harness] conversation_tokens`); `${max_words}` is the length | no (one per thread) |

A task type overrides a per-type prompt with `task-types/<type>/prompts/<name>.md`; users do the
same for their own types by cloning a prompt on the admin page's Instructions tab. The JSON
the prompts ask for (`split.md`, `plan.md`, `consolidate.md`, `select_tools.md`) is what the runtime parses; if a reply
does not parse, it falls back to one task, one turn, the working memory's notes as they are, or tools chosen by search.

The wording of the runtime's own tools (control tools, `memory_search`, the tool each skill is
offered as) is in `../core-tools/`, which operators do not override; a task type may reword them
for its tasks (`../core-tools/README.md`).
