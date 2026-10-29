[runtime] Before you start, plan this task. Do not do the work yet.
Break it into at most ${max_items} turns, in order. Each turn has one goal; a small task has one turn.
Reply with JSON only:
{"items": [{"id": "t1", "goal": "...", "phase": "<a WORKFLOW.md phase, or empty>",
            "needs": ["<tools or capabilities it needs>"], "depends_on": ["<earlier ids>"],
            "verify": [{"kind": "tool_succeeded", "tool": "<tool>"} | {"kind": "tests_pass"} |
                       {"kind": "produces", "what": "..."} | {"kind": "judge", "what": "..."}]}]}
