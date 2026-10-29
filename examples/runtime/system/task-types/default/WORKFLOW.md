# Workflow

The agent operates in these phases:

## PLAN

Goal:
Understand the requested outcome.

Allowed actions:
- inspect task
- inspect repository structure
- formulate plan

Transition to RESEARCH when:
- required information is missing

Transition to EXECUTE when:
- sufficient information is available


## RESEARCH

Goal:
Gather evidence required to solve the task.

Preferred tools:
- search_code
- read_file
- web_search

Do not modify files during this phase.

Transition to EXECUTE when:
- probable solution has been identified


## EXECUTE

Goal:
Apply changes.

Allowed tools:
- read_file
- edit_file
- run_command

After modifying code:
transition to VERIFY.


## VERIFY

Goal:
Determine whether the requested outcome was achieved.

Required:
- run relevant tests
- inspect failures

If verification fails:
transition back to RESEARCH or EXECUTE.

If verification succeeds:
transition to DONE.
