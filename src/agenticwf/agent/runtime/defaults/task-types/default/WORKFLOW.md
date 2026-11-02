# Workflow

The agent works in these phases:

## PLAN

Goal:
Understand the requested outcome.

Transition to RESEARCH when:
- required information is missing

Transition to EXECUTE when:
- sufficient information is available


## RESEARCH

Goal:
Gather the evidence required to answer or act.

Prefer the tools listed for this task; use `search_tools` when none fits.
Do not change anything during this phase.

Transition to EXECUTE when:
- the answer or the action is clear


## EXECUTE

Goal:
Answer, or carry out the change, with the listed tools.


## VERIFY

Goal:
Check that the requested outcome was achieved: the answer covers the question and cites its
sources, or the change did what was asked.

If it was not:
transition back to RESEARCH or EXECUTE.

If it was:
transition to DONE.
