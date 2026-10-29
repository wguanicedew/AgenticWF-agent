# Agent Instructions

## General workflow

For every engineering task:

1. Understand the request.
2. Inspect relevant files before making assumptions.
3. Create a short plan when the task is multi-step.
4. Execute the plan using available tools.
5. Run appropriate tests after modifications.
6. If tests fail, investigate and repair.
7. Do not finish until verification succeeds.

## Tool routing

Use `search_code` when:
- the location of code is unknown
- you need to locate a symbol

Use `read_file` when:
- the file path is already known

Use `edit_file` only after:
- the relevant source has been read

Use `run_tests` after:
- modifying source code

## Safety

Ask for confirmation before:
- deleting files
- changing production configuration
- installing production dependencies
- performing irreversible actions
