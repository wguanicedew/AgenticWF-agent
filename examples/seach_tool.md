# Tool Selection Rules

## search_code

Use when:
- symbol location is unknown
- searching across repository

Do not use when:
- exact file path is already known


## read_file

Use when:
- exact path is known
- source must be inspected before editing


## edit_file

Preconditions:
- file has already been read
- intended modification is understood


## run_tests

Use:
- after code modification
- during verification
