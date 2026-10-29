---
name: run_tests
summary: Run the project's automated tests and report passes and failures.
tags: [code, test, verify]
input_schema:
  type: object
  properties:
    selection:
      type: string
      description: Optional test selection, e.g. a file or test name
scopes: [code:read]
effects: none
approval: auto
idempotent: true
timeout: 600
handler:
  kind: builtin
  ref: demo.run_tests
---
Run the project's automated test suite.

Use:
- after modifying code
- during verification
