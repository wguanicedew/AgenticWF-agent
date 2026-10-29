---
name: search_code
summary: Search the repository for a symbol, text or file name.
tags: [code, search, read-only]
input_schema:
  type: object
  properties:
    query:
      type: string
      description: Text, regular expression or symbol to find
  required: [query]
scopes: [code:read]
effects: none
approval: auto
idempotent: true
parallel_safe: true
timeout: 30
handler:
  kind: builtin
  ref: demo.search_code
---
Search the repository for files, functions, classes, symbols or text, and return the matching
files.

Use when:
- the symbol's location is unknown
- searching across the repository

Do not use when:
- the exact file path is already known (use `read_file`)
