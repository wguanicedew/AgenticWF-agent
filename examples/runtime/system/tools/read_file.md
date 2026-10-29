---
name: read_file
summary: Read the contents of a file whose path is known.
tags: [code, file, read-only]
input_schema:
  type: object
  properties:
    path:
      type: string
      description: Path relative to the repository root
  required: [path]
scopes: [code:read]
effects: none
approval: auto
idempotent: true
parallel_safe: true
timeout: 30
handler:
  kind: builtin
  ref: demo.read_file
---
Read a file and return its contents.

Use when:
- the exact path is known
- source must be inspected before editing

Use `search_code` first when the path is unknown.
