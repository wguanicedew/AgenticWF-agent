---
name: edit_file
summary: Replace an exact piece of text in a file.
tags: [code, file, edit]
input_schema:
  type: object
  properties:
    path:
      type: string
    old_text:
      type: string
      description: Exact text to replace; must occur once
    new_text:
      type: string
  required: [path, old_text, new_text]
scopes: [code:write]
effects: local
approval: auto
idempotent: false
timeout: 30
handler:
  kind: builtin
  ref: demo.edit_file
---
Modify a file by replacing existing text with new text. `old_text` must match exactly once.

Preconditions:
- the file has already been read
- the intended modification is understood
