---
name: delete_file
summary: Delete a file from the repository.
tags: [code, file, delete]
input_schema:
  type: object
  properties:
    path:
      type: string
  required: [path]
scopes: [code:write]
effects: local
# AGENTS.md says "ask for confirmation before deleting files"; this makes it a guarantee.
approval: ask
idempotent: true
timeout: 30
handler:
  kind: builtin
  ref: demo.delete_file
---
Delete a file. The user is asked to approve every deletion.
