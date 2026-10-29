---
name: memory_search
summary: Search your notes from earlier tasks.
input_schema:
  type: object
  properties:
    query:
      type: string
      description: What you want to recall
    limit:
      type: integer
      minimum: 1
      maximum: 20
  required: [query]
---
Search the long-term memory of the person you work for: notes kept from their earlier tasks. The
most relevant few are already in your context; call this for more. Notes may be outdated: check
them before you rely on them.
