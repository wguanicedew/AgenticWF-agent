---
name: search_tools
summary: Find more tools by describing what you need.
input_schema:
  type: object
  properties:
    query: {type: string}
  required: [query]
---
Search the tools available to this task that are not loaded yet. Matching tools become callable
from your next step.
