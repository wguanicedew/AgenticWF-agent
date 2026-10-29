---
name: ask_user
summary: Ask the user a question and wait for the answer.
input_schema:
  type: object
  properties:
    question: {type: string}
  required: [question]
---
Ask the user for information you cannot find yourself. The task pauses until they reply.
