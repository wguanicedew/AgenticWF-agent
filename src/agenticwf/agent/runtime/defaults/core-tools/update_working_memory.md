---
name: update_working_memory
summary: Update the goal, task state or notes in working memory.
input_schema:
  type: object
  properties:
    goal: {type: string}
    task_state: {type: string}
    add_note: {type: string}
---
Keep your working memory current: the goal (requested outcome and acceptance criteria), the task
state (done / in progress / blocked) and notes (decisions, open questions). Working memory is never
summarized away.
