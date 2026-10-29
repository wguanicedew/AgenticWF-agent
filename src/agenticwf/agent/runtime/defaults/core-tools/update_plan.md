---
name: update_plan
summary: Record the current plan and WORKFLOW.md phase.
input_schema:
  type: object
  properties:
    phase:
      type: string
      description: Current phase from WORKFLOW.md, e.g. RESEARCH
    steps:
      type: array
      items:
        type: object
        properties:
          text: {type: string}
          status: {type: string, enum: [pending, in_progress, done]}
        required: [text]
---
Record your plan for this task and the workflow phase you are in. The user sees it. Call it when
the plan or phase changes.
