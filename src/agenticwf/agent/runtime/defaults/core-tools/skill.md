---
name: skill
summary: The tool each skill is offered as (skill__<name>); ${description} is the skill's own.
input_schema:
  type: object
  properties:
    purpose:
      type: string
      description: What you will use the skill for
---
Skill: ${description}
Call it to load the skill's instructions before you do this kind of work; they then guide the rest
of the task.
