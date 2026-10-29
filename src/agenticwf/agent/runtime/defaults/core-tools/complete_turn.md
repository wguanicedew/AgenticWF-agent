---
name: complete_turn
summary: Finish the current turn of the plan, with its result.
input_schema:
  type: object
  properties:
    summary: {type: string}
    outputs: {type: object}
  required: [summary]
---
Call this when the goal of the current turn is reached. Give a short summary of what you found or
did, and any structured outputs later turns need. The runtime checks the turn before the next one
starts.
