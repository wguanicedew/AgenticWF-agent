---
name: report_check
summary: Say how the check went, with its numbers.
input_schema:
  type: object
  properties:
    status:
      type: string
      enum: [ok, issues, degraded, unable]
      description: "ok: nothing wrong; issues: problems reported; degraded: partly checked; unable: could not check"
    summary: {type: string, description: One or two sentences for the shift}
    metrics:
      type: array
      description: The numbers the check records (only those its instructions declare)
      items:
        type: object
        properties:
          name: {type: string}
          value: {type: number}
          labels:
            type: object
            additionalProperties: {type: string}
          link: {type: string, description: The original data this number comes from}
        required: [name, value]
  required: [status]
---
Say how this check went, once, at the end: `ok` when you looked and found nothing wrong,
`issues` when you reported problems, `degraded` when you could only check part of it, `unable`
when the services or tools you need did not answer (then nothing is concluded from this run).
Record the check's numbers in `metrics`, each with the link to where it came from.
