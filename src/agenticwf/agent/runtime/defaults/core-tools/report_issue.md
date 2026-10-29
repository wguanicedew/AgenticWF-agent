---
name: report_issue
summary: Report one problem this check found.
input_schema:
  type: object
  properties:
    key:
      type: object
      description: 'The key fields of the check (its instructions name them), e.g. {"site": "...", "error_class": "..."}' 
      additionalProperties: {type: string}
    title: {type: string, description: One line that says what is wrong}
    severity: {type: string, description: One of the check's severities}
    details: {type: string, description: 'Markdown: what you saw, error samples, numbers'}
    links:
      type: array
      items: {type: string}
      description: Links to the original data (monitoring pages, job or transfer lists, logs)
    evidence: {type: object, description: Small structured data backing the finding}
    related:
      type: array
      items: {type: string}
      description: Ids of other open issues that likely share this problem's cause
    draft_ticket: {type: string, description: If a ticket is needed, its text for a person to file}
  required: [key, title, severity]
---
Report one problem you found in this check. Call it once per problem, with the same key values
whenever you see the same problem again: that is how the problem is followed from run to run. Do
not report what a suppression or an ignored issue already covers. Always give links to the
original data.
