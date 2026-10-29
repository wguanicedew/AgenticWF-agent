---
name: read_skill_file
summary: Read one of the running skill's files.
input_schema:
  type: object
  properties:
    path:
      type: string
      description: The file, as the skill lists it
  required: [path]
---
Read a file that belongs to the skill you are using (its instructions list them), such as a
reference document or a template.
