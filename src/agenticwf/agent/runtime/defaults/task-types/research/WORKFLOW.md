# Workflow

## ROUTE

Goal: decide where the answer may live.

- Read the question and the knowledge bases listed in your context.
- Decide whether the knowledge bases are likely enough, or the web is needed too (current
  events, public information, things the knowledge bases do not cover).

Transition to KNOWLEDGE.

## KNOWLEDGE

Goal: gather evidence from the knowledge service.

Preferred tools:
- kb_research (preset auto)
- kb_search, kb_grep, kb_read for exact details

Transition to WEB when the evidence is missing, incomplete or out of date, or the question
needs public or current information. Otherwise transition to ANSWER.

## WEB

Goal: fill the gaps from the public web.

Preferred tools:
- web_search with queries that contain no private details (provider `auto`, or the one that
  suits the question)
- web_fetch on the results you will rely on

Transition to ANSWER.

## ANSWER

Goal: one merged answer.

- Combine the knowledge-base and web evidence; note disagreements and which source is newer.
- Cite every claim (`kb:…`, `web:…`) and end with a Sources list giving each source's full
  citation.
- Say what none of the sources covers.
