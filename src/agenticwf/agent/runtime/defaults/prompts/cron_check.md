Run the check `${check}` of `${cron}`: ${description}

It is ${now} (${timezone}). Look at what happened ${window}.

Report every problem you find with `report_issue`. Its key fields are: ${key}. Use the same
values whenever you see the same problem again (the site as the errors name it). Severity is one
of: ${severities} (the first is the most severe). Give links to the original data. End with
`report_check`: the status, a one-line summary${metrics}.

Issues this check is following (say whether each is still there by reporting it again; one you
no longer see is solved by itself after a few runs):
${open_issues}

Other checks' open issues at the same sites (if one cause explains several, name their ids in
`related`):
${other_issues}

Not followed ("won't follow" rules and ignored issues; do not report these):
${suppressions}
