#!/usr/bin/env bash
# supervisord event listener (supervisord.d/fatal.ini): when a program gives up (FATAL, after its
# start retries), stop supervisord and so the container. A container that stays up with a dead
# agent looks healthy to `docker run --restart`; one that exits is restarted by that policy.
# stdout is supervisord's event protocol; messages go to stderr.
set -uo pipefail
printf 'READY\n'
while IFS= read -r header; do
    len=$(sed -n 's/.*len:\([0-9]*\).*/\1/p' <<<"$header")
    payload=$(head -c "${len:-0}")
    echo "[supervisor-fatal] a program gave up (${payload}); stopping the container" >&2
    printf 'RESULT 2\nOK'
    kill -TERM "$(cat /run/supervisord.pid)" 2>/dev/null || kill -TERM 1
    printf 'READY\n'
done
