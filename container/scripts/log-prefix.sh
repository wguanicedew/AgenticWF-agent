#!/usr/bin/env bash
# Run a supervisord program with "[NAME] " before each line of its stdout and stderr, so the
# programs can be told apart in `docker logs` (supervisord.d/*.ini: `log-prefix.sh NAME CMD...`).
# The command is exec'd, so supervisord still signals it directly. sed ignores INT/TERM so that
# stopasgroup does not kill it before the program's last lines; it exits at EOF.
set -euo pipefail
prefix=${1:?usage: log-prefix.sh NAME COMMAND [ARG...]}
shift
exec > >(trap '' INT TERM; exec sed -u "s/^/[${prefix}] /") 2>&1
exec "$@"
