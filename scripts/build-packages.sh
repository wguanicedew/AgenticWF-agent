#!/usr/bin/env bash
# Build the pip packages of this repository into dist/:
#   server   agenticwf-agent-server   the repository root: the service (api, core, runtime) and its
#                                     commands (agenticwf-agent, agenticwf-agent-worker)
#   client   agenticwf-agent-client   client/: AgentClient and the `agenticwf agent` command
#
#   scripts/build-packages.sh                 both wheels
#   scripts/build-packages.sh client          one of them (client or server)
#   SDIST=1 scripts/build-packages.sh         source distributions too (needs `pip install build`)
#   OUT=/tmp/wheels PYTHON=python3.11 scripts/build-packages.sh
#
# Install, e.g. on a laptop: pip install dist/agenticwf_auth_client-*.whl dist/agenticwf_agent_client-*.whl
# (agenticwf-auth-client from the AgenticWF-auth repository, scripts/build-packages.sh client).
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
out=${OUT:-$root/dist}
python=${PYTHON:-python3}

source_of() {
    case $1 in
        server) echo "$root" ;;
        client) echo "$root/client" ;;
        *) echo "unknown package '$1' (client or server)" >&2; return 1 ;;
    esac
}

[[ $# -gt 0 ]] || set -- server client
mkdir -p "$out"
for name in "$@"; do
    src=$(source_of "$name")
    echo "[build-packages] $name: $src"
    # setuptools copies into build/lib and never removes from it: a module moved or deleted since
    # the last build would still be packaged.
    rm -rf "$src/build"
    "$python" -m pip wheel --quiet --no-deps --wheel-dir "$out" "$src"
    if [[ ${SDIST:-0} == 1 ]]; then
        "$python" -m build --sdist --outdir "$out" "$src"
    fi
done
ls -1 "$out"
