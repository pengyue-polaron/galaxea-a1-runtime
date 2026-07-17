#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_tmux.sh"

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  a1_usage "$0 <base-runtime> <model-session> -- <bridge-command> [args...]" >&2
  exit 2
fi

BASE_RUNTIME="$1"
MODEL_SESSION="$2"
shift 3
CLEANED_UP=false

cleanup() {
  if [[ "${CLEANED_UP}" == "true" ]]; then
    return
  fi
  CLEANED_UP=true
  a1_cleanup "EEF policy bridge ended; locking and stopping the A1 runtime and policy server."
  if ! "${BASE_RUNTIME}" stop >/dev/null 2>&1; then
    a1_fail "Cleanup failed to stop the A1 runtime: ${BASE_RUNTIME}"
  fi
  a1_tmux_stop "${MODEL_SESSION}"
}

trap cleanup EXIT HUP INT TERM

if "$@"; then
  rc=0
else
  rc=$?
fi
cleanup
trap - EXIT HUP INT TERM
echo "BRIDGE_EXIT=${rc}"
exit "${rc}"
