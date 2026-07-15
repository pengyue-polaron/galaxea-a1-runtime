#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_tmux.sh"

if (( $# < 4 )) || [[ "$3" != "--" ]]; then
  echo "Usage: $0 <base-runtime> <model-session> -- <bridge-command> [args...]" >&2
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
  echo "[CLEANUP] LingBot bridge ended; locking and stopping the A1 runtime and policy server."
  if ! "${BASE_RUNTIME}" stop >/dev/null 2>&1; then
    echo "[CLEANUP ERROR] Failed to stop the A1 runtime: ${BASE_RUNTIME}" >&2
  fi
  a1_tmux_stop "${MODEL_SESSION}"
}

trap cleanup EXIT HUP INT TERM

"$@"
rc=$?
cleanup
trap - EXIT HUP INT TERM
echo "BRIDGE_EXIT=${rc}"
exit "${rc}"
