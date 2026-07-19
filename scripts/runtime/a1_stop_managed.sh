#!/usr/bin/env bash
# Configuration-independent fallback shutdown for resources created by this repo.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_services.sh"
source "${ROOT}/scripts/runtime/a1_tmux.sh"
source "${ROOT}/scripts/runtime/a1_processes.sh"

process_exclusions=()
if [[ "${1:-}" == "--keep-camera-monitor" ]]; then
  process_exclusions+=("a1-camera-web")
  shift
fi
if (( $# != 0 )); then
  a1_fail "Unknown managed-stop argument: $1"
  exit 2
fi

status=0
a1_process_stop_all_managed 5 "${process_exclusions[@]}" || status=1
a1_tmux_stop_all_managed || status=1
a1_remove_all_managed_containers || status=1
if (( status != 0 )); then
  a1_fail "Some marked Galaxea A1 runtime resources could not be stopped."
  exit "${status}"
fi
a1_success "All marked Galaxea A1 runtime resources stopped."
