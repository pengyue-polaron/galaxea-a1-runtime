#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_console.sh"

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
RESET_APP="${ROOT}/scripts/apps/reset/a1_reset.py"
SYSTEM_CONFIG_PATH="${ROOT}/configs/system/a1.toml"
SYSTEM_CONFIG_SET=false
POSE_PATH=""

while (( $# > 0 )); do
  case "$1" in
    --system-config)
      if [[ "${SYSTEM_CONFIG_SET}" == "true" || -z "${2:-}" ]]; then
        a1_fail "--system-config requires one path and may be provided only once."
        exit 2
      fi
      SYSTEM_CONFIG_PATH="$2"
      SYSTEM_CONFIG_SET=true
      shift 2
      ;;
    --pose)
      if [[ -n "${POSE_PATH}" || -z "${2:-}" ]]; then
        a1_fail "--pose requires one path and may be provided only once."
        exit 2
      fi
      POSE_PATH="$2"
      shift 2
      ;;
    -h|--help)
      a1_usage "$0 --pose PATH [--system-config PATH]"
      exit 0
      ;;
    *)
      a1_fail "Unknown A1 reset argument: $1"
      exit 2
      ;;
  esac
done

if [[ -z "${POSE_PATH}" ]]; then
  a1_fail "--pose is required."
  exit 2
fi

cd "${ROOT}"
a1_step "Validating the tracked A1 reset before runtime startup."
"${PYTHON_BIN}" "${RESET_APP}" \
  --system-config "${SYSTEM_CONFIG_PATH}" \
  --pose "${POSE_PATH}" \
  --validate-only

runtime_started=false
cleanup_reset_runtime() {
  local status=$?
  trap - EXIT HUP INT TERM
  if [[ "${runtime_started}" == "true" ]]; then
    a1_cleanup "Locking and stopping the A1 reset runtime."
    if ! A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}" \
      "${BASE_RUNTIME}" stop; then
      a1_fail "A1 reset cleanup did not stop every owned runtime resource."
      if (( status == 0 )); then
        status=1
      fi
    fi
  fi
  exit "${status}"
}
trap cleanup_reset_runtime EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

a1_step "Starting the staged A1 reset runtime."
runtime_started=true
A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}" "${BASE_RUNTIME}" services

a1_step "Moving A1 to the tracked reset pose."
"${PYTHON_BIN}" "${RESET_APP}" \
  --system-config "${SYSTEM_CONFIG_PATH}" \
  --pose "${POSE_PATH}"
