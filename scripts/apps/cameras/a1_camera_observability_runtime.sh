#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CAMERA_RUNTIME="${ROOT}/scripts/apps/cameras/a1_camera_web_runtime.sh"
OBSERVABILITY_RUNTIME="${ROOT}/scripts/runtime/a1_observability_runtime.sh"
CONFIG_PATH=""

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "[FAIL] --config requires a path." >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

camera_command() {
  local args=()
  if [[ -n "${CONFIG_PATH}" ]]; then
    args+=(--config "${CONFIG_PATH}")
  fi
  "${CAMERA_RUNTIME}" "${args[@]}" "$@"
}

observability_command() {
  A1_SYSTEM_CONFIG_PATH="${CONFIG_PATH}" "${OBSERVABILITY_RUNTIME}" "$@"
}

start_monitoring() {
  local camera_was_ready=false
  if camera_command status >/dev/null 2>&1; then
    camera_was_ready=true
  fi
  cleanup_failed_start() {
    if [[ "${camera_was_ready}" != "true" ]]; then
      camera_command stop >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_failed_start ERR
  camera_command start
  observability_command start
  trap - ERR
  echo "[PASS] Persistent cameras and read-only Foxglove monitoring are ready."
}

stop_monitoring() {
  local result=0
  observability_command stop || result=$?
  camera_command stop || result=$?
  return "${result}"
}

status_monitoring() {
  local result=0
  camera_command status || result=$?
  observability_command status || result=$?
  return "${result}"
}

logs_monitoring() {
  camera_command logs || true
  observability_command logs || true
}

case "${1:-start}" in
  start) start_monitoring ;;
  stop) stop_monitoring ;;
  status) status_monitoring ;;
  logs) logs_monitoring ;;
  help|-h|--help)
    echo "Usage: $0 [--config <path>] [start|stop|status|logs]"
    echo "Starts persistent cameras and the shared read-only Foxglove stack together."
    ;;
  *)
    echo "[FAIL] Unknown camera monitoring command: ${1:-}" >&2
    exit 2
    ;;
esac
