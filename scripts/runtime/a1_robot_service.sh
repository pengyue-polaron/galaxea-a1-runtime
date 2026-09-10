#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
source "${ROOT}/scripts/runtime/a1_processes.sh"

SYSTEM_CONFIG_PATH="${A1_SYSTEM_CONFIG_PATH:-}"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

if [[ "${1:-help}" != "stop" && "${1:-help}" != "logs" ]]; then
  config_args=(--repo-root "${ROOT}" --shell)
  if [[ -n "${SYSTEM_CONFIG_PATH}" ]]; then
    config_args+=("${SYSTEM_CONFIG_PATH}")
  fi
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.configuration.system "${config_args[@]}"
fi

PROCESS_NAME="a1-robot-service"
LOG_FILE="${A1_PROCESS_STATE_ROOT}/logs/${PROCESS_NAME}.log"

start_service() {
  if a1_process_is_running "${PROCESS_NAME}"; then
    a1_success "A1 Robot service is already running at ${A1_ROBOT_SERVICE_ENDPOINT}."
    return
  fi
  local socket_path="${A1_ROBOT_SERVICE_ENDPOINT#unix://}"
  if [[ "${socket_path}" == "${A1_ROBOT_SERVICE_ENDPOINT}" || -e "${socket_path}" ]]; then
    a1_fail "A1 Robot service endpoint is already occupied: ${A1_ROBOT_SERVICE_ENDPOINT}"
    return 2
  fi
  a1_process_start \
    "${PROCESS_NAME}" "${ROOT}" "${LOG_FILE}" \
    env PYTHONUNBUFFERED=1 \
      PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.robot_service.server \
      --system-config "${SYSTEM_CONFIG_PATH}"

  local deadline=$((SECONDS + ${A1_ROBOT_SERVICE_SERVER_STARTUP_TIMEOUT_S%.*}))
  while (( SECONDS < deadline )); do
    if ! a1_process_is_running "${PROCESS_NAME}"; then
      a1_fail "A1 Robot service exited during startup. Log: ${LOG_FILE}"
      tail -n 120 "${LOG_FILE}" >&2 || true
      return 2
    fi
    if grep -Fq "A1 Robot service ready at ${A1_ROBOT_SERVICE_ENDPOINT}" "${LOG_FILE}"; then
      a1_success "A1 Robot service ready at ${A1_ROBOT_SERVICE_ENDPOINT}."
      return
    fi
    sleep 0.1
  done
  a1_process_stop "${PROCESS_NAME}" "${A1_ROBOT_SERVICE_SERVER_SHUTDOWN_TIMEOUT_S}" || true
  a1_fail "A1 Robot service did not become ready. Log: ${LOG_FILE}"
  tail -n 120 "${LOG_FILE}" >&2 || true
  return 2
}

case "${1:-help}" in
  start)
    start_service
    ;;
  stop)
    a1_process_stop "${PROCESS_NAME}" "${A1_ROBOT_SERVICE_SERVER_SHUTDOWN_TIMEOUT_S:-5}"
    ;;
  status)
    a1_process_status "${PROCESS_NAME}"
    ;;
  logs)
    tail -n "${A1_LOG_TAIL:-120}" "${LOG_FILE}" 2>/dev/null || true
    ;;
  *)
    a1_usage "$0 <start|stop|status|logs>"
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown A1 Robot service command: ${1:-}"
      exit 2
    fi
    ;;
esac
