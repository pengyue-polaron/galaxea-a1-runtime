#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH=""
PROCESS_NAME="a1-camera-web"
source "${ROOT}/scripts/runtime/a1_config.sh"
source "${ROOT}/scripts/runtime/a1_processes.sh"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    a1_usage "$0 [--config <path>] [stop|status|logs]" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
VIEWER_SCRIPT="${ROOT}/scripts/apps/cameras/a1_camera_web.py"
LOG_FILE="${A1_PROCESS_STATE_ROOT}/${PROCESS_NAME}.log"
CONFIG_STATE_FILE="${A1_PROCESS_STATE_ROOT}/${PROCESS_NAME}.config"
BRIDGE_SOCKET="${A1_PROCESS_STATE_ROOT}/a1-camera-bridge.sock"

load_camera_config() {
  local config_args=(--repo-root "${ROOT}" --shell)
  if [[ -n "${CONFIG_PATH}" ]]; then
    config_args+=("${CONFIG_PATH}")
  fi
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.configuration.system "${config_args[@]}"
}

require_command() {
  local command="$1"
  if ! command -v "${command}" >/dev/null 2>&1; then
    a1_fail "Persistent camera monitor requires ${command}."
    return 2
  fi
}

health_url() {
  printf 'http://127.0.0.1:%s/healthz\n' "${WEB_PREVIEW_PORT}"
}

viewer_endpoint_present() {
  local payload
  payload="$(curl -sS --max-time 1 "$(health_url)" 2>/dev/null)" || return 1
  [[ "${payload}" == *'"streams"'* && "${payload}" == *'"agent"'* && "${payload}" == *'"wrist"'* ]]
}

viewer_is_healthy() {
  [[ -S "${BRIDGE_SOCKET}" ]] && \
    curl -fsS --max-time 1 "$(health_url)" >/dev/null 2>&1
}

wait_for_viewer() {
  local timeout_s="${WEB_PREVIEW_STARTUP_TIMEOUT_S%.*}"
  local deadline=$((SECONDS + timeout_s))
  while (( SECONDS < deadline )); do
    if viewer_is_healthy; then
      return 0
    fi
    if ! a1_process_is_running "${PROCESS_NAME}"; then
      return 1
    fi
    sleep 0.2
  done
  return 1
}

stop_viewer() {
  load_camera_config
  require_command curl
  if ! a1_process_is_running "${PROCESS_NAME}" && viewer_endpoint_present; then
    a1_fail "Camera Web endpoint is owned by an unmanaged process."
    return 2
  fi
  a1_process_stop "${PROCESS_NAME}" "${WEB_PREVIEW_SHUTDOWN_TIMEOUT_S}"
  rm -f "${CONFIG_STATE_FILE}"
  a1_success "Persistent camera monitor stopped."
}

ensure_viewer() {
  load_camera_config
  require_command curl
  require_command ss
  if a1_process_is_running "${PROCESS_NAME}"; then
    local active_config=""
    if [[ -f "${CONFIG_STATE_FILE}" ]]; then
      IFS= read -r active_config <"${CONFIG_STATE_FILE}" || true
    fi
    if [[ "${active_config}" != "${SYSTEM_CONFIG_PATH}" ]]; then
      a1_fail "Camera monitor config identity mismatch: active=${active_config:-missing}, requested=${SYSTEM_CONFIG_PATH}."
      return 2
    fi
    if wait_for_viewer; then
      a1_success "Persistent Camera Bridge is already ready: http://${WEB_PREVIEW_BIND}:${WEB_PREVIEW_PORT}"
      return 0
    fi
    a1_warn "Marked camera monitor is unhealthy; restarting it."
    a1_process_stop "${PROCESS_NAME}" "${WEB_PREVIEW_SHUTDOWN_TIMEOUT_S}" || return
    rm -f "${CONFIG_STATE_FILE}"
  elif ss -H -ltn "sport = :${WEB_PREVIEW_PORT}" | grep -q .; then
    if viewer_endpoint_present; then
      a1_fail "Camera Web endpoint exists without its marked persistent owner."
      return 2
    fi
    a1_fail "Web preview port ${WEB_PREVIEW_PORT} is occupied by an unmanaged service."
    return 2
  else
    rm -f "${CONFIG_STATE_FILE}"
  fi

  local command=("${PYTHON_BIN}" "${VIEWER_SCRIPT}")
  if [[ -n "${CONFIG_PATH}" ]]; then
    command+=(--config "${CONFIG_PATH}")
  else
    command+=(--config "${SYSTEM_CONFIG_PATH}")
  fi
  a1_process_start \
    "${PROCESS_NAME}" "${ROOT}" "${LOG_FILE}" \
    env PYTHONUNBUFFERED=1 "PYTHONPATH=${ROOT}:${PYTHONPATH:-}" "${command[@]}"
  mkdir -p "$(dirname "${CONFIG_STATE_FILE}")"
  local config_state_tmp="${CONFIG_STATE_FILE}.tmp.$$"
  printf '%s\n' "${SYSTEM_CONFIG_PATH}" >"${config_state_tmp}"
  mv "${config_state_tmp}" "${CONFIG_STATE_FILE}"
  if ! wait_for_viewer; then
    tail -n 120 "${LOG_FILE}" >&2 || true
    a1_process_stop "${PROCESS_NAME}" "${WEB_PREVIEW_SHUTDOWN_TIMEOUT_S}" || true
    rm -f "${CONFIG_STATE_FILE}"
    a1_fail "Persistent Camera Bridge did not become healthy within ${WEB_PREVIEW_STARTUP_TIMEOUT_S}s."
    return 2
  fi
  a1_success "Persistent Camera Bridge ready: http://${WEB_PREVIEW_BIND}:${WEB_PREVIEW_PORT}"
}

show_status() {
  load_camera_config
  require_command curl
  if a1_process_is_running "${PROCESS_NAME}"; then
    a1_process_status "${PROCESS_NAME}"
    if viewer_is_healthy; then
      a1_success "Raw Camera Bridge and Web streams are healthy: http://${WEB_PREVIEW_BIND}:${WEB_PREVIEW_PORT}"
      return 0
    fi
    a1_fail "Camera monitor process is running but its streams are unhealthy."
    return 1
  fi
  if viewer_endpoint_present; then
    a1_fail "Camera Web endpoint exists without its marked persistent owner."
    return 1
  fi
  a1_info "Persistent camera monitor is not running."
  return 1
}

case "${1:-}" in
  "") ensure_viewer ;;
  stop) stop_viewer ;;
  status) show_status ;;
  logs)
    if [[ -f "${LOG_FILE}" ]]; then
      tail -n 160 "${LOG_FILE}"
    else
      a1_info "No persistent camera monitor log: ${LOG_FILE}"
    fi
    ;;
  help|-h|--help)
    a1_usage "$0 [--config <path>] [stop|status|logs]"
    ;;
  *)
    a1_fail "Unknown camera-web command: ${1:-}"
    a1_usage "$0 [--config <path>] [stop|status|logs]" >&2
    exit 2
    ;;
esac
