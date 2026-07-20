#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
source "${ROOT}/scripts/runtime/a1_tmux.sh"
A1_REPO_PYTHONPATH="$(a1_repo_pythonpath "${ROOT}")"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
CONFIG_PATH=""

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
SETUP_SCRIPT="${ROOT}/scripts/apps/pi05/setup_pi05_inference.py"
VERIFY_SCRIPT="${ROOT}/scripts/apps/pi05/verify_pi05_inference.py"
SMOKE_SCRIPT="${ROOT}/scripts/apps/pi05/smoke_pi05_inference.py"
PROBE_SCRIPT="${ROOT}/scripts/apps/pi05/probe_pi05_server.py"
SERVER_SCRIPT="${ROOT}/scripts/apps/pi05/pi05_policy_server.py"
BRIDGE_SCRIPT="${ROOT}/scripts/apps/pi05/pi05_ee_bridge.py"
BRIDGE_GUARD="${ROOT}/scripts/apps/a1_eef_policy_bridge_guard.sh"
CAMERA_RUNTIME="${ROOT}/scripts/apps/cameras/a1_camera_web_runtime.sh"
SELECTED_TASK_ID=""

config_args=(--repo-root "${ROOT}" --shell)
if [[ -n "${CONFIG_PATH}" ]]; then
  config_args+=("${CONFIG_PATH}")
fi
a1_load_shell_config env \
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
  -m galaxea_a1_runtime.apps.pi05.config "${config_args[@]}"

verify_inference() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${VERIFY_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

setup_inference() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${SETUP_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

probe_server() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${PROBE_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

start_server() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    a1_fail "pi0.5 deployment.ready=false."
    exit 2
  fi
  verify_inference
  if ss -H -ltn "sport = :${MODEL_PORT}" | grep -q .; then
    if a1_tmux_has_session "${MODEL_SESSION}"; then
      probe_server
      a1_info "OpenPI pi0.5 server is already running in ${MODEL_SESSION}."
      return
    fi
    a1_fail "${MODEL_HOST}:${MODEL_PORT} is occupied by an unmanaged process."
    exit 2
  fi

  local command=(
    "${MODEL_PYTHON}" "${SERVER_SCRIPT}"
    --repo-root "${ROOT}"
    --config "${CONFIG_PATH}"
  )
  local command_q
  printf -v command_q "%q " "${command[@]}"
  a1_tmux_start "${MODEL_SESSION}" "${MODEL_CHECKOUT}" \
    "bash -lc 'export PYTHONPATH=\"${A1_REPO_PYTHONPATH}:\${PYTHONPATH:-}\"; ${command_q}; rc=\$?; echo SERVER_EXIT=\$rc; exec bash'"

  local timeout_s="${MODEL_STARTUP_TIMEOUT%.*}"
  if ! a1_tmux_wait_for_http_health \
    "${MODEL_SESSION}" "http://${MODEL_HOST}:${MODEL_PORT}/healthz" \
    "SERVER_EXIT=" "OpenPI pi0.5 server" "${timeout_s}" 120; then
    exit 2
  fi
  probe_server
  a1_success "OpenPI pi0.5 server is listening on ${MODEL_HOST}:${MODEL_PORT}."
}

smoke_inference() {
  start_server
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    "${SMOKE_SCRIPT}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

start_services() {
  "${BASE_RUNTIME}" services
}

select_task() {
  if [[ -n "${SELECTED_TASK_ID}" ]]; then
    return
  fi
  local selected
  if ! selected="$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.task_selection \
      --catalog "${TASK_CATALOG_PATH}"
  )"; then
    a1_fail "Pi0.5 task selection cancelled; no model or motion runtime was started; the camera monitor remains available."
    return 2
  fi
  SELECTED_TASK_ID="${selected}"
  a1_info "Selected Pi0.5 task: ${SELECTED_TASK_ID}"
}

start_bridge() {
  select_task
  "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"
  if [[ ! -x "${BRIDGE_GUARD}" ]]; then
    a1_fail "EEF bridge guard is missing: ${BRIDGE_GUARD}"
    exit 2
  fi
  local bridge_command=(
    "${PYTHON_BIN}" "${BRIDGE_SCRIPT}"
    --config "${CONFIG_PATH}"
    --task-id "${SELECTED_TASK_ID}"
  )
  a1_step "Reading uncompressed frames from the persistent Camera Bridge."
  local guarded_command=(
    "${BRIDGE_GUARD}" "${BASE_RUNTIME}" "${MODEL_SESSION}" --
    "${bridge_command[@]}"
  )
  local guarded_command_q
  printf -v guarded_command_q "%q " "${guarded_command[@]}"
  if ! a1_tmux_start "${SESSION}" "${ROOT}" \
    "bash -lc 'export PYTHONPATH=\"${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:\${PYTHONPATH:-}\"; ${guarded_command_q}; exec bash'"; then
    return 2
  fi
  if ! a1_tmux_verify_startup \
    "${SESSION}" "BRIDGE_EXIT=" "OpenPI pi0.5 bridge" "${TMUX_STARTUP_GRACE_S}" 60; then
    exit 2
  fi
}

start_pipeline() {
  "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"
  select_task
  cleanup_failed_pipeline() {
    local status=$?
    if [[ "${status}" != "0" ]]; then
      a1_cleanup "pi0.5 startup failed; stopping repository-owned partial runtime."
      a1_tmux_stop "${SESSION}"
      a1_tmux_stop "${MODEL_SESSION}"
      "${BASE_RUNTIME}" stop >/dev/null 2>&1 || true
      "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}" >/dev/null 2>&1 || true
    fi
  }
  trap cleanup_failed_pipeline EXIT
  start_server
  start_services
  start_bridge
  trap - EXIT
}

stop_runtime() {
  a1_tmux_stop "${SESSION}"
  a1_tmux_stop "${MODEL_SESSION}"
  "${BASE_RUNTIME}" stop
  "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"
  a1_success "OpenPI pi0.5 bridge, server, and A1 runtime stopped."
}

case "${1:-help}" in
  setup)
    setup_inference
    ;;
  verify)
    verify_inference
    ;;
  server)
    start_server
    ;;
  smoke)
    smoke_inference
    ;;
  start)
    start_pipeline
    a1_success "OpenPI pi0.5 runtime started. Attach with: tmux attach -t ${SESSION}"
    ;;
  services)
    start_services
    ;;
  tmux)
    start_bridge
    ;;
  server-stop)
    a1_tmux_stop "${MODEL_SESSION}"
    ;;
  stop)
    stop_runtime
    ;;
  server-logs|logs)
    a1_tmux_capture "${MODEL_SESSION}" 160
    ;;
  status)
    "${BASE_RUNTIME}" status || true
    a1_tmux_status "${SESSION}" || true
    a1_tmux_status "${MODEL_SESSION}"
    ;;
  *)
    a1_usage "$0 [--config PATH] <setup|verify|start|server|smoke|services|tmux|stop|server-stop|server-logs|status>"
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown pi0.5 command: ${1:-}"
      exit 2
    fi
    ;;
esac
