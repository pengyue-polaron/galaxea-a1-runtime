#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
ROBOT_SERVICE="${ROOT}/scripts/runtime/a1_robot_service.sh"
CAMERA_RUNTIME="${ROOT}/scripts/apps/cameras/a1_camera_web_runtime.sh"
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
config_args=(--repo-root "${ROOT}" --shell)
if [[ -n "${CONFIG_PATH}" ]]; then
  config_args+=("${CONFIG_PATH}")
fi
a1_load_shell_config env \
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
  -m galaxea_a1_runtime.apps.tfp.config "${config_args[@]}"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"

run_root_module() {
  local module="$1"
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m "${module}" --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

ensure_camera_monitor() {
  "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"
}

cleanup_pipeline() {
  local status=$?
  trap - EXIT HUP INT TERM
  a1_cleanup "TFP run ended; locking and stopping its A1 execution services."
  "${ROBOT_SERVICE}" stop >/dev/null 2>&1 || true
  "${BASE_RUNTIME}" stop >/dev/null 2>&1 || true
  ensure_camera_monitor >/dev/null 2>&1 || true
  return "${status}"
}

run_pipeline() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    a1_fail "TFP deployment.ready=false."
    return 2
  fi
  # Artifact/config/GPU verification must complete before any hardware owner starts.
  run_root_module galaxea_a1_runtime.apps.tfp.verify
  ensure_camera_monitor
  trap cleanup_pipeline EXIT HUP INT TERM
  "${BASE_RUNTIME}" services
  if [[ "${TFP_RESET_BEFORE_RUN}" == "1" ]]; then
    a1_step "Resetting A1 to the tracked TFP training start pose."
    PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
      "${PYTHON_BIN}" "${ROOT}/scripts/apps/reset/a1_reset.py" \
      --system-config "${SYSTEM_CONFIG_PATH}" \
      --pose "${TFP_RESET_CONFIG}"
  fi
  "${ROBOT_SERVICE}" start
  a1_success "TFP runtime ready; the first valid command activates the guarded relay."
  a1_info "AgentView dashboard: http://${WEB_PREVIEW_BIND}:${WEB_PREVIEW_PORT}"
  env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${A1_REPO_PYTHONPATH}:${PYTHONPATH:-}" \
    "${MODEL_PYTHON}" -m galaxea_a1_runtime.apps.tfp.rollout \
      --repo-root "${ROOT}" --config "${CONFIG_PATH}"
}

stop_runtime() {
  local status=0
  "${ROBOT_SERVICE}" stop || status=$?
  "${BASE_RUNTIME}" stop || status=$?
  ensure_camera_monitor || status=$?
  return "${status}"
}

case "${1:-help}" in
  setup)
    run_root_module galaxea_a1_runtime.apps.tfp.setup
    ;;
  verify)
    run_root_module galaxea_a1_runtime.apps.tfp.verify
    ;;
  smoke)
    run_root_module galaxea_a1_runtime.apps.tfp.smoke
    ;;
  run|start)
    run_pipeline
    ;;
  stop)
    stop_runtime
    ;;
  status)
    "${BASE_RUNTIME}" status || true
    "${ROBOT_SERVICE}" status || true
    ;;
  logs)
    "${BASE_RUNTIME}" logs
    "${ROBOT_SERVICE}" logs
    ;;
  *)
    a1_usage "$0 [--config PATH] <setup|verify|smoke|run|stop|status|logs>"
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown TFP command: ${1:-}"
      exit 2
    fi
    ;;
esac
