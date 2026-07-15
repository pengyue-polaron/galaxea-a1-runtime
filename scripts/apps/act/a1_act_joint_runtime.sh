#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
JOINT_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
CONFIG_PATH=""
source "${ROOT}/scripts/runtime/a1_tmux.sh"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    a1_usage "$0 --config <path> <start|services|tmux|stop|doctor|status|logs>" >&2
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
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.act.config \
    "${config_args[@]}"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"

check_act_app() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    a1_fail "ACT deployment_ready=false; register and review the new square-AgentView checkpoint first."
    exit 2
  fi
  if [[ ! -d "${CHECKPOINT}" ]]; then
    a1_fail "ACT checkpoint not found: ${CHECKPOINT}"
    exit 2
  fi
  if [[ "${WRIST_BACKEND}" == "v4l2" && "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    a1_fail "Wrist camera not found: ${WRIST_CAMERA}"
    exit 2
  fi
}

joint_runtime_env() {
  env \
    A1_RUNTIME_PREFIX="${PREFIX}" \
    A1_TRACKER_NODE="${JOINT_TRACKER_NODE}" \
    "$@"
}

start_services() {
  a1_info "Config: ${CONFIG_PATH}"
  joint_runtime_env "${JOINT_RUNTIME}" services
}

start_tmux() {
  check_act_app
  a1_info "Config: ${CONFIG_PATH}"
  local bridge_command=(
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/act/act_joint_policy_bridge.py"
    --config "${CONFIG_PATH}"
  )
  local bridge_command_q
  printf -v bridge_command_q "%q " "${bridge_command[@]}"
  a1_tmux_start "${SESSION}" "${ROOT}" \
    "export PYTHONPATH=\"${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:\${PYTHONPATH:-}\"; ${bridge_command_q}; rc=\$?; echo ACT_BRIDGE_EXIT=\$rc; exec bash"
  a1_tmux_verify_startup \
    "${SESSION}" "ACT_BRIDGE_EXIT=" "ACT bridge" "${TMUX_STARTUP_GRACE_S}"
}

doctor() {
  local args=("$@")
  joint_runtime_env "${JOINT_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/act/a1_act_doctor.py" \
      --config "${CONFIG_PATH}" \
      "${args[@]}"
}

stop_runtime() {
  a1_tmux_stop "${SESSION}"
  joint_runtime_env "${JOINT_RUNTIME}" stop
  a1_success "ACT A1 joint bridge stopped."
}

status() {
  local rc=0
  joint_runtime_env "${JOINT_RUNTIME}" status || rc=$?
  echo
  a1_info "ACT tmux sessions"
  a1_tmux_status "${SESSION}" || rc=$?
  return "${rc}"
}

case "${1:-help}" in
  start)
    check_act_app
    start_services
    start_tmux
    echo
    a1_success "ACT runtime started."
    a1_info "Attach with: tmux attach -t ${SESSION}"
    ;;
  services)
    start_services
    ;;
  tmux)
    start_tmux
    ;;
  stop)
    stop_runtime
    ;;
  doctor)
    shift
    doctor "$@"
    ;;
  status)
    status
    ;;
  logs)
    joint_runtime_env "${JOINT_RUNTIME}" logs
    echo "===== ${SESSION} tmux ====="
    a1_tmux_capture "${SESSION}" "${A1_LOG_TAIL:-120}" 2>&1 || true
    ;;
  *)
    a1_usage "$0 [--config PATH] <start|services|tmux|stop|doctor|status|logs>"
    cat <<EOF
  start     Start A1 joint runtime, then open the interactive ACT bridge tmux
  services  Start only ROS, A1 driver, jointTracker, and locked relay
  tmux      Start only the interactive ACT bridge
  stop      Stop the ACT tmux and A1 joint runtime
  doctor    Run joint runtime checks plus ACT app checks
  status    Joint runtime status plus tmux state
  logs      Tail joint runtime logs

Config:
  ${CONFIG_PATH}
EOF
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown ACT command: ${1:-}"
      exit 2
    fi
    ;;
esac
