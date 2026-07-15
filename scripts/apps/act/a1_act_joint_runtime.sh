#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOINT_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
CONFIG_PATH="${ROOT}/configs/deployments/act_joint.toml"
source "${ROOT}/scripts/runtime/a1_tmux.sh"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --config <path> <start|services|tmux|stop|doctor|status|logs>" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

eval "$(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.apps.act.config \
    --repo-root "${ROOT}" \
    --shell \
    "${CONFIG_PATH}"
)"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"

check_act_app() {
  if [[ "${DEPLOYMENT_READY}" != "1" ]]; then
    echo "[FAIL] ACT deployment_ready=false; register and review the new square-AgentView checkpoint first." >&2
    exit 2
  fi
  if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "[FAIL] ACT checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
  fi
  if [[ "${WRIST_BACKEND}" == "v4l2" && "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    echo "[FAIL] Wrist camera not found: ${WRIST_CAMERA}" >&2
    exit 2
  fi
}

joint_runtime_env() {
  env \
    A1_RUNTIME_PREFIX="${PREFIX}" \
    A1_TRACKER_NODE="/jointTracker_demo_node" \
    "$@"
}

start_services() {
  echo "Using ACT config: ${CONFIG_PATH}"
  joint_runtime_env "${JOINT_RUNTIME}" services
}

start_tmux() {
  check_act_app
  echo "Using ACT config: ${CONFIG_PATH}"
  local bridge_command=(
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/act/act_joint_policy_bridge.py"
    "${BRIDGE_ARGS[@]}"
  )
  local bridge_command_q
  printf -v bridge_command_q "%q " "${bridge_command[@]}"
  a1_tmux_start "${SESSION}" "${ROOT}" \
    "export PYTHONPATH=\"${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:\${PYTHONPATH:-}\"; ${bridge_command_q}; rc=\$?; echo ACT_BRIDGE_EXIT=\$rc; exec bash"
  sleep 4
  if ! a1_tmux_has_session "${SESSION}"; then
    echo "[FAIL] tmux session exited during startup." >&2
    exit 2
  fi
  local pane
  pane="$(a1_tmux_capture "${SESSION}" 80 || true)"
  printf "%s\n" "${pane}"
  if grep -q "ACT_BRIDGE_EXIT=" <<<"${pane}"; then
    echo "[FAIL] ACT bridge exited during startup." >&2
    exit 2
  fi
}

doctor() {
  local args=("$@")
  joint_runtime_env "${JOINT_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/act/a1_act_doctor.py" \
      --checkpoint "${CHECKPOINT}" \
      --wrist-backend "${WRIST_BACKEND}" \
      --wrist-serial "${WRIST_SERIAL}" \
      --wrist-camera "${WRIST_CAMERA}" \
      "${args[@]}"
}

stop_runtime() {
  a1_tmux_stop "${SESSION}"
  joint_runtime_env "${JOINT_RUNTIME}" stop
  echo "ACT A1 joint bridge stopped."
}

status() {
  joint_runtime_env "${JOINT_RUNTIME}" status
  echo
  echo "tmux:"
  a1_tmux_status "${SESSION}"
}

case "${1:-help}" in
  start)
    check_act_app
    start_services
    start_tmux
    echo
    echo "Attach with: tmux attach -t ${SESSION}"
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
    cat <<EOF
Usage: $0 [--config configs/deployments/act_joint.toml] <start|services|tmux|stop|doctor|status|logs>

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
    ;;
esac
