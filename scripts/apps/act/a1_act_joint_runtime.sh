#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOINT_RUNTIME="${ROOT}/scripts/runtime/a1_joint_runtime.sh"
CONFIG_PATH="${ROOT}/configs/inference/act_joint_a1.toml"

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

check_act_app() {
  if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "[FAIL] ACT checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
  fi
  if [[ "${WRIST_CAMERA}" != "auto" && ! -e "${WRIST_CAMERA}" ]]; then
    echo "[FAIL] Wrist camera not found: ${WRIST_CAMERA}" >&2
    exit 2
  fi
}

joint_runtime_env() {
  env \
    A1_NOETIC_IMAGE="${IMAGE}" \
    A1_SERIAL="${SERIAL}" \
    A1_RUNTIME_PREFIX="${PREFIX}" \
    A1_JOINT_TARGET_TOPIC="${TARGET_TOPIC}" \
    A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
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
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  local bridge_command=(
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/act/act_joint_policy_bridge.py"
    "${BRIDGE_ARGS[@]}"
  )
  local bridge_command_q
  printf -v bridge_command_q "%q " "${bridge_command[@]}"
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
    "export PYTHONPATH=\"${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/lerobot/src:\${PYTHONPATH:-}\"; ${bridge_command_q}; rc=\$?; echo ACT_BRIDGE_EXIT=\$rc; exec bash"
  sleep 4
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[FAIL] tmux session exited during startup." >&2
    exit 2
  fi
  local pane
  pane="$(tmux capture-pane -pt "${SESSION}" -S -80 2>/dev/null || true)"
  printf "%s\n" "${pane}"
  if grep -q "ACT_BRIDGE_EXIT=" <<<"${pane}"; then
    echo "[FAIL] ACT bridge exited during startup." >&2
    exit 2
  fi
}

doctor() {
  local args=("$@")
  joint_runtime_env "${JOINT_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/lerobot/src:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/act/a1_act_doctor.py" \
      --checkpoint "${CHECKPOINT}" \
      --wrist-camera "${WRIST_CAMERA}" \
      "${args[@]}"
}

stop_runtime() {
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  joint_runtime_env "${JOINT_RUNTIME}" stop
  echo "ACT A1 joint bridge stopped."
}

status() {
  joint_runtime_env "${JOINT_RUNTIME}" status
  echo
  echo "tmux:"
  tmux list-sessions 2>/dev/null | grep "${SESSION}" || echo "${SESSION}: not running"
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
    tmux capture-pane -pt "${SESSION}" -S -"${A1_LOG_TAIL:-120}" 2>&1 || true
    ;;
  *)
    cat <<EOF
Usage: $0 [--config configs/inference/act_joint_a1.toml] <start|services|tmux|stop|doctor|status|logs>

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
