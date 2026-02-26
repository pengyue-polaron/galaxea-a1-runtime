#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CURRENT_USER="${USER:-$(id -un)}"

if [[ -f "${SDK_ROOT}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${SDK_ROOT}/install/setup.bash"
  set -u
else
  set -u
fi

STATE_FILE="/tmp/a1_drag_mode_${CURRENT_USER}.state"
PID_FILE="/tmp/a1_drag_mode_${CURRENT_USER}.pid"
LEGACY_STATE_FILE="/tmp/a1_drag_mode.state"
LEGACY_PID_FILE="/tmp/a1_drag_mode.pid"
LOG_DIR="${A1_RECORD_OUT_DIR:-${SDK_ROOT}/data/records}"
DEFAULT_KP="${A1_DRAG_KP:-1.5,1.5,1.2,0.8,0.6,0.4}"
DEFAULT_KD="${A1_DRAG_KD:-0.08,0.08,0.06,0.05,0.04,0.03}"
DEFAULT_MODE="${A1_DRAG_MODE:-0}"
ZERO_GRIPPER_FORCE="${A1_DRAG_ZERO_GRIPPER_FORCE:-1}"
HOLD_GRIPPER_POSITION="${A1_DRAG_HOLD_GRIPPER_POSITION:-1}"
ROS_RUNTIME_DIR="${LOG_DIR}/ros_runtime"

print_usage() {
  cat <<'EOF'
Usage:
  tools/a1_drag_mode.sh start [kp_csv] [kd_csv] [mode]
  tools/a1_drag_mode.sh stop
  tools/a1_drag_mode.sh status

Examples:
  tools/a1_drag_mode.sh start
  tools/a1_drag_mode.sh start "1.0,1.0,0.8,0.6,0.4,0.3" "0.06,0.06,0.05,0.04,0.03,0.02"
  tools/a1_drag_mode.sh start "0.8,0.8,0.6,0.4,0.3,0.2" "0.04,0.04,0.03,0.02,0.02,0.01" 1
  # disable zero-force override for gripper:
  A1_DRAG_ZERO_GRIPPER_FORCE=0 tools/a1_drag_mode.sh start
  # disable continuous gripper position hold:
  A1_DRAG_HOLD_GRIPPER_POSITION=0 tools/a1_drag_mode.sh start
EOF
}

is_running() {
  active_pid_file >/dev/null
}

pid_is_drag_proc() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"a1_drag_compliance.py"* ]]
}

active_pid_file() {
  local f pid
  for f in "${PID_FILE}" "${LEGACY_PID_FILE}"; do
    [[ -f "${f}" ]] || continue
    pid="$(cat "${f}" 2>/dev/null || true)"
    if pid_is_drag_proc "${pid}"; then
      echo "${f}"
      return 0
    fi
  done
  return 1
}

cmd_start() {
  local running_file running_pid
  if running_file="$(active_pid_file)"; then
    running_pid="$(cat "${running_file}" 2>/dev/null || true)"
    echo "Drag mode already running (pid=${running_pid})."
    exit 1
  fi

  local ts kp kd mode log_file
  ts="$(date +%Y%m%d_%H%M%S)"
  kp="${1:-${DEFAULT_KP}}"
  kd="${2:-${DEFAULT_KD}}"
  mode="${3:-${DEFAULT_MODE}}"
  mkdir -p "${LOG_DIR}"
  mkdir -p "${ROS_RUNTIME_DIR}/log" "${ROS_RUNTIME_DIR}/home"
  export ROS_LOG_DIR="${ROS_RUNTIME_DIR}/log"
  export ROS_HOME="${ROS_RUNTIME_DIR}/home"
  log_file="${LOG_DIR}/a1_drag_mode_${ts}.log"

  local extra_args=()
  if [[ "${ZERO_GRIPPER_FORCE}" == "1" ]]; then
    extra_args+=(--zero-gripper-force)
  fi
  if [[ "${HOLD_GRIPPER_POSITION}" == "1" ]]; then
    extra_args+=(--hold-gripper-position)
  fi

  nohup "${SCRIPT_DIR}/a1_drag_compliance.py" \
    --joint-topic /joint_states_host \
    --cmd-topic /arm_joint_command_host \
    --kp "${kp}" \
    --kd "${kd}" \
    --mode "${mode}" \
    --rate 100 \
    --wait-timeout 0 \
    "${extra_args[@]}" \
    >"${log_file}" 2>&1 &

  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  {
    echo "pid=${pid}"
    echo "kp=${kp}"
    echo "kd=${kd}"
    echo "mode=${mode}"
    echo "zero_gripper_force=${ZERO_GRIPPER_FORCE}"
    echo "hold_gripper_position=${HOLD_GRIPPER_POSITION}"
    echo "log_file=${log_file}"
    echo "start_time=${ts}"
  } > "${STATE_FILE}"

  sleep 2
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Drag mode started."
    echo "  pid: ${pid}"
    echo "  kp: ${kp}"
    echo "  kd: ${kd}"
    echo "  mode: ${mode}"
    echo "  zero_gripper_force: ${ZERO_GRIPPER_FORCE}"
    echo "  hold_gripper_position: ${HOLD_GRIPPER_POSITION}"
    echo "  log: ${log_file}"
  else
    echo "Failed to start drag mode. Check ROS core and driver."
    rm -f "${PID_FILE}" "${STATE_FILE}"
    exit 1
  fi
}

cmd_stop() {
  local pid_file
  if ! pid_file="$(active_pid_file)"; then
    echo "Drag mode is not running."
    exit 1
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if pid_is_drag_proc "${pid}"; then
    kill -INT "${pid}"
    for _ in {1..5}; do
      if ! pid_is_drag_proc "${pid}"; then
        break
      fi
      sleep 1
    done
  fi

  if pid_is_drag_proc "${pid}"; then
    kill -TERM "${pid}" || true
  fi

  echo "Drag mode stopped."
  rm -f "${PID_FILE}" "${STATE_FILE}" "${LEGACY_PID_FILE}" "${LEGACY_STATE_FILE}" || true
}

cmd_status() {
  local pid_file
  if pid_file="$(active_pid_file)"; then
    local pid state_file
    pid="$(cat "${pid_file}")"
    state_file="${STATE_FILE}"
    if [[ "${pid_file}" == "${LEGACY_PID_FILE}" ]]; then
      state_file="${LEGACY_STATE_FILE}"
    fi
    echo "Drag mode running (pid=${pid})."
    if [[ -f "${state_file}" ]]; then
      # shellcheck disable=SC1090
      source "${state_file}"
      zero_gripper_force="${zero_gripper_force:-unknown}"
      hold_gripper_position="${hold_gripper_position:-unknown}"
      echo "  kp: ${kp}"
      echo "  kd: ${kd}"
      echo "  mode: ${mode}"
      echo "  zero_gripper_force: ${zero_gripper_force}"
      echo "  hold_gripper_position: ${hold_gripper_position}"
      echo "  log: ${log_file}"
      echo "  start: ${start_time}"
    fi
  else
    echo "Drag mode not running."
  fi
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    start)
      cmd_start "${2:-}" "${3:-}" "${4:-}"
      ;;
    stop)
      cmd_stop
      ;;
    status)
      cmd_status
      ;;
    *)
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
