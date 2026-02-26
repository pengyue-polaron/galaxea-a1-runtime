#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CURRENT_USER="${USER:-$(id -un)}"

if [[ -f "${SDK_ROOT}/install/setup.bash" ]]; then
  # `setup.bash` may reference unset env vars, so avoid nounset during source.
  set +u
  # shellcheck disable=SC1091
  source "${SDK_ROOT}/install/setup.bash"
  set -u
else
  set -u
fi

STATE_FILE="/tmp/a1_eef_record_${CURRENT_USER}.state"
PID_FILE="/tmp/a1_eef_record_${CURRENT_USER}.pid"
LEGACY_STATE_FILE="/tmp/a1_eef_record.state"
LEGACY_PID_FILE="/tmp/a1_eef_record.pid"
OUT_DIR="${A1_RECORD_OUT_DIR:-${SDK_ROOT}/data/records}"
TOPICS=(
  /end_effector_pose
  /joint_states_host
  /arm_status_host
  /gripper_position_control_host
  /gripper_force_control_host
  /gripper_command_host
  /gripper_stroke_host
  /tf
  /tf_static
)

print_usage() {
  cat <<'EOF'
Usage:
  tools/a1_record.sh start [tag]
  tools/a1_record.sh stop
  tools/a1_record.sh status

Notes:
  - start: begins rosbag recording in background.
  - stop: sends SIGINT to rosbag for a clean bag shutdown.
  - status: shows whether recorder is currently running.
  - default recorded topics include arm EEF/joint state and gripper command+feedback.
  - output dir can be set via A1_RECORD_OUT_DIR.
EOF
}

is_running() {
  active_pid_file >/dev/null
}

pid_is_record_proc() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"rosbag"* ]] && [[ "${cmdline}" == *"record"* ]]
}

active_pid_file() {
  local f pid
  for f in "${PID_FILE}" "${LEGACY_PID_FILE}"; do
    [[ -f "${f}" ]] || continue
    pid="$(cat "${f}" 2>/dev/null || true)"
    if pid_is_record_proc "${pid}"; then
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
    echo "Recorder already running (pid=${running_pid})."
    exit 1
  fi

  local ts tag bag_prefix log_file
  ts="$(date +%Y%m%d_%H%M%S)"
  tag="${1:-drag}"
  mkdir -p "${OUT_DIR}"
  bag_prefix="${OUT_DIR}/a1_eef_${tag}_${ts}"
  log_file="${OUT_DIR}/a1_eef_${tag}_${ts}.log"

  nohup rosbag record -O "${bag_prefix}" "${TOPICS[@]}" >"${log_file}" 2>&1 &
  local pid=$!
  echo "${pid}" > "${PID_FILE}"
  {
    echo "pid=${pid}"
    echo "bag_prefix=${bag_prefix}"
    echo "log_file=${log_file}"
    echo "start_time=${ts}"
  } > "${STATE_FILE}"

  sleep 1
  if kill -0 "${pid}" 2>/dev/null; then
    echo "Recording started."
    echo "  pid: ${pid}"
    echo "  bag: ${bag_prefix}.bag"
    echo "  log: ${log_file}"
  else
    echo "Failed to start rosbag recorder. Check ROS environment and roscore."
    rm -f "${PID_FILE}" "${STATE_FILE}"
    exit 1
  fi
}

cmd_stop() {
  local pid_file
  if ! pid_file="$(active_pid_file)"; then
    echo "Recorder is not running."
    exit 1
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if pid_is_record_proc "${pid}"; then
    kill -INT "${pid}"
    for _ in {1..10}; do
      if ! pid_is_record_proc "${pid}"; then
        break
      fi
      sleep 1
    done
  fi

  if pid_is_record_proc "${pid}"; then
    echo "Recorder did not stop after SIGINT. Sending SIGTERM."
    kill -TERM "${pid}" || true
  fi

  local state_file="${STATE_FILE}"
  if [[ "${pid_file}" == "${LEGACY_PID_FILE}" ]]; then
    state_file="${LEGACY_STATE_FILE}"
  fi

  if [[ -f "${state_file}" ]]; then
    # shellcheck disable=SC1090
    source "${state_file}"
    echo "Recording stopped."
    echo "  bag: ${bag_prefix}.bag"
    echo "  log: ${log_file}"
  else
    echo "Recording stopped."
  fi

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
    echo "Recorder running (pid=${pid})."
    if [[ -f "${state_file}" ]]; then
      # shellcheck disable=SC1090
      source "${state_file}"
      echo "  bag: ${bag_prefix}.bag"
      echo "  log: ${log_file}"
      echo "  start: ${start_time}"
    fi
  else
    echo "Recorder not running."
  fi
}

main() {
  local cmd="${1:-}"
  case "${cmd}" in
    start)
      cmd_start "${2:-}"
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
