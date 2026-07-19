#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_runtime.sh"
CAMERA_RUNTIME="${ROOT}/scripts/apps/cameras/a1_camera_web_runtime.sh"
CONFIG_PATH=""
COLLECTION_TASK=""

runtime_args=()
while (( $# > 0 )); do
  case "$1" in
    --config)
      if [[ -n "${CONFIG_PATH}" || -z "${2:-}" ]]; then
        a1_fail "--config requires one path."
        exit 2
      fi
      CONFIG_PATH="$2"
      shift 2
      ;;
    --task)
      if [[ -n "${COLLECTION_TASK}" || -z "${2:-}" ]]; then
        a1_fail "--task requires one non-empty task description."
        exit 2
      fi
      COLLECTION_TASK="$2"
      shift 2
      ;;
    *)
      runtime_args+=("$1")
      shift
      ;;
  esac
done
set -- "${runtime_args[@]}"

if [[ -n "${COLLECTION_TASK}" && "${1:-}" != "collect" ]]; then
  a1_fail "--task is valid only with collect."
  exit 2
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
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.teleop.config \
    "${config_args[@]}"
export A1_SYSTEM_CONFIG_PATH="${SYSTEM_CONFIG_PATH}"

ROSCORE_CONTAINER="${PREFIX}-roscore"
DRIVER_CONTAINER="${PREFIX}-driver"
TRACKER_CONTAINER="${PREFIX}-joint-tracker-staged"
RELAY_CONTAINER="${PREFIX}-command-relay"
LOG_DIR="${RUN_DIR}/logs"
BRIDGE_PID_FILE="${RUN_DIR}/bridge.pid"
source "${ROOT}/scripts/runtime/a1_services.sh"

bridge_group_has_live_process() {
  local pgid="$1"
  ps -eo pgid=,stat=,comm=,args= | awk -v pgid="${pgid}" '
    $1 == pgid && $2 !~ /^Z/ && $3 ~ /^python/ && \
      $0 ~ /python[^ ]*[[:space:]]+[^ ]*so100_joint_bridge\.py([[:space:]]|$)/ { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

stop_bridge() {
  if [[ -f "${BRIDGE_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${BRIDGE_PID_FILE}")"
    if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]]; then
      a1_fail "Invalid teleop bridge pid file: ${BRIDGE_PID_FILE}"
      return 1
    fi
    if ! bridge_group_has_live_process "${pid}"; then
      rm -f "${BRIDGE_PID_FILE}"
      return 0
    fi
    kill -- "-${pid}" >/dev/null 2>&1 || true
    local deadline=$((SECONDS + ${BRIDGE_STOP_TIMEOUT_S%.*}))
    while bridge_group_has_live_process "${pid}" && (( SECONDS < deadline )); do
      sleep 0.1
    done
    if bridge_group_has_live_process "${pid}"; then
      a1_fail "Teleop bridge process group ${pid} did not stop within ${BRIDGE_STOP_TIMEOUT_S} seconds."
      return 1
    fi
    rm -f "${BRIDGE_PID_FILE}"
  fi
}

stop_runtime() {
  stop_bridge
  a1_remove_runtime_containers \
    "${RELAY_CONTAINER}" \
    "${TRACKER_CONTAINER}" \
    "${DRIVER_CONTAINER}" \
    "${ROSCORE_CONTAINER}"
  a1_success "A1 teleop runtime stopped."
}

check_host_inputs() {
  if [[ ! -e "${SERIAL}" ]]; then
    a1_fail "A1 serial device not found: ${SERIAL}"
    exit 2
  fi
  if [[ ! -e "${LEADER_PORT}" ]]; then
    a1_fail "Teleop leader port not found: ${LEADER_PORT}"
    ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
    exit 2
  fi
}

start_services() {
  a1_info "Config: ${CONFIG_PATH}"
  check_host_inputs
  a1_preflight_container_runtime
  "${BASE_RUNTIME}" stop >/dev/null 2>&1 || true
  stop_runtime >/dev/null 2>&1 || true
  mkdir -p "${LOG_DIR}"

  a1_step "1/4 ROS master"
  a1_ensure_roscore "${ROSCORE_CONTAINER}"

  a1_step "2/4 A1 driver"
  a1_start_driver "${DRIVER_CONTAINER}"
  a1_wait_valid_joint_feedback "${DRIVER_CONTAINER}" "${JOINT_STATES_TOPIC}"

  a1_step "3/4 Joint tracker"
  a1_container_run tracker "${TRACKER_CONTAINER}" \
    "${A1_ROS_PREFIX} && exec roslaunch /workspace/scripts/runtime/joint_tracker_staged.launch staged_command_topic:=${STAGED_TOPIC} joint_states_topic:=${JOINT_STATES_TOPIC} target_topic:=${JOINT_TARGET_TOPIC} ee_pose_topic:=${EEF_POSE_TOPIC} tracker_node:=${JOINT_TRACKER_NODE_NAME}"
  a1_wait_topic "${TRACKER_CONTAINER}" "${EEF_POSE_TOPIC}"

  a1_step "4/4 Command relay"
  a1_start_command_relay "${RELAY_CONTAINER}"
  a1_wait_topic "${RELAY_CONTAINER}" "${RELAY_STATUS_TOPIC}"

  a1_success "Teleop services ready."
}

start_bridge() {
  local quiet="${1:-}"
  if [[ "${quiet}" != "--quiet" ]]; then
    a1_step "Starting Teleop bridge"
  fi
  check_host_inputs
  mkdir -p "${LOG_DIR}"
  stop_bridge
  local log_file="${LOG_DIR}/bridge.log"
  : > "${log_file}"
  setsid env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${ROOT}/scripts/apps/teleop/so100_joint_bridge.py" \
      --config "${CONFIG_PATH}" \
      >> "${log_file}" 2>&1 < /dev/null &
  echo "$!" > "${BRIDGE_PID_FILE}"
  wait_bridge_live "${log_file}"
  if [[ "${quiet}" != "--quiet" ]]; then
    a1_success "Teleop bridge ready."
  fi
}

wait_bridge_live() {
  local log_file="$1"
  local pid
  pid="$(cat "${BRIDGE_PID_FILE}")"
  local deadline=$((SECONDS + ${BRIDGE_STARTUP_TIMEOUT_S%.*}))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      a1_fail "Teleop bridge exited during startup. Log: ${log_file}"
      print_bridge_failure_log "${log_file}"
      exit 2
    fi
    if grep -q "relay ACTIVE; teleop is live" "${log_file}" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  a1_fail "Teleop bridge did not become live within ${BRIDGE_STARTUP_TIMEOUT_S} seconds. Log: ${log_file}"
  print_bridge_failure_log "${log_file}"
  exit 2
}

print_bridge_failure_log() {
  local log_file="$1"
  tail -n 120 "${log_file}" >&2 || true
}

doctor() {
  local args=("$@")
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/a1_teleop_doctor.py" \
      --config "${CONFIG_PATH}" \
      "${args[@]}"
  A1_TRACKER_NODE="${JOINT_TRACKER_NODE}" \
    "${BASE_RUNTIME}" doctor "${args[@]}"
}

collect() {
  local experiment="${1:-}"
  shift || true
  if [[ -z "${experiment}" ]]; then
    a1_fail "collect requires an experiment name."
    a1_usage "$0 collect <experiment>" >&2
    exit 2
  fi
  if [[ "$#" -gt 0 ]]; then
    a1_fail "Per-run collector args are disabled. Edit ${CONFIG_PATH} instead."
    exit 2
  fi
  if ! PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - "${experiment}" <<'PY'
import sys
from galaxea_a1_runtime.collection import validate_experiment_name
from galaxea_a1_runtime.console import failure

try:
    validate_experiment_name(sys.argv[1])
except ValueError as exc:
    failure(str(exc))
    raise SystemExit(2)
PY
  then
    exit 2
  fi
  if ! PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.apps.teleop.dataset_doctor \
    --repo-root "${ROOT}" \
    --config "${CONFIG_PATH}" \
    --experiment "${experiment}" >/dev/null; then
    exit 2
  fi
  if [[ -n "${COLLECTION_TASK}" ]]; then
    if ! PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
      -m galaxea_a1_runtime.apps.teleop.collection_task \
      --repo-root "${ROOT}" \
      --config "${CONFIG_PATH}" \
      --experiment "${experiment}" \
      --task "${COLLECTION_TASK}" >/dev/null; then
      exit 2
    fi
  fi
  cleanup_collect() {
    a1_info "Stopping teleop runtime after collection."
    stop_runtime >/dev/null 2>&1 || true
    if ! "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"; then
      a1_fail "Persistent Camera Bridge became unavailable during collection cleanup."
    fi
  }
  trap cleanup_collect EXIT
  "${CAMERA_RUNTIME}" --config "${SYSTEM_CONFIG_PATH}"
  start_services
  start_bridge
  a1_step "Recording a canonical LeRobotDataset v3 from the persistent Camera Bridge."
  local collector_args=(
    --experiment "${experiment}"
    --config "${CONFIG_PATH}"
  )
  if [[ -n "${COLLECTION_TASK}" ]]; then
    collector_args+=(--task "${COLLECTION_TASK}")
  fi
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/teleop_collect.py" \
      "${collector_args[@]}"
}

reset_live() {
  a1_step "Pausing teleop for reset"
  stop_bridge
  if ! PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/a1_so100_reset.py" \
      --config "${CONFIG_PATH}"; then
    a1_fail "Reset failed; teleop remains stopped."
    return 1
  fi
  start_bridge --quiet
  a1_success "Reset complete; teleop ready."
}

reset() {
  cleanup_reset() {
    stop_runtime >/dev/null 2>&1 || true
  }
  trap cleanup_reset EXIT
  start_services
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/a1_so100_reset.py" \
      --config "${CONFIG_PATH}"
}

status() {
  a1_info "Teleop containers"
  docker ps -a --format '{{.Names}}\t{{.Status}}' |
    grep -E "^${PREFIX}-" || a1_info "No ${PREFIX}-* containers."
  echo
  local bridge_pid=""
  if [[ -f "${BRIDGE_PID_FILE}" ]]; then
    bridge_pid="$(cat "${BRIDGE_PID_FILE}")"
  fi
  if [[ "${bridge_pid}" =~ ^[1-9][0-9]*$ ]] && bridge_group_has_live_process "${bridge_pid}"; then
    a1_success "Teleop bridge running (pid=${bridge_pid})."
  else
    a1_info "Teleop bridge is not running."
  fi
  echo
  doctor
}

logs() {
  for name in "${DRIVER_CONTAINER}" "${TRACKER_CONTAINER}" "${RELAY_CONTAINER}" "${ROSCORE_CONTAINER}"; do
    a1_info "Logs: ${name}"
    docker logs --tail "${A1_LOG_TAIL:-120}" "${name}" 2>&1 || true
  done
  a1_info "Logs: teleop bridge"
  tail -n "${A1_LOG_TAIL:-120}" "${LOG_DIR}/bridge.log" 2>/dev/null || true
}

case "${1:-help}" in
  start)
    start_services
    start_bridge
    ;;
  services)
    start_services
    ;;
  bridge)
    start_bridge
    ;;
  collect)
    shift
    collect "$@"
    ;;
  reset)
    reset
    ;;
  _reset-live)
    reset_live
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
    logs
    ;;
  *)
    a1_usage "$0 [--config PATH] [--task TEXT] <start|services|bridge|collect|reset|stop|doctor|status|logs>"
    cat <<EOF
  start     Start staged joint teleop services and SO leader bridge
  services  Start ROS master, A1 driver, staged joint tracker, locked relay
  bridge    Start only the SO leader bridge
  collect   Start teleop, then run the interactive recorder
  reset     Reset A1 and SO leader using ${RESET_CONFIG_PATH}
  stop      Stop bridge and teleop containers
  doctor    Static/import checks plus base runtime doctor
  status    Containers and bridge process state
  logs      Runtime and bridge logs

Config:
  ${CONFIG_PATH}

Important values:
  A1 serial       ${SERIAL}
  SO leader port  ${LEADER_PORT}
  SO leader id    ${LEADER_ID}
  Gripper stroke  ${GRIPPER_MIN_STROKE_MM}..${GRIPPER_MAX_STROKE_MM} mm
EOF
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown teleop command: ${1:-}"
      exit 2
    fi
    ;;
esac
