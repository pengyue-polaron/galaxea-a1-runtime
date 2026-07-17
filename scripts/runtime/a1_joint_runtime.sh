#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
SYSTEM_CONFIG_PATH="${A1_SYSTEM_CONFIG_PATH:-}"
PREFIX="${A1_RUNTIME_PREFIX:-a1-joint-runtime}"
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
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.configuration.system \
      "${config_args[@]}"
fi
ROSCORE_CONTAINER="${PREFIX}-roscore"
DRIVER_CONTAINER="${PREFIX}-driver"
TRACKER_CONTAINER="${PREFIX}-joint-tracker-staged"
RELAY_CONTAINER="${PREFIX}-command-relay"
TRACKER_NODE="${A1_TRACKER_NODE:-${JOINT_TRACKER_NODE:-}}"
source "${ROOT}/scripts/runtime/a1_services.sh"

stop_runtime() {
  a1_remove_runtime_containers \
    "${RELAY_CONTAINER}" \
    "${TRACKER_CONTAINER}" \
    "${DRIVER_CONTAINER}" \
    "${ROSCORE_CONTAINER}"
  a1_cleanup_shared_ros_nodes
  a1_success "A1 joint execution runtime stopped."
}

start_services() {
  local startup_complete=0
  cleanup_failed_start() {
    if [[ "${startup_complete}" != "1" ]]; then
      a1_cleanup "Startup failed; stopping partial A1 joint runtime."
      stop_runtime >/dev/null
    fi
  }
  trap cleanup_failed_start ERR

  a1_info "Config: ${SYSTEM_CONFIG_PATH}"
  a1_preflight_container_runtime
  stop_runtime >/dev/null
  a1_step "0/4 Ensuring ROS master"
  a1_ensure_roscore "${ROSCORE_CONTAINER}"

  a1_step "1/4 Starting A1 driver"
  a1_start_driver "${DRIVER_CONTAINER}"
  a1_wait_valid_joint_feedback "${DRIVER_CONTAINER}" "${JOINT_STATES_TOPIC}"

  a1_step "2/4 Starting isolated joint tracker"
  a1_container_run tracker "${TRACKER_CONTAINER}" \
    "${A1_ROS_PREFIX} && exec roslaunch /workspace/scripts/runtime/joint_tracker_staged.launch staged_command_topic:=${STAGED_TOPIC} joint_states_topic:=${JOINT_STATES_TOPIC} target_topic:=${JOINT_TARGET_TOPIC} ee_pose_topic:=${EEF_POSE_TOPIC} tracker_node:=${JOINT_TRACKER_NODE_NAME}"
  a1_wait_topic "${TRACKER_CONTAINER}" "${EEF_POSE_TOPIC}"

  a1_step "3/4 Starting fail-closed relay (LOCKED)"
  a1_start_command_relay "${RELAY_CONTAINER}"
  a1_wait_topic "${RELAY_CONTAINER}" "${RELAY_STATUS_TOPIC}"

  a1_success "4/4 Joint runtime services ready; relay remains LOCKED"
  startup_complete=1
  trap - ERR
}

doctor() {
  local args=("$@")
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --frozen --project "${ROOT}" python "${ROOT}/scripts/runtime/a1_runtime_doctor.py" \
      --system-config "${SYSTEM_CONFIG_PATH}" \
      --tracker-node "${TRACKER_NODE}" \
      "${args[@]}"
}

status() {
  a1_info "Joint runtime containers"
  docker ps -a --format '{{.Names}}\t{{.Status}}' |
    grep -E "^${PREFIX}-" || a1_info "No ${PREFIX}-* containers."
  echo
  doctor
}

logs() {
  for name in "${DRIVER_CONTAINER}" "${TRACKER_CONTAINER}" "${RELAY_CONTAINER}" "${ROSCORE_CONTAINER}"; do
    a1_info "Logs: ${name}"
    docker logs --tail "${A1_LOG_TAIL:-120}" "${name}" 2>&1 || true
  done
}

eef_nudge() {
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --frozen --project "${ROOT}" python "${ROOT}/scripts/runtime/eef_nudge.py" \
      --config "${SYSTEM_CONFIG_PATH}" \
      "$@"
}

case "${1:-help}" in
  start|services)
    start_services
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
  eef-nudge)
    shift
    eef_nudge "$@"
    ;;
  *)
    a1_usage "$0 <start|services|stop|doctor|status|logs|eef-nudge>"
    cat <<EOF
  start     Start ROS master, A1 driver, isolated joint tracker, and locked relay
  services  Alias for start
  stop      Stop A1 joint execution runtime containers
  doctor    Layered health check; add --require-execution after a target is staged
  status    Containers and doctor summary
  logs      Tail runtime logs
  eef-nudge Interactive staged EEF-to-IK nudge tool; pass --execute to move hardware

Environment:
  A1_SYSTEM_CONFIG_PATH=${SYSTEM_CONFIG_PATH}
  A1_RUNTIME_PREFIX=${PREFIX}
  A1_TRACKER_NODE=${TRACKER_NODE}
EOF
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown joint-runtime command: ${1:-}"
      exit 2
    fi
    ;;
esac
