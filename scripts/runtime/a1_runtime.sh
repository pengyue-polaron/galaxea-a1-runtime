#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
SYSTEM_CONFIG_PATH="${A1_SYSTEM_CONFIG_PATH:-}"
PREFIX="${A1_RUNTIME_PREFIX:-a1-runtime}"
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
TRACKER_CONTAINER="${PREFIX}-tracker-staged"
RELAY_CONTAINER="${PREFIX}-command-relay"
TRACKER_NODE="${A1_TRACKER_NODE:-${EE_TRACKER_NODE:-}}"
source "${ROOT}/scripts/runtime/a1_services.sh"

stop_runtime() {
  a1_remove_runtime_containers \
    "${RELAY_CONTAINER}" \
    "${TRACKER_CONTAINER}" \
    "${DRIVER_CONTAINER}" \
    "${ROSCORE_CONTAINER}"
  a1_cleanup_shared_ros_nodes
  a1_success "A1 execution runtime stopped."
}

start_services() {
  local startup_complete=0
  cleanup_failed_start() {
    if [[ "${startup_complete}" != "1" ]]; then
      a1_cleanup "Startup failed; stopping partial A1 runtime."
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

  a1_step "2/4 Starting isolated EE tracker"
  a1_container_run tracker "${TRACKER_CONTAINER}" \
    "${A1_ROS_PREFIX} && exec roslaunch /workspace/scripts/runtime/ee_tracker_staged.launch staged_command_topic:=${STAGED_TOPIC} joint_states_topic:=${JOINT_STATES_TOPIC} target_topic:=${EEF_TARGET_TOPIC} ee_pose_topic:=${EEF_POSE_TOPIC} tracker_node:=${EE_TRACKER_NODE_NAME}"
  a1_wait_topic "${TRACKER_CONTAINER}" "${EEF_POSE_TOPIC}"
  a1_wait_topic "${TRACKER_CONTAINER}" "${STAGED_TOPIC}"

  a1_step "3/4 Starting fail-closed relay (LOCKED)"
  a1_start_command_relay "${RELAY_CONTAINER}"
  a1_wait_topic "${RELAY_CONTAINER}" "${RELAY_STATUS_TOPIC}"

  a1_step "4/4 Running execution doctor"
  if ! doctor --require-execution; then
    a1_fail "Execution doctor failed; stopping partial A1 runtime."
    stop_runtime >/dev/null
    exit 1
  fi
  startup_complete=1
  trap - ERR
  a1_success "A1 execution runtime is ready."
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
  a1_info "Runtime containers"
  docker ps -a --format '{{.Names}}\t{{.Status}}' |
    grep -E "^${PREFIX}-" || a1_info "No ${PREFIX}-* containers."
  echo
  a1_info "Shared ROS containers"
  docker ps --format '{{.Names}}\t{{.Status}}' |
    grep -E '^galaxea-a1-runtime-a1-noetic-run-' || a1_info "No running shared a1-noetic container."
  echo
  doctor
}

logs() {
  for name in "${DRIVER_CONTAINER}" "${TRACKER_CONTAINER}" "${RELAY_CONTAINER}" "${ROSCORE_CONTAINER}"; do
    a1_info "Logs: ${name}"
    docker logs --tail "${A1_LOG_TAIL:-120}" "${name}" 2>&1 || true
  done
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
  *)
    a1_usage "$0 <start|services|stop|doctor|status|logs>"
    cat <<EOF
  start     Start ROS master, A1 driver, isolated tracker, and locked relay
  services  Alias for start
  stop      Stop A1 execution runtime containers
  doctor    Layered health check; add --require-execution after power-on
  status    Containers and doctor summary
  logs      Tail runtime logs
Environment:
  A1_SYSTEM_CONFIG_PATH=${SYSTEM_CONFIG_PATH}
  A1_RUNTIME_PREFIX=${PREFIX}
  A1_TRACKER_NODE=${TRACKER_NODE}
EOF
    if [[ "${1:-help}" != "help" && "${1:-}" != "-h" && "${1:-}" != "--help" ]]; then
      a1_fail "Unknown runtime command: ${1:-}"
      exit 2
    fi
    ;;
esac
