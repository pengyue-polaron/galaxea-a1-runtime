#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_config.sh"
source "${ROOT}/scripts/runtime/a1_services.sh"

CONFIG_PATH="${A1_SYSTEM_CONFIG_PATH:-}"
CONTAINER="a1-rosbag"
OUTPUT_DIR="${ROOT}/outputs/rosbags"
STATE_FILE="/tmp/a1_rosbag_${USER:-$(id -un)}.state"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    a1_usage "$0 [--config path] <start [tag]|stop|status|logs>" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

load_config() {
  local config_args=(--repo-root "${ROOT}" --shell)
  if [[ -n "${CONFIG_PATH}" ]]; then
    config_args+=("${CONFIG_PATH}")
  fi
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    -m galaxea_a1_runtime.configuration.system "${config_args[@]}"
}

container_state() {
  docker inspect --format '{{.State.Status}}' "${CONTAINER}" 2>/dev/null || true
}

start_recording() {
  local tag="${1:-session}"
  if [[ ! "${tag}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    a1_fail "Invalid recording tag: ${tag}"
    return 2
  fi

  load_config
  a1_info "Config: ${SYSTEM_CONFIG_PATH}"
  a1_preflight_container_host
  if [[ -n "$(container_state)" ]]; then
    a1_fail "Recorder container ${CONTAINER} already exists; run 'just rosbag status' or 'just rosbag stop'."
    return 1
  fi
  if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; then
    a1_fail "ROS master is not reachable; start an A1 runtime before recording."
    return 1
  fi

  local timestamp bag_name bag_path topic_args bag_path_q
  timestamp="$(date +%Y%m%d_%H%M%S)"
  bag_name="a1_${tag}_${timestamp}"
  bag_path="${OUTPUT_DIR}/${bag_name}.bag"
  mkdir -p "${OUTPUT_DIR}"

  local topics=(
    "${EEF_POSE_TOPIC}"
    "${EEF_TARGET_TOPIC}"
    "${JOINT_STATES_TOPIC}"
    "${JOINT_TARGET_TOPIC}"
    "${STAGED_TOPIC}"
    "${HOST_COMMAND_TOPIC}"
    "${MOTOR_STATUS_TOPIC}"
    "${MOTION_ENABLE_TOPIC}"
    "${RELAY_STATUS_TOPIC}"
    "${GRIPPER_TARGET_TOPIC}"
    "${GRIPPER_COMMAND_TOPIC}"
    "${GRIPPER_FEEDBACK_TOPIC}"
    /tf
    /tf_static
  )
  printf -v topic_args '%q ' "${topics[@]}"
  printf -v bag_path_q '%q' "/workspace/outputs/rosbags/${bag_name}"

  a1_step "Starting ROS bag recorder"
  a1_container_run output-writer "${CONTAINER}" \
    "${A1_ROS_PREFIX} && exec rosbag record -O ${bag_path_q} ${topic_args}"
  printf '%s\n' "${bag_path}" >"${STATE_FILE}"
  sleep 1
  if ! a1_require_running_container "${CONTAINER}" "ROS bag recording"; then
    a1_remove_runtime_containers "${CONTAINER}"
    rm -f "${STATE_FILE}"
    return 1
  fi

  a1_success "ROS bag recording started."
  a1_info "Output: ${bag_path}"
}

stop_recording() {
  local state
  state="$(container_state)"
  if [[ -z "${state}" ]]; then
    a1_fail "ROS bag recorder is not running."
    return 1
  fi

  if [[ "${state}" == "running" ]]; then
    a1_step "Stopping ROS bag recorder cleanly"
    docker kill --signal=INT "${CONTAINER}" >/dev/null
    local deadline=$((SECONDS + 15))
    while [[ "$(container_state)" == "running" && ${SECONDS} -lt ${deadline} ]]; do
      sleep 1
    done
    if [[ "$(container_state)" == "running" ]]; then
      a1_warn "Recorder did not stop after SIGINT; sending SIGTERM."
      docker stop --time 5 "${CONTAINER}" >/dev/null
    fi
  fi

  a1_remove_runtime_containers "${CONTAINER}"
  a1_success "ROS bag recording stopped."
  if [[ -f "${STATE_FILE}" ]]; then
    a1_info "Output: $(<"${STATE_FILE}")"
  fi
  rm -f "${STATE_FILE}"
}

show_status() {
  local state
  state="$(container_state)"
  if [[ -z "${state}" ]]; then
    a1_info "ROS bag recorder is not running."
    return 1
  fi
  a1_info "Recorder container: ${CONTAINER} (${state})"
  if [[ -f "${STATE_FILE}" ]]; then
    a1_info "Output: $(<"${STATE_FILE}")"
  fi
  [[ "${state}" == "running" ]]
}

show_logs() {
  if [[ -z "$(container_state)" ]]; then
    a1_fail "ROS bag recorder container does not exist."
    return 1
  fi
  docker logs --tail "${A1_LOG_TAIL:-120}" "${CONTAINER}"
}

print_usage() {
  a1_usage "$0 [--config path] <start [tag]|stop|status|logs>"
  echo "  start   Record configured A1 ROS topics under outputs/rosbags/"
  echo "  stop    Finalize the active bag and stop the recorder"
  echo "  status  Show recorder state and output path"
  echo "  logs    Show recorder logs"
}

case "${1:-help}" in
  start)
    start_recording "${2:-}"
    ;;
  stop)
    stop_recording
    ;;
  status)
    show_status
    ;;
  logs)
    show_logs
    ;;
  help|-h|--help)
    print_usage
    ;;
  *)
    a1_fail "Unknown rosbag command: ${1:-}"
    print_usage >&2
    exit 2
    ;;
esac
