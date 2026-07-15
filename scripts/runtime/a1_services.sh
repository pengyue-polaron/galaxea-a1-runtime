#!/usr/bin/env bash
# Shared, app-agnostic Docker/ROS service primitives for A1 runtime entrypoints.
# Callers own lifecycle, tracker selection, UI, and failure cleanup policy.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a1_console.sh"

A1_ROS_PREFIX='source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash"'
A1_MANAGED_CONTAINER_LABEL='io.galaxea.a1-runtime.managed=true'

a1_require_runtime_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    a1_fail "Required runtime value ${name} is unset."
    return 2
  fi
}

a1_preflight_container_runtime() {
  a1_require_runtime_value ROOT
  a1_require_runtime_value IMAGE
  a1_require_runtime_value SERIAL
  if ! command -v docker >/dev/null 2>&1; then
    a1_fail "Docker CLI is not installed."
    return 2
  fi
  if ! docker info >/dev/null 2>&1; then
    a1_fail "Docker daemon is unavailable."
    return 2
  fi
  if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    a1_fail "Runtime image is missing: ${IMAGE}"
    return 2
  fi
  if [[ ! -d "${ROOT}/third_party/A1_SDK" ]]; then
    a1_fail "Vendored A1 SDK is missing under ${ROOT}/third_party/A1_SDK."
    return 2
  fi
  if [[ ! -c "${SERIAL}" ]]; then
    a1_fail "A1 serial path is not a character device: ${SERIAL}"
    return 2
  fi
}

a1_container_run() {
  if (( $# != 3 )); then
    a1_fail "a1_container_run expects <profile> <name> <command>."
    return 2
  fi
  local profile="$1"
  local name="$2"
  local command="$3"
  a1_require_runtime_value ROOT
  a1_require_runtime_value IMAGE
  local access_args=(--network host -v "${ROOT}:/workspace:ro")
  case "${profile}" in
    core|relay)
      ;;
    driver)
      a1_require_runtime_value SERIAL
      access_args+=(--device "${SERIAL}:${SERIAL}")
      ;;
    tracker)
      access_args=(--network host --ipc host -v "${ROOT}:/workspace:rw")
      ;;
    *)
      a1_fail "Unknown A1 container profile: ${profile}"
      return 2
      ;;
  esac
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${name}" \
    --label "${A1_MANAGED_CONTAINER_LABEL}" \
    "${access_args[@]}" \
    -e A1_SDK_ROOT=/workspace/third_party/A1_SDK \
    "${IMAGE}" \
    bash -lc "${command}" \
    >/dev/null
}

a1_remove_runtime_containers() {
  docker rm -f "$@" >/dev/null 2>&1 || true
}

a1_remove_all_managed_containers() {
  if ! command -v docker >/dev/null 2>&1; then
    return 0
  fi
  local container_ids=()
  local listing
  if ! listing="$(docker ps -aq --filter "label=${A1_MANAGED_CONTAINER_LABEL}" 2>/dev/null)"; then
    a1_fail "Could not list managed A1 runtime containers."
    return 2
  fi
  mapfile -t container_ids <<<"${listing}"
  if (( ${#container_ids[@]} == 1 )) && [[ -z "${container_ids[0]}" ]]; then
    container_ids=()
  fi
  if (( ${#container_ids[@]} > 0 )); then
    if ! docker rm -f "${container_ids[@]}" >/dev/null; then
      a1_fail "Could not remove every managed A1 runtime container."
      return 2
    fi
  fi
  if ! listing="$(docker ps -aq --filter "label=${A1_MANAGED_CONTAINER_LABEL}" 2>/dev/null)"; then
    a1_fail "Could not verify managed A1 runtime container shutdown."
    return 2
  fi
  if [[ -n "${listing}" ]]; then
    a1_fail "Managed A1 runtime containers remain after shutdown."
    return 2
  fi
}

a1_cleanup_shared_ros_nodes() {
  local ros_container
  ros_container="$(docker ps --format '{{.Names}}' | grep -E '^galaxea-a1-runtime-a1-noetic-run-' | head -n 1 || true)"
  if [[ -n "${ros_container}" ]]; then
    docker exec "${ros_container}" bash -lc \
      'source /opt/ros/noetic/setup.bash; rosnode cleanup <<< y >/dev/null 2>&1 || true' \
      >/dev/null 2>&1 || true
  fi
}

a1_ensure_roscore() {
  local container="$1"
  if timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; then
    return 0
  fi
  a1_container_run core "${container}" "${A1_ROS_PREFIX} && exec roscore"
  a1_require_runtime_value ROS_MASTER_STARTUP_TIMEOUT_S
  local deadline=$((SECONDS + ${ROS_MASTER_STARTUP_TIMEOUT_S%.*}))
  until timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      a1_fail "ROS master did not start."
      return 2
    fi
    sleep 0.5
  done
}

a1_start_driver() {
  local container="$1"
  a1_require_runtime_value SERIAL
  a1_container_run driver "${container}" \
    "${A1_ROS_PREFIX} && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${SERIAL}"
}

a1_wait_valid_joint_feedback() {
  local container="$1"
  local topic="$2"
  a1_require_runtime_value JOINT_FEEDBACK_STARTUP_TIMEOUT_S
  local deadline=$((SECONDS + ${JOINT_FEEDBACK_STARTUP_TIMEOUT_S%.*}))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${A1_ROS_PREFIX}; timeout 2 rostopic echo -n1 '${topic}' | grep -Eq '^position: \\[[^]]+\\]'" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  a1_fail "No non-empty ${topic} after ${JOINT_FEEDBACK_STARTUP_TIMEOUT_S}s."
  return 1
}

a1_wait_topic() {
  local container="$1"
  local topic="$2"
  a1_require_runtime_value TOPIC_STARTUP_TIMEOUT_S
  local deadline=$((SECONDS + ${TOPIC_STARTUP_TIMEOUT_S%.*}))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${A1_ROS_PREFIX}; timeout 2 rostopic echo -n1 '${topic}' >/dev/null" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  a1_fail "No message on ${topic} after ${TOPIC_STARTUP_TIMEOUT_S}s."
  return 1
}

a1_start_command_relay() {
  local container="$1"
  a1_require_runtime_value SYSTEM_CONFIG_PATH
  a1_require_runtime_value ROOT
  local relative_config="${SYSTEM_CONFIG_PATH#${ROOT}/}"
  if [[ "${relative_config}" == "${SYSTEM_CONFIG_PATH}" ]]; then
    a1_fail "System config must be inside the repository for Docker: ${SYSTEM_CONFIG_PATH}"
    return 2
  fi
  a1_container_run relay "${container}" \
    "${A1_ROS_PREFIX} && exec python3 /workspace/scripts/runtime/safe_arm_command_relay.py \
      --config '/workspace/${relative_config}'"
}
