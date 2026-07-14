#!/usr/bin/env bash
# Shared, app-agnostic Docker/ROS service primitives for A1 runtime entrypoints.
# Callers own lifecycle, tracker selection, UI, and failure cleanup policy.

A1_ROS_PREFIX='source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash"'

a1_require_runtime_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "[FAIL] Required runtime value ${name} is unset." >&2
    return 2
  fi
}

a1_container_run() {
  local name="$1"
  shift
  a1_require_runtime_value ROOT
  a1_require_runtime_value IMAGE
  docker rm -f "${name}" >/dev/null 2>&1 || true
  docker run -d \
    --name "${name}" \
    --network host \
    --ipc host \
    --privileged \
    -v "${ROOT}:/workspace:rw" \
    -v /dev:/dev:rw \
    -e A1_SDK_ROOT=/workspace/third_party/A1_SDK \
    "${IMAGE}" \
    bash -lc "$*" \
    >/dev/null
}

a1_remove_runtime_containers() {
  docker rm -f "$@" >/dev/null 2>&1 || true
}

a1_cleanup_shared_ros_nodes() {
  local ros_container
  ros_container="$(docker ps --format '{{.Names}}' | grep -E '^(galaxea-a1-runtime|a1-research)-a1-noetic-run-' | head -n 1 || true)"
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
  a1_container_run "${container}" "${A1_ROS_PREFIX} && exec roscore"
  local deadline=$((SECONDS + 10))
  until timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[FAIL] ROS master did not start." >&2
      return 2
    fi
    sleep 0.5
  done
}

a1_start_driver() {
  local container="$1"
  a1_require_runtime_value SERIAL
  a1_container_run "${container}" \
    "${A1_ROS_PREFIX} && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${SERIAL}"
}

a1_wait_valid_joint_feedback() {
  local container="$1"
  local topic="$2"
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${A1_ROS_PREFIX}; timeout 2 rostopic echo -n1 '${topic}' | grep -Eq '^position: \\[[^]]+\\]'" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] No non-empty ${topic} after 20 seconds." >&2
  return 1
}

a1_wait_topic() {
  local container="$1"
  local topic="$2"
  local deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${A1_ROS_PREFIX}; timeout 2 rostopic echo -n1 '${topic}' >/dev/null" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] No message on ${topic} after 15 seconds." >&2
  return 1
}

a1_start_command_relay() {
  local container="$1"
  local required=(
    STAGED_TOPIC HOST_COMMAND_TOPIC JOINT_STATES_TOPIC MOTOR_STATUS_TOPIC
    RELAY_ENABLE_TOPIC RELAY_STATUS_TOPIC GRIPPER_TARGET_TOPIC
    GRIPPER_COMMAND_TOPIC GRIPPER_MIN_STROKE_MM GRIPPER_MAX_STROKE_MM
    RELAY_MAX_INPUT_AGE_S RELAY_ARMING_TIMEOUT_S RELAY_MAX_INITIAL_ERROR_RAD
  )
  local name
  for name in "${required[@]}"; do
    a1_require_runtime_value "${name}"
  done
  a1_container_run "${container}" \
    "${A1_ROS_PREFIX} && exec python3 /workspace/scripts/runtime/safe_arm_command_relay.py \
      --input-topic '${STAGED_TOPIC}' --output-topic '${HOST_COMMAND_TOPIC}' \
      --joint-topic '${JOINT_STATES_TOPIC}' --motor-status-topic '${MOTOR_STATUS_TOPIC}' \
      --enable-topic '${RELAY_ENABLE_TOPIC}' --relay-status-topic '${RELAY_STATUS_TOPIC}' \
      --gripper-input-topic '${GRIPPER_TARGET_TOPIC}' --gripper-output-topic '${GRIPPER_COMMAND_TOPIC}' \
      --gripper-min-stroke-mm '${GRIPPER_MIN_STROKE_MM}' --gripper-max-stroke-mm '${GRIPPER_MAX_STROKE_MM}' \
      --max-input-age '${RELAY_MAX_INPUT_AGE_S}' --arming-timeout '${RELAY_ARMING_TIMEOUT_S}' \
      --max-initial-error '${RELAY_MAX_INITIAL_ERROR_RAD}'"
}
