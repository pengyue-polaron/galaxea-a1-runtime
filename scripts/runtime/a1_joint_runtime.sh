#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${A1_NOETIC_IMAGE:-a1-research/a1-noetic-arm64:local}"
SERIAL="${A1_SERIAL:-/dev/a1}"
PREFIX="${A1_RUNTIME_PREFIX:-a1-joint-runtime}"
ROSCORE_CONTAINER="${PREFIX}-roscore"
DRIVER_CONTAINER="${PREFIX}-driver"
TRACKER_CONTAINER="${PREFIX}-joint-tracker-staged"
RELAY_CONTAINER="${PREFIX}-command-relay"
TARGET_TOPIC="${A1_JOINT_TARGET_TOPIC:-/arm_joint_target_position}"
STAGED_TOPIC="${A1_STAGED_COMMAND_TOPIC:-/arm_joint_command_a1_staged}"
RELAY_ENABLE_TOPIC="${A1_RELAY_ENABLE_TOPIC:-/a1_arm_motion_enable}"
RELAY_STATUS_TOPIC="${A1_RELAY_STATUS_TOPIC:-/a1_arm_relay_status}"
TRACKER_NODE="${A1_TRACKER_NODE:-/jointTracker_demo_node}"

ros_prefix='source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash"'

container_run() {
  local name="$1"
  shift
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
    bash -lc "$*"
}

stop_runtime() {
  docker rm -f \
    "${RELAY_CONTAINER}" \
    "${TRACKER_CONTAINER}" \
    "${DRIVER_CONTAINER}" \
    "${ROSCORE_CONTAINER}" \
    >/dev/null 2>&1 || true

  local ros_container
  ros_container="$(docker ps --format '{{.Names}}' | grep -E '^a1-research-a1-noetic-run-' | head -n 1 || true)"
  if [[ -n "${ros_container}" ]]; then
    docker exec "${ros_container}" bash -lc \
      'source /opt/ros/noetic/setup.bash; rosnode cleanup <<< y >/dev/null 2>&1 || true' \
      >/dev/null 2>&1 || true
  fi
  echo "A1 joint execution runtime stopped."
}

wait_valid_joint_feedback() {
  local deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    if docker exec "${DRIVER_CONTAINER}" bash -lc \
      "${ros_prefix}; timeout 2 rostopic echo -n1 /joint_states_host | grep -Eq '^position: \\[[^]]+\\]'" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] No non-empty /joint_states_host after 20 seconds." >&2
  return 1
}

wait_topic() {
  local container="$1"
  local topic="$2"
  local deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    if docker exec "${container}" bash -lc \
      "${ros_prefix}; timeout 2 rostopic echo -n1 '${topic}' >/dev/null" \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "[FAIL] No message on ${topic} after 15 seconds." >&2
  return 1
}

start_services() {
  local startup_complete=0
  cleanup_failed_start() {
    if [[ "${startup_complete}" != "1" ]]; then
      echo "[CLEANUP] Startup failed; stopping partial A1 joint runtime." >&2
      stop_runtime >/dev/null
    fi
  }
  trap cleanup_failed_start ERR

  if [[ ! -e "${SERIAL}" ]]; then
    echo "[FAIL] ${SERIAL} is missing. Power on the A1 and reconnect USB first." >&2
    exit 2
  fi

  stop_runtime >/dev/null
  if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; then
    echo "[0/4] Starting ROS master..."
    container_run "${ROSCORE_CONTAINER}" \
      "${ros_prefix} && exec roscore"
    local deadline=$((SECONDS + 10))
    until timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; do
      if (( SECONDS >= deadline )); then
        echo "[FAIL] ROS master did not start." >&2
        exit 2
      fi
      sleep 0.5
    done
  fi

  echo "[1/4] Starting A1 driver..."
  container_run "${DRIVER_CONTAINER}" \
    "${ros_prefix} && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${SERIAL}"
  wait_valid_joint_feedback

  echo "[2/4] Starting isolated joint tracker..."
  container_run "${TRACKER_CONTAINER}" \
    "${ros_prefix} && exec roslaunch /workspace/scripts/runtime/joint_tracker_staged.launch staged_command_topic:=${STAGED_TOPIC} target_topic:=${TARGET_TOPIC}"
  wait_topic "${TRACKER_CONTAINER}" /end_effector_pose

  echo "[3/4] Starting fail-closed relay (LOCKED)..."
  container_run "${RELAY_CONTAINER}" \
    "${ros_prefix} && exec python3 /workspace/scripts/runtime/safe_arm_command_relay.py --input-topic '${STAGED_TOPIC}' --enable-topic '${RELAY_ENABLE_TOPIC}' --relay-status-topic '${RELAY_STATUS_TOPIC}'"
  wait_topic "${RELAY_CONTAINER}" "${RELAY_STATUS_TOPIC}"

  echo "[4/4] Joint runtime services ready."
  startup_complete=1
  trap - ERR
}

doctor() {
  local args=("$@")
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/runtime/a1_runtime_doctor.py" \
      --serial "${SERIAL}" \
      --staged-command-topic "${STAGED_TOPIC}" \
      --relay-status-topic "${RELAY_STATUS_TOPIC}" \
      --tracker-node "${TRACKER_NODE}" \
      "${args[@]}"
}

status() {
  echo "Joint runtime containers:"
  docker ps -a --format '{{.Names}}\t{{.Status}}' |
    grep -E "^${PREFIX}-" || echo "no ${PREFIX}-* containers"
  echo
  doctor || true
}

logs() {
  for name in "${DRIVER_CONTAINER}" "${TRACKER_CONTAINER}" "${RELAY_CONTAINER}" "${ROSCORE_CONTAINER}"; do
    echo "===== ${name} ====="
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
    cat <<EOF
Usage: $0 <start|services|stop|doctor|status|logs>

  start     Start ROS master, A1 driver, isolated joint tracker, and locked relay
  services  Alias for start
  stop      Stop A1 joint execution runtime containers
  doctor    Layered health check; add --require-execution after a target is staged
  status    Containers and doctor summary
  logs      Tail runtime logs

Environment:
  A1_SERIAL=${SERIAL}
  A1_JOINT_TARGET_TOPIC=${TARGET_TOPIC}
  A1_STAGED_COMMAND_TOPIC=${STAGED_TOPIC}
  A1_RELAY_ENABLE_TOPIC=${RELAY_ENABLE_TOPIC}
  A1_RELAY_STATUS_TOPIC=${RELAY_STATUS_TOPIC}
  A1_TRACKER_NODE=${TRACKER_NODE}
EOF
    ;;
esac
