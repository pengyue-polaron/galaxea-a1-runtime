#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEM_CONFIG_PATH="${A1_SYSTEM_CONFIG_PATH:-${ROOT}/configs/system/a1.toml}"
PREFIX="${A1_RUNTIME_PREFIX:-a1-joint-runtime}"
PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
if [[ "${1:-help}" != "stop" && "${1:-help}" != "logs" ]]; then
  eval "$(
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.configuration.system \
      --repo-root "${ROOT}" --shell "${SYSTEM_CONFIG_PATH}"
  )"
fi
ROSCORE_CONTAINER="${PREFIX}-roscore"
DRIVER_CONTAINER="${PREFIX}-driver"
TRACKER_CONTAINER="${PREFIX}-joint-tracker-staged"
RELAY_CONTAINER="${PREFIX}-command-relay"
TARGET_TOPIC="${JOINT_TARGET_TOPIC}"
TRACKER_NODE="${A1_TRACKER_NODE:-/jointTracker_demo_node}"
source "${ROOT}/scripts/runtime/a1_services.sh"

stop_runtime() {
  a1_remove_runtime_containers \
    "${RELAY_CONTAINER}" \
    "${TRACKER_CONTAINER}" \
    "${DRIVER_CONTAINER}" \
    "${ROSCORE_CONTAINER}"
  a1_cleanup_shared_ros_nodes
  echo "A1 joint execution runtime stopped."
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
  echo "[0/4] Ensuring ROS master..."
  a1_ensure_roscore "${ROSCORE_CONTAINER}"

  echo "[1/4] Starting A1 driver..."
  a1_start_driver "${DRIVER_CONTAINER}"
  a1_wait_valid_joint_feedback "${DRIVER_CONTAINER}" "${JOINT_STATES_TOPIC}"

  echo "[2/4] Starting isolated joint tracker..."
  a1_container_run "${TRACKER_CONTAINER}" \
    "${A1_ROS_PREFIX} && exec roslaunch /workspace/scripts/runtime/joint_tracker_staged.launch staged_command_topic:=${STAGED_TOPIC} target_topic:=${TARGET_TOPIC}"
  a1_wait_topic "${TRACKER_CONTAINER}" "${EEF_POSE_TOPIC}"

  echo "[3/4] Starting fail-closed relay (LOCKED)..."
  a1_start_command_relay "${RELAY_CONTAINER}"
  a1_wait_topic "${RELAY_CONTAINER}" "${RELAY_STATUS_TOPIC}"

  echo "[4/4] Joint runtime services ready."
  startup_complete=1
  trap - ERR
}

doctor() {
  local args=("$@")
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/runtime/a1_runtime_doctor.py" \
      --system-config "${SYSTEM_CONFIG_PATH}" \
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
  A1_SYSTEM_CONFIG_PATH=${SYSTEM_CONFIG_PATH}
  A1_RUNTIME_PREFIX=${PREFIX}
  A1_TRACKER_NODE=${TRACKER_NODE}
EOF
    ;;
esac
