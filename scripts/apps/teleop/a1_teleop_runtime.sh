#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_runtime.sh"
CONFIG_PATH="${ROOT}/configs/teleop/a1_so100.toml"
RESET_CONFIG_PATH="${ROOT}/configs/poses/a1_so100_collection_start.toml"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: $0 --config <path> <start|services|bridge|collect|reset|stop|doctor|status|logs>" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

PYTHON_BIN="${ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

eval "$(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m galaxea_a1_runtime.teleop.config \
    --repo-root "${ROOT}" \
    --shell \
    "${CONFIG_PATH}"
)"

ROSCORE_CONTAINER="${PREFIX}-roscore"
DRIVER_CONTAINER="${PREFIX}-driver"
TRACKER_CONTAINER="${PREFIX}-joint-tracker-staged"
RELAY_CONTAINER="${PREFIX}-command-relay"
LOG_DIR="${RUN_DIR}/logs"
BRIDGE_PID_FILE="${RUN_DIR}/bridge.pid"

ros_prefix='source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash"'

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  COLOR_CYAN=$'\033[1;36m'
  COLOR_GREEN=$'\033[1;32m'
  COLOR_RED=$'\033[1;31m'
  COLOR_RESET=$'\033[0m'
else
  COLOR_CYAN=""
  COLOR_GREEN=""
  COLOR_RED=""
  COLOR_RESET=""
fi

step() { echo "${COLOR_CYAN}$*${COLOR_RESET}"; }
success() { echo "${COLOR_GREEN}$*${COLOR_RESET}"; }
failure() { echo "${COLOR_RED}$*${COLOR_RESET}" >&2; }

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
    bash -lc "$*" \
    >/dev/null
}

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
      echo "[FAIL] Invalid teleop bridge pid file: ${BRIDGE_PID_FILE}" >&2
      return 1
    fi
    if ! bridge_group_has_live_process "${pid}"; then
      rm -f "${BRIDGE_PID_FILE}"
      return 0
    fi
    kill -- "-${pid}" >/dev/null 2>&1 || true
    local deadline=$((SECONDS + 5))
    while bridge_group_has_live_process "${pid}" && (( SECONDS < deadline )); do
      sleep 0.1
    done
    if bridge_group_has_live_process "${pid}"; then
      echo "[FAIL] Teleop bridge process group ${pid} did not stop within 5 seconds." >&2
      return 1
    fi
    rm -f "${BRIDGE_PID_FILE}"
  fi
}

stop_runtime() {
  stop_bridge
  docker rm -f \
    "${RELAY_CONTAINER}" \
    "${TRACKER_CONTAINER}" \
    "${DRIVER_CONTAINER}" \
    "${ROSCORE_CONTAINER}" \
    >/dev/null 2>&1 || true
  echo "A1 teleop runtime stopped."
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

check_host_inputs() {
  if [[ ! -e "${SERIAL}" ]]; then
    echo "[FAIL] A1 serial device not found: ${SERIAL}" >&2
    exit 2
  fi
  if [[ ! -e "${LEADER_PORT}" ]]; then
    echo "[FAIL] Teleop leader port not found: ${LEADER_PORT}" >&2
    ls -l /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
    exit 2
  fi
}

start_services() {
  step "[Setup] ${CONFIG_PATH}"
  check_host_inputs
  "${BASE_RUNTIME}" stop >/dev/null 2>&1 || true
  stop_runtime >/dev/null 2>&1 || true
  mkdir -p "${LOG_DIR}"

  if ! timeout 1 bash -c '</dev/tcp/127.0.0.1/11311' >/dev/null 2>&1; then
    step "[1/4] ROS master"
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

  step "[2/4] A1 driver"
  container_run "${DRIVER_CONTAINER}" \
    "${ros_prefix} && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${SERIAL}"
  wait_valid_joint_feedback

  step "[3/4] Joint tracker"
  container_run "${TRACKER_CONTAINER}" \
    "${ros_prefix} && exec roslaunch /workspace/scripts/runtime/joint_tracker_staged.launch staged_command_topic:=${STAGED_TOPIC} target_topic:=${TARGET_TOPIC}"
  wait_topic "${TRACKER_CONTAINER}" /end_effector_pose

  step "[4/4] Command relay"
  container_run "${RELAY_CONTAINER}" \
    "${ros_prefix} && exec python3 /workspace/scripts/runtime/safe_arm_command_relay.py --input-topic '${STAGED_TOPIC}' --enable-topic '${RELAY_ENABLE_TOPIC}' --relay-status-topic '${RELAY_STATUS_TOPIC}'"
  wait_topic "${RELAY_CONTAINER}" "${RELAY_STATUS_TOPIC}"

  success "[Setup] Services ready"
}

start_bridge() {
  local quiet="${1:-}"
  if [[ "${quiet}" != "--quiet" ]]; then
    step "[Setup] Teleop bridge"
  fi
  check_host_inputs
  mkdir -p "${LOG_DIR}"
  stop_bridge
  local log_file="${LOG_DIR}/bridge.log"
  : > "${log_file}"
  setsid env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/lerobot/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" "${ROOT}/scripts/apps/teleop/so100_joint_bridge.py" \
      "${BRIDGE_ARGS[@]}" \
      >> "${log_file}" 2>&1 < /dev/null &
  echo "$!" > "${BRIDGE_PID_FILE}"
  wait_bridge_live "${log_file}"
  if [[ "${quiet}" != "--quiet" ]]; then
    success "[Setup] Teleop ready"
  fi
}

wait_bridge_live() {
  local log_file="$1"
  local pid
  pid="$(cat "${BRIDGE_PID_FILE}")"
  local deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[FAIL] Teleop bridge exited during startup. Log: ${log_file}" >&2
      tail -120 "${log_file}" >&2 || true
      exit 2
    fi
    if grep -q "relay ACTIVE; teleop is live" "${log_file}" 2>/dev/null; then
      return 0
    fi
    sleep 0.5
  done
  echo "[FAIL] Teleop bridge did not become live within 15 seconds. Log: ${log_file}" >&2
  tail -120 "${log_file}" >&2 || true
  exit 2
}

doctor() {
  local args=("$@")
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/lerobot/src:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python - "${LEADER_PORT}" <<'PY'
import importlib.util
import sys
from pathlib import Path

leader = Path(sys.argv[1])
checks = {
    "leader_port": leader.exists(),
    "so_leader_import": importlib.util.find_spec("lerobot.teleoperators.so_leader") is not None,
    "a1_so_leader_import": importlib.util.find_spec("galaxea_a1_runtime.teleop.a1_so_leader") is not None,
    "ros_import": importlib.util.find_spec("rospy") is not None,
    "signal_arm_import": importlib.util.find_spec("signal_arm") is not None,
}
for name, ok in checks.items():
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {leader if name == 'leader_port' else ''}")
raise SystemExit(0 if all(checks.values()) else 1)
PY
  A1_SERIAL="${SERIAL}" \
    A1_STAGED_COMMAND_TOPIC="${STAGED_TOPIC}" \
    A1_RELAY_ENABLE_TOPIC="${RELAY_ENABLE_TOPIC}" \
    A1_RELAY_STATUS_TOPIC="${RELAY_STATUS_TOPIC}" \
    A1_TRACKER_NODE="/jointTracker_demo_node" \
    "${BASE_RUNTIME}" doctor "${args[@]}"
}

collect() {
  local experiment="${1:-}"
  shift || true
  if [[ -z "${experiment}" ]]; then
    echo "Usage: $0 collect <experiment>" >&2
    exit 2
  fi
  if [[ "$#" -gt 0 ]]; then
    echo "[FAIL] Per-run teleop collector args are disabled. Edit ${CONFIG_PATH} instead." >&2
    exit 2
  fi
  cleanup_collect() {
    echo "[collect] stopping teleop runtime..."
    stop_runtime >/dev/null 2>&1 || true
  }
  trap cleanup_collect EXIT
  start_services
  start_bridge
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/teleop_collect.py" \
      --experiment "${experiment}" \
      --reset-runtime-script "${ROOT}/scripts/apps/teleop/a1_teleop_runtime.sh" \
      --teleop-config "${CONFIG_PATH}" \
      "${COLLECT_ARGS[@]}"
}

reset_live() {
  step "[Reset] Pausing teleop"
  stop_bridge
  if ! PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/lerobot/src:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/a1_so100_reset.py" \
      --config "${RESET_CONFIG_PATH}"; then
    failure "[Reset] Failed; teleop remains stopped"
    return 1
  fi
  start_bridge --quiet
  success "[Reset] Teleop ready"
}

reset() {
  cleanup_reset() {
    stop_runtime >/dev/null 2>&1 || true
  }
  trap cleanup_reset EXIT
  start_services
  PYTHONPATH="${ROOT}/third_party/A1_SDK/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${ROOT}/third_party/lerobot/src:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/teleop/a1_so100_reset.py" \
      --config "${RESET_CONFIG_PATH}"
}

status() {
  echo "Teleop containers:"
  docker ps -a --format '{{.Names}}\t{{.Status}}' |
    grep -E "^${PREFIX}-" || echo "no ${PREFIX}-* containers"
  echo
  local bridge_pid=""
  if [[ -f "${BRIDGE_PID_FILE}" ]]; then
    bridge_pid="$(cat "${BRIDGE_PID_FILE}")"
  fi
  if [[ "${bridge_pid}" =~ ^[1-9][0-9]*$ ]] && bridge_group_has_live_process "${bridge_pid}"; then
    echo "bridge: running pid=${bridge_pid}"
  else
    echo "bridge: not running"
  fi
}

logs() {
  for name in "${DRIVER_CONTAINER}" "${TRACKER_CONTAINER}" "${RELAY_CONTAINER}" "${ROSCORE_CONTAINER}"; do
    echo "===== ${name} ====="
    docker logs --tail "${A1_LOG_TAIL:-120}" "${name}" 2>&1 || true
  done
  echo "===== teleop bridge ====="
  tail -n "${A1_LOG_TAIL:-120}" "${LOG_DIR}/bridge.log" 2>/dev/null || true
}

camera_diagnostics() {
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/cameras/a1_camera_diagnostics.py" \
      --config "${CONFIG_PATH}" \
      "$@"
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
  cameras)
    shift
    camera_diagnostics "$@"
    ;;
  *)
    cat <<EOF
Usage: $0 [--config configs/teleop/a1_so100.toml] <start|services|bridge|collect|reset|stop|doctor|status|logs|cameras>

  start     Start staged joint teleop services and SO leader bridge
  services  Start ROS master, A1 driver, staged joint tracker, locked relay
  bridge    Start only the SO leader bridge
  collect   Start teleop, then run the interactive recorder
  reset     Reset A1 and SO leader to configs/poses/a1_so100_collection_start.toml
  stop      Stop bridge and teleop containers
  doctor    Static/import checks plus base runtime doctor
  status    Containers and bridge process state
  logs      Runtime and bridge logs
  cameras   Capture front/wrist/depth snapshots from the tracked teleop config

Config:
  ${CONFIG_PATH}

Important values:
  A1 serial       ${SERIAL}
  SO leader port  ${LEADER_PORT}
  SO leader id    ${LEADER_ID}
  Gripper stroke  ${GRIPPER_MIN_STROKE_MM}..${GRIPPER_MAX_STROKE_MM} mm
EOF
    ;;
esac
