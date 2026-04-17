#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.a1-noetic.yml"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_a1_sdk_runtime.sh"
SERVICE="a1-noetic"
DOCKER_CMD=()

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is not installed."
  exit 1
fi

if docker info >/dev/null 2>&1; then
  DOCKER_CMD=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  DOCKER_CMD=(sudo docker)
else
  cat <<'EOF'
[ERROR] Docker is installed but not usable from this shell.
[ERROR] Fix one of these:
  1) add the current user to the docker group and re-login
  2) allow passwordless sudo for docker
EOF
  exit 1
fi

compose() {
  "${DOCKER_CMD[@]}" compose -f "${COMPOSE_FILE}" "$@"
}

run_service() {
  local shell_cmd="$1"
  local cid
  cid=$(compose run -d --service-ports "${SERVICE}" bash -lc "${shell_cmd}")
  # Follow logs in foreground so nohup/start_service can track the process.
  # docker wait keeps this process alive until the container exits,
  # even when docker logs -f finishes early (no stdout).
  "${DOCKER_CMD[@]}" logs -f "${cid}" &
  "${DOCKER_CMD[@]}" wait "${cid}" >/dev/null 2>&1
}

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_data/a1_noetic_docker.sh <command> [args...]

Commands:
  prepare                 Generate third_party/A1_SDK_runtime from official arm SDK + overlay
  build                   Build the Noetic arm64 image
  shell                   Open an interactive shell inside the container
  doctor                  Show ROS / SDK paths inside the container
  roscore                 Run roscore in the container
  driver [serial]         Launch A1 single_arm_node
  ee-record [serial]      Launch record-only pipeline (driver + FK EE publisher)
  ee-tracker              Launch eeTrackerdemo
  tracker                 Launch jointTrackerdemo
EOF
}

cmd="${1:-}"
case "${cmd}" in
  prepare)
    "${PREPARE_SCRIPT}"
    ;;
  build)
    "${PREPARE_SCRIPT}"
    compose build "${SERVICE}"
    ;;
  shell)
    "${PREPARE_SCRIPT}"
    compose run --rm --service-ports "${SERVICE}" bash
    ;;
  doctor)
    "${PREPARE_SCRIPT}"
    run_service 'source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash" && echo "ROS_DISTRO=${ROS_DISTRO}" && echo "A1_SDK_ROOT=${A1_SDK_ROOT}" && rospack find signal_arm && rospack find mobiman'
    ;;
  roscore)
    "${PREPARE_SCRIPT}"
    run_service 'source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash" && exec roscore'
    ;;
  driver)
    "${PREPARE_SCRIPT}"
    serial="${2:-/dev/a1}"
    run_service "source /opt/ros/noetic/setup.bash && source \"\${A1_SDK_ROOT}/install/setup.bash\" && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${serial}"
    ;;
  ee-record)
    "${PREPARE_SCRIPT}"
    serial="${2:-/dev/a1}"
    run_service "source /opt/ros/noetic/setup.bash && source \"\${A1_SDK_ROOT}/install/setup.bash\" && exec roslaunch \"\${A1_SDK_ROOT}/install/share/mobiman/launch/simpleExample/ee_record_only.launch\" serial_port_path:=${serial}"
    ;;
  ee-tracker)
    "${PREPARE_SCRIPT}"
    run_service 'source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash" && exec roslaunch mobiman eeTrackerdemo.launch'
    ;;
  tracker)
    "${PREPARE_SCRIPT}"
    run_service 'source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash" && exec roslaunch mobiman jointTrackerdemo.launch'
    ;;
  joint-relay)
    run_service 'source /opt/ros/noetic/setup.bash && source "${A1_SDK_ROOT}/install/setup.bash" && exec python3 /workspace/scripts/inference/joint_target_relay.py'
    ;;
  bridge)
    leader_port="${2:-/dev/ttyACM0}"
    run_service "PY312=\$(uv python find 3.12) && export HF_LEROBOT_CALIBRATION=/home/nyu/.cache/huggingface/lerobot/calibration && PYTHONPATH=\"/workspace/third_party/lerobot/.venv/lib/python3.12/site-packages:/workspace/third_party/lerobot/src:/opt/ros/noetic/lib/python3/dist-packages:/usr/lib/python3/dist-packages:\${A1_SDK_ROOT}/install/lib/python3/dist-packages\" exec \$PY312 /workspace/third_party/lerobot/src/lerobot/scripts/lerobot_a1_jointtracker_bridge.py --leader-port ${leader_port} --leader-id my_leader --gripper-min-stroke-mm 0 --gripper-max-stroke-mm 200"
    ;;
  *)
    usage
    exit 1
    ;;
esac
