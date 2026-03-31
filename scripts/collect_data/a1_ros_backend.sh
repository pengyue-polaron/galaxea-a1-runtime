#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCKER_BACKEND_SCRIPT="${SCRIPT_DIR}/a1_noetic_docker.sh"
PREPARE_SCRIPT="${SCRIPT_DIR}/prepare_a1_sdk_runtime.sh"
HOST_ROS_SETUP="/opt/ros/noetic/setup.bash"
HOST_SDK_ROOT_DEFAULT="${PROJECT_ROOT}/third_party/A1_SDK"

requested_backend() {
  if [[ -n "${A1_ROS_BACKEND:-}" ]]; then
    echo "${A1_ROS_BACKEND}"
    return 0
  fi
  if [[ "${A1_USE_DOCKER:-}" == "1" ]]; then
    echo "docker"
    return 0
  fi
  if [[ "${A1_USE_DOCKER:-}" == "0" ]]; then
    echo "host"
    return 0
  fi
  echo "auto"
}

host_ready() {
  [[ -f "${HOST_ROS_SETUP}" && -f "${HOST_SDK_ROOT_DEFAULT}/install/setup.bash" ]]
}

select_backend() {
  local requested
  requested="$(requested_backend)"
  case "${requested}" in
    host|docker)
      echo "${requested}"
      ;;
    auto)
      if host_ready; then
        echo "host"
      else
        echo "docker"
      fi
      ;;
    *)
      echo "[ERROR] Unsupported A1_ROS_BACKEND=${requested}. Use auto, host, or docker." >&2
      exit 1
      ;;
  esac
}

host_sdk_root() {
  echo "${A1_SDK_ROOT:-${HOST_SDK_ROOT_DEFAULT}}"
}

run_host() {
  local cmd="$1"
  shift || true
  local sdk_root
  sdk_root="$(host_sdk_root)"

  if [[ ! -f "${HOST_ROS_SETUP}" ]]; then
    echo "[ERROR] Host ROS Noetic is not available: ${HOST_ROS_SETUP}" >&2
    echo "[ERROR] Set A1_ROS_BACKEND=docker or install host ROS Noetic." >&2
    exit 1
  fi

  if [[ ! -f "${sdk_root}/install/setup.bash" ]]; then
    echo "[ERROR] Host A1 SDK is not available: ${sdk_root}/install/setup.bash" >&2
    echo "[ERROR] Set A1_SDK_ROOT or use A1_ROS_BACKEND=docker." >&2
    exit 1
  fi

  case "${cmd}" in
    doctor)
      bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && echo 'A1_ROS_BACKEND=host' && echo 'A1_SDK_ROOT=${sdk_root}' && echo 'ROS_DISTRO=\${ROS_DISTRO}' && rospack find signal_arm && rospack find mobiman"
      ;;
    roscore)
      exec bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && exec roscore"
      ;;
    driver)
      local serial="${1:-/dev/a1}"
      exec bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && exec roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=${serial}"
      ;;
    ee-record)
      local serial="${1:-/dev/a1}"
      exec bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && exec roslaunch '${sdk_root}/install/share/mobiman/launch/simpleExample/ee_record_only.launch' serial_port_path:=${serial}"
      ;;
    ee-tracker)
      exec bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && exec roslaunch mobiman eeTrackerdemo.launch"
      ;;
    tracker)
      exec bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && exec roslaunch mobiman jointTrackerdemo.launch"
      ;;
    shell)
      exec bash -lc "source '${HOST_ROS_SETUP}' && source '${sdk_root}/install/setup.bash' && exec bash -i"
      ;;
    build|prepare)
      echo "[INFO] Host backend selected; no Docker image build required."
      ;;
    *)
      echo "[ERROR] Unsupported host command: ${cmd}" >&2
      exit 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_data/a1_ros_backend.sh <command> [args...]

Commands:
  doctor
  prepare
  build
  shell
  roscore
  driver [serial]
  ee-record [serial]
  ee-tracker
  tracker

Backend selection:
  A1_ROS_BACKEND=auto|host|docker   # default: auto
  A1_USE_DOCKER=1|0                 # legacy shortcut
EOF
}

cmd="${1:-}"
if [[ -z "${cmd}" ]]; then
  usage
  exit 1
fi
shift || true

backend="$(select_backend)"

case "${backend}" in
  host)
    run_host "${cmd}" "$@"
    ;;
  docker)
    case "${cmd}" in
      prepare|build)
        "${PREPARE_SCRIPT}"
        ;;
    esac
    exec "${DOCKER_BACKEND_SCRIPT}" "${cmd}" "$@"
    ;;
esac
