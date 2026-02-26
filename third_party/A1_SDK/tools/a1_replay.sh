#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ORIG_ARGS=("$@")
CURRENT_USER="${USER:-$(id -un)}"

if [[ -f "${SDK_ROOT}/install/setup.bash" ]]; then
  set +u
  set --
  # shellcheck disable=SC1091
  source "${SDK_ROOT}/install/setup.bash"
  set -- "${ORIG_ARGS[@]}"
  set -u
else
  set -u
fi

print_usage() {
  cat <<'EOF'
Usage:
  tools/a1_replay.sh --bag <path> [--rate 1.0] [--pause] [--gripper-mode auto]

Behavior:
  - Replays /end_effector_pose as /a1_ee_target.
  - Replays one gripper control topic to avoid conflicts (auto priority):
      1) /gripper_position_control_host
      2) /gripper_command_host
      3) /gripper_force_control_host

Options:
  --gripper-mode auto|position|command|force|none
EOF
}

contains_topic() {
  local target="$1"
  local t
  for t in "${BAG_TOPICS[@]}"; do
    if [[ "${t}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

pid_is_drag_proc() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  local cmdline
  cmdline="$(tr '\0' ' ' <"/proc/${pid}/cmdline" 2>/dev/null || true)"
  [[ "${cmdline}" == *"a1_drag_compliance.py"* ]]
}

BAG_PATH=""
RATE="1.0"
PAUSE="0"
GRIPPER_MODE="auto"
ALLOW_DRAG_CONFLICT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bag)
      BAG_PATH="${2:-}"
      shift 2
      ;;
    --rate)
      RATE="${2:-}"
      shift 2
      ;;
    --pause)
      PAUSE="1"
      shift
      ;;
    --gripper-mode)
      GRIPPER_MODE="${2:-}"
      shift 2
      ;;
    --allow-drag-conflict)
      ALLOW_DRAG_CONFLICT="1"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      print_usage
      exit 1
      ;;
  esac
done

if [[ -z "${BAG_PATH}" ]]; then
  echo "--bag is required."
  print_usage
  exit 1
fi

BAG_PATH="$(realpath "${BAG_PATH}")"
if [[ "${BAG_PATH}" == *.bag.active ]]; then
  echo "Bag is still recording (.bag.active). Stop recorder first."
  exit 1
fi
if [[ ! -f "${BAG_PATH}" ]]; then
  echo "Bag file not found: ${BAG_PATH}"
  exit 1
fi

if ! rostopic list >/dev/null 2>&1; then
  echo "ROS master is not reachable. Start roscore and required nodes first."
  exit 1
fi

DRAG_PID_FILE="/tmp/a1_drag_mode_${CURRENT_USER}.pid"
LEGACY_DRAG_PID_FILE="/tmp/a1_drag_mode.pid"
drag_running=0
for f in "${DRAG_PID_FILE}" "${LEGACY_DRAG_PID_FILE}"; do
  [[ -f "${f}" ]] || continue
  pid="$(cat "${f}" 2>/dev/null || true)"
  if pid_is_drag_proc "${pid}"; then
    drag_running=1
    break
  fi
done

if [[ "${drag_running}" == "1" ]]; then
  if [[ "${ALLOW_DRAG_CONFLICT}" != "1" ]]; then
    echo "Drag mode is running and may override replayed gripper commands."
    echo "Run './tools/a1_drag_mode.sh stop' first, or pass --allow-drag-conflict to ignore."
    exit 1
  fi
fi

mapfile -t BAG_TOPICS < <(rosbag info "${BAG_PATH}" | awk '/topics:/{p=1;next} p && $1 ~ /^\// {print $1}')
if [[ ${#BAG_TOPICS[@]} -eq 0 ]]; then
  echo "No topics found in bag: ${BAG_PATH}"
  exit 1
fi

REPLAY_TOPICS=()
if contains_topic "/end_effector_pose"; then
  REPLAY_TOPICS+=("/end_effector_pose")
fi

SELECTED_GRIPPER_TOPIC=""
case "${GRIPPER_MODE}" in
  auto)
    if contains_topic "/gripper_position_control_host"; then
      SELECTED_GRIPPER_TOPIC="/gripper_position_control_host"
    elif contains_topic "/gripper_command_host"; then
      SELECTED_GRIPPER_TOPIC="/gripper_command_host"
    elif contains_topic "/gripper_force_control_host"; then
      SELECTED_GRIPPER_TOPIC="/gripper_force_control_host"
    fi
    ;;
  position)
    if contains_topic "/gripper_position_control_host"; then
      SELECTED_GRIPPER_TOPIC="/gripper_position_control_host"
    else
      echo "Requested --gripper-mode position but topic not found in bag."
      exit 1
    fi
    ;;
  command)
    if contains_topic "/gripper_command_host"; then
      SELECTED_GRIPPER_TOPIC="/gripper_command_host"
    else
      echo "Requested --gripper-mode command but topic not found in bag."
      exit 1
    fi
    ;;
  force)
    if contains_topic "/gripper_force_control_host"; then
      SELECTED_GRIPPER_TOPIC="/gripper_force_control_host"
    else
      echo "Requested --gripper-mode force but topic not found in bag."
      exit 1
    fi
    ;;
  none)
    ;;
  *)
    echo "Invalid --gripper-mode: ${GRIPPER_MODE}"
    print_usage
    exit 1
    ;;
esac

if [[ -n "${SELECTED_GRIPPER_TOPIC}" ]]; then
  REPLAY_TOPICS+=("${SELECTED_GRIPPER_TOPIC}")
fi

if [[ ${#REPLAY_TOPICS[@]} -eq 0 ]]; then
  echo "No replayable arm/gripper topics found in bag."
  exit 1
fi

echo "Replay bag: ${BAG_PATH}"
echo "Replay rate: ${RATE}"
echo "Replay topics:"
printf '  %s\n' "${REPLAY_TOPICS[@]}"
echo "Remap: /end_effector_pose -> /a1_ee_target (if present)"
if [[ -n "${SELECTED_GRIPPER_TOPIC}" ]]; then
  echo "Selected gripper topic: ${SELECTED_GRIPPER_TOPIC}"
else
  echo "Selected gripper topic: <none>"
fi

CMD=(rosbag play "${BAG_PATH}" -r "${RATE}" --topics "${REPLAY_TOPICS[@]}")
if [[ "${PAUSE}" == "1" ]]; then
  CMD+=(--pause)
fi
if contains_topic "/end_effector_pose"; then
  CMD+=("/end_effector_pose:=/a1_ee_target")
fi

exec "${CMD[@]}"
