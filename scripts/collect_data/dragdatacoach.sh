#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
A1_SDK_ROOT="${A1_SDK_ROOT:-${PROJECT_ROOT}/third_party/A1_SDK}"

if [[ ! -d "${A1_SDK_ROOT}" ]]; then
  echo "A1_SDK root not found: ${A1_SDK_ROOT}"
  echo "Run: scripts/collect_data/sync_a1_sdk.sh /home/eric/A1_SDK"
  exit 1
fi

set +u
# shellcheck disable=SC1091
if [[ -f "/opt/ros/noetic/setup.bash" ]]; then
  source "/opt/ros/noetic/setup.bash"
fi
# shellcheck disable=SC1091
source "${A1_SDK_ROOT}/install/setup.bash"
set -u

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_data/dragdatacoach.sh <command> [args...]

Commands:
  launch-driver [serial]          # default serial=/dev/ttyACM0
  launch-ee-record [serial]       # driver + /end_effector_pose publisher
  drag-start [kp] [kd] [mode]
  drag-stop
  gripper-keyboard [args...]
  gripper-stop                    # stop background keyboard gripper publishers
  record-start [tag]
  record-stop
  launch-tracker
  replay --bag <path> [--rate 1.0] [--gripper-mode position]
  collect                        # start DataCoach replay bridge + cameras + collector
  doctor                         # check runtime dependencies and selected python
  which-python                   # print selected DataCoach python
EOF
}

pick_datacoach_python() {
  if [[ -n "${DATACOACH_PYTHON:-}" ]] && [[ -x "${DATACOACH_PYTHON}" ]]; then
    echo "${DATACOACH_PYTHON}"
    return 0
  fi

  local candidates=(
    "${PROJECT_ROOT}/.conda/envs/dragdatacoach/bin/python"
    "/home/pengyue/miniconda3/envs/datacoach/bin/python"
    "/home/jolia/miniconda3/envs/datacoach/bin/python"
    "$(command -v python || true)"
  )

  local py
  for py in "${candidates[@]}"; do
    [[ -n "${py}" && -x "${py}" ]] || continue
    if "${py}" - <<'PY' >/dev/null 2>&1
import importlib.util
required = ("hydra", "zmq", "cv2")
missing = [m for m in required if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
    then
      echo "${py}"
      return 0
    fi
  done
  return 1
}

check_python_deps() {
  local py="$1"
  "${py}" - <<'PY'
import importlib.util
required = ("hydra", "zmq", "cv2", "numpy", "scipy")
optional = ("lerobot", "pyrealsense2")

missing = [m for m in required if importlib.util.find_spec(m) is None]
print("Required:", ", ".join(required))
if missing:
    print("Missing required:", ", ".join(missing))
    raise SystemExit(1)
print("Missing required: <none>")

missing_optional = [m for m in optional if importlib.util.find_spec(m) is None]
if missing_optional:
    print("Missing optional:", ", ".join(missing_optional))
else:
    print("Missing optional: <none>")
PY
}

cmd="${1:-}"
case "${cmd}" in
  launch-driver)
    serial="${2:-/dev/ttyACM0}"
    echo "Using serial port: ${serial}"
    roslaunch signal_arm single_arm_node.launch "single_arm_serial_port_path:=${serial}"
    ;;
  launch-ee-record)
    serial="${2:-/dev/ttyACM0}"
    echo "Using serial port: ${serial}"
    roslaunch "${A1_SDK_ROOT}/install/share/mobiman/launch/simpleExample/ee_record_only.launch" "serial_port_path:=${serial}"
    ;;
  drag-start)
    shift
    "${A1_SDK_ROOT}/tools/a1_drag_mode.sh" start "$@"
    ;;
  drag-stop)
    "${A1_SDK_ROOT}/tools/a1_drag_mode.sh" stop
    ;;
  gripper-keyboard)
    shift
    "${A1_SDK_ROOT}/tools/a1_gripper_keyboard.py" "$@"
    ;;
  gripper-stop)
    if pgrep -f "a1_gripper_keyboard.py" >/dev/null 2>&1; then
      pkill -f "a1_gripper_keyboard.py" || true
      echo "Stopped a1_gripper_keyboard.py"
    else
      echo "No running a1_gripper_keyboard.py found."
    fi
    ;;
  record-start)
    tag="${2:-drag}"
    if command -v rostopic >/dev/null 2>&1; then
      if rostopic list >/dev/null 2>&1; then
        if ! rostopic list | grep -qx "/end_effector_pose"; then
          cat <<'EOF'
[ERROR] /end_effector_pose is not being published.
[ERROR] This recording will NOT replay arm motion (gripper-only replay).
[ERROR] Start one of these before recording:
  1) scripts/collect_data/dragdatacoach.sh launch-ee-record /dev/ttyACM0
  2) roslaunch mobiman eeTrackerdemo.launch
EOF
          if [[ "${A1_RECORD_ALLOW_NO_EE_POSE:-0}" != "1" ]]; then
            echo "[ERROR] Aborting record-start. Set A1_RECORD_ALLOW_NO_EE_POSE=1 to bypass."
            exit 1
          fi
          echo "[WARN] Bypass enabled (A1_RECORD_ALLOW_NO_EE_POSE=1)."
        fi
      fi
    fi
    "${A1_SDK_ROOT}/tools/a1_record.sh" start "${tag}"
    ;;
  record-stop)
    "${A1_SDK_ROOT}/tools/a1_record.sh" stop
    ;;
  launch-tracker)
    roslaunch mobiman eeTrackerdemo.launch
    ;;
  replay)
    shift
    if pgrep -f "a1_gripper_keyboard.py" >/dev/null 2>&1; then
      cat <<'EOF'
[ERROR] Detected running a1_gripper_keyboard.py.
[ERROR] It continuously publishes gripper commands and will conflict with bag replay.
[ERROR] Stop it first:
  scripts/collect_data/dragdatacoach.sh gripper-stop
EOF
      if [[ "${A1_REPLAY_ALLOW_GRIPPER_KEYBOARD:-0}" != "1" ]]; then
        echo "[ERROR] Aborting replay. Set A1_REPLAY_ALLOW_GRIPPER_KEYBOARD=1 to bypass."
        exit 1
      fi
      echo "[WARN] Bypass enabled (A1_REPLAY_ALLOW_GRIPPER_KEYBOARD=1)."
    fi

    if command -v rostopic >/dev/null 2>&1 && rostopic list >/dev/null 2>&1; then
      if ! rostopic list | grep -qx "/a1_ee_target"; then
        cat <<'EOF'
[WARN] /a1_ee_target topic is missing. eeTracker may not be running.
[WARN] Without eeTracker, replay can move gripper but arm will not move.
[WARN] Start:
  scripts/collect_data/dragdatacoach.sh launch-tracker
EOF
      fi
    fi

    replay_args=("$@")
    replay_bag_path=""
    for ((i=0; i<${#replay_args[@]}; i++)); do
      if [[ "${replay_args[$i]}" == "--bag" ]] && [[ $((i+1)) -lt ${#replay_args[@]} ]]; then
        replay_bag_path="${replay_args[$((i+1))]}"
        break
      fi
    done
    if [[ -n "${replay_bag_path}" ]] && [[ -f "${replay_bag_path}" ]] && command -v rosbag >/dev/null 2>&1; then
      if ! rosbag info "${replay_bag_path}" | awk '/topics:/{p=1;next} p && $1=="/end_effector_pose"{found=1} END{exit !found}'; then
        cat <<'EOF'
[WARN] Bag does not contain /end_effector_pose.
[WARN] Current replay path drives arm motion only from /end_effector_pose -> /a1_ee_target.
[WARN] Result: gripper may move, but arm trajectory will not replay.
EOF
      fi
    fi
    "${A1_SDK_ROOT}/tools/a1_replay.sh" "$@"
    ;;
  collect)
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "Could not find a usable DataCoach python interpreter."
      echo "Set DATACOACH_PYTHON explicitly, or prepare env with hydra/zmq/cv2."
      exit 1
    fi
    DATACOACH_PYTHON="${py}" "${py}" "${PROJECT_ROOT}/scripts/collect_data/run_drag_replay_collection.py"
    ;;
  doctor)
    echo "A1_SDK_ROOT=${A1_SDK_ROOT}"
    if ! command -v roscore >/dev/null 2>&1; then
      echo "ROS tools not found in PATH. Source ROS setup first."
    fi
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "No usable DataCoach python found."
      exit 1
    fi
    echo "DATACOACH_PYTHON=${py}"
    check_python_deps "${py}"
    ;;
  which-python)
    py="$(pick_datacoach_python || true)"
    if [[ -z "${py}" ]]; then
      echo "NOT_FOUND"
      exit 1
    fi
    echo "${py}"
    ;;
  *)
    usage
    exit 1
    ;;
esac
