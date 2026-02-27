#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SESSION="dragdatacoach"
SERIAL="/dev/ttyACM0"
TAG="drag_demo"
BAG=""
RATE="1.0"
GRIPPER_MODE="position"
SKIP_RECORD=0
WITH_GRIPPER_KEYBOARD=1
AUTO_STOP=1
ON_EXISTING="restart"

usage() {
  cat <<'EOF'
Usage:
  scripts/collect_data/dragdatacoach_all_in_one.sh [options]

Options:
  --serial <path>            Serial device for A1 driver (default: /dev/ttyACM0)
  --tag <name>               Bag record tag for drag stage (default: drag_demo)
  --bag <path>               Replay bag path. If omitted, use latest recorded bag.
  --rate <float>             Replay rate (default: 1.0)
  --gripper-mode <mode>      Replay gripper mode (default: position)
  --session <name>           tmux session name (default: dragdatacoach)
  --skip-record              Skip drag recording stage, replay from provided/latest bag.
  --no-gripper-keyboard      Do not launch keyboard gripper during drag stage.
  --no-auto-stop             Keep replay launch windows running at the end.
  --on-existing <policy>     Existing tmux session policy:
                             ask | restart | attach | new | abort
                             (default: restart)
  -h, --help                 Show this help.
EOF
}

next_session_name() {
  local base="$1"
  local idx=2
  local candidate
  candidate="${base}_${idx}"
  while tmux has-session -t "${candidate}" 2>/dev/null; do
    idx=$((idx + 1))
    candidate="${base}_${idx}"
  done
  echo "${candidate}"
}

handle_existing_session() {
  local base_session="$1"
  local policy="$2"
  local choice=""

  case "${policy}" in
    attach)
      tmux attach -t "${base_session}"
      exit 0
      ;;
    restart)
      echo "[INFO] Removing existing session: ${base_session}"
      tmux kill-session -t "${base_session}"
      return
      ;;
    new)
      SESSION="$(next_session_name "${base_session}")"
      echo "[INFO] Existing session detected, switching to: ${SESSION}"
      return
      ;;
    abort)
      echo "tmux session '${base_session}' already exists."
      echo "Use --on-existing <restart|attach|new>."
      exit 1
      ;;
    ask)
      if [[ -t 0 && -t 1 ]]; then
        echo "tmux session '${base_session}' already exists."
        echo "Choose action: [r]estart, [a]ttach, [n]ew, [q]uit"
        read -r -p "Action (default: r): " choice
      else
        choice="r"
      fi

      case "${choice}" in
        ""|r|R)
          echo "[INFO] Removing existing session: ${base_session}"
          tmux kill-session -t "${base_session}"
          ;;
        a|A)
          tmux attach -t "${base_session}"
          exit 0
          ;;
        n|N)
          SESSION="$(next_session_name "${base_session}")"
          echo "[INFO] Existing session detected, switching to: ${SESSION}"
          ;;
        q|Q)
          echo "Abort."
          exit 1
          ;;
        *)
          echo "Unknown choice: ${choice}"
          exit 1
          ;;
      esac
      ;;
    *)
      echo "Invalid --on-existing policy: ${policy}"
      echo "Valid values: ask, restart, attach, new, abort"
      exit 1
      ;;
  esac
}

pick_latest_bag() {
  ls -t "${PROJECT_ROOT}"/third_party/A1_SDK/data/records/*.bag 2>/dev/null | head -n 1 || true
}

pick_latest_bag_since() {
  local since_ts="$1"
  local tag="$2"
  local bag=""
  local best_m=0
  local f
  local m
  local name

  shopt -s nullglob
  for f in "${PROJECT_ROOT}"/third_party/A1_SDK/data/records/*.bag; do
    name="$(basename "${f}")"
    if [[ "${name}" != *"${tag}"* ]]; then
      continue
    fi
    m="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
    if (( m >= since_ts && m > best_m )); then
      best_m="${m}"
      bag="${f}"
    fi
  done

  if [[ -z "${bag}" ]]; then
    for f in "${PROJECT_ROOT}"/third_party/A1_SDK/data/records/*.bag; do
      m="$(stat -c %Y "${f}" 2>/dev/null || echo 0)"
      if (( m >= since_ts && m > best_m )); then
        best_m="${m}"
        bag="${f}"
      fi
    done
  fi
  shopt -u nullglob

  echo "${bag}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial)
      SERIAL="$2"
      shift 2
      ;;
    --tag)
      TAG="$2"
      shift 2
      ;;
    --bag)
      BAG="$2"
      shift 2
      ;;
    --rate)
      RATE="$2"
      shift 2
      ;;
    --gripper-mode)
      GRIPPER_MODE="$2"
      shift 2
      ;;
    --session)
      SESSION="$2"
      shift 2
      ;;
    --skip-record)
      SKIP_RECORD=1
      shift
      ;;
    --no-gripper-keyboard)
      WITH_GRIPPER_KEYBOARD=0
      shift
      ;;
    --no-auto-stop)
      AUTO_STOP=0
      shift
      ;;
    --on-existing)
      ON_EXISTING="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v just >/dev/null 2>&1; then
  echo "just is not installed or not in PATH."
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed. Install tmux first."
  exit 1
fi

cd "${PROJECT_ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  handle_existing_session "${SESSION}" "${ON_EXISTING}"
fi

tmux_new_window() {
  local window_name="$1"
  local command="$2"
  tmux new-window -t "${SESSION}:" -n "${window_name}" "cd '${PROJECT_ROOT}'; ${command}"
}

tmux_ctrl_c() {
  local window_name="$1"
  tmux send-keys -t "${SESSION}:${window_name}" C-c || true
}

collect_cmd="just collect"
if ! id -nG | grep -qw video; then
  video_members="$(getent group video | awk -F: '{print $4}' || true)"
  if [[ ",${video_members}," == *",${USER},"* ]]; then
    collect_cmd="sg video -c 'cd \"${PROJECT_ROOT}\" && just collect'"
  else
    echo "[WARN] Current user is not in video group. Camera permissions may fail."
    echo "[WARN] Run: sudo usermod -aG video ${USER} && relogin"
  fi
fi

tmux new-session -d -s "${SESSION}" -n control "cd '${PROJECT_ROOT}'; bash"

echo "[Stage 1/2] Drag Record"
if [[ "${SKIP_RECORD}" -eq 0 ]]; then
  tmux_new_window "record_driver" "just launch ee-record '${SERIAL}'"
  sleep 3

  # Previous interrupted runs may leave drag mode alive outside tmux.
  # Always clear it before starting a new drag session.
  just drag stop >/dev/null 2>&1 || true
  if [[ "${WITH_GRIPPER_KEYBOARD}" -eq 1 ]]; then
    A1_DRAG_HOLD_GRIPPER_POSITION=0 just drag start
  else
    just drag start
  fi

  read -r -p "Press Enter to START bag recording..."
  record_start_ts="$(date +%s)"
  just record start "${TAG}"

  if [[ "${WITH_GRIPPER_KEYBOARD}" -eq 1 ]]; then
    echo "Recording now. Control gripper with keyboard in THIS terminal."
    echo "Press Enter to stop recording and exit gripper control."
    just gripper start --quit-on-enter || true
    just gripper stop || true
    just record stop || true
  else
    echo "Recording now. Move the arm. Press Enter when finished."
    read -r -p "Press Enter to STOP bag recording..."
    just record stop || true
  fi
  just drag stop || true
  just gripper stop || true

  if [[ -z "${BAG}" ]]; then
    BAG="$(pick_latest_bag_since "${record_start_ts}" "${TAG}")"
    if [[ -z "${BAG}" ]]; then
      BAG="$(pick_latest_bag)"
    fi
  fi
  echo "Recorded bag: ${BAG}"

  tmux_ctrl_c "record_driver"
  sleep 2
else
  echo "Skip record stage enabled."
fi

if [[ -z "${BAG}" ]]; then
  BAG="$(pick_latest_bag)"
fi

if [[ -z "${BAG}" || ! -f "${BAG}" ]]; then
  echo "Replay bag not found: ${BAG}"
  exit 1
fi

echo "[Stage 2/2] Replay + Collect"
echo "Using bag: ${BAG}"

tmux_new_window "replay_driver" "just launch ee-record '${SERIAL}'"
sleep 2
tmux_new_window "tracker" "just launch tracker"
sleep 2
scripts/collect_data/dragdatacoach.sh require-cameras "replay"
tmux_new_window "collect" "${collect_cmd}"
sleep 4
read -r -p "Press Enter to START replay..."
tmux send-keys -t "${SESSION}:collect" Enter
sleep 1

A1_SKIP_CAMERA_PREFLIGHT=1 just replay "${BAG}" "${RATE}" "${GRIPPER_MODE}"

echo "Replay finished. Stopping collector..."
tmux_ctrl_c "collect"
sleep 2
tmux_ctrl_c "replay_driver"
sleep 2
tmux send-keys -t "${SESSION}:replay_driver" "just launch driver '${SERIAL}'" Enter
sleep 2

if [[ "${AUTO_STOP}" -eq 1 ]]; then
  tmux_ctrl_c "tracker"
  tmux_ctrl_c "replay_driver"
fi

echo
echo "All-in-one flow finished."
echo "tmux session: ${SESSION}"
echo "Attach logs: tmux attach -t ${SESSION}"
echo "Cleanup:     tmux kill-session -t ${SESSION}"
