#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUNTIME="${ROOT}/scripts/runtime/a1_runtime.sh"
SESSION="${A1_LINGBOT_TMUX:-lingbot-a1}"
WRIST_CAMERA="${A1_WRIST_CAMERA:-/dev/v4l/by-id/usb-Global_Shutter_Camera_Global_Shutter_Camera_01.00.00-video-index0}"
PROMPT="${A1_LINGBOT_PROMPT:-pick up that bowl}"
LINGBOT_HOST="${A1_LINGBOT_HOST:-127.0.0.1}"
LINGBOT_PORT="${A1_LINGBOT_PORT:-1106}"
BRIDGE_EXTRA_ARGS="${A1_LINGBOT_BRIDGE_EXTRA_ARGS:-}"

check_lingbot_app() {
  if [[ ! -e "${WRIST_CAMERA}" ]]; then
    echo "[FAIL] Wrist camera not found: ${WRIST_CAMERA}" >&2
    exit 2
  fi
  if ! uv run --project "${ROOT}" python - "${LINGBOT_HOST}" "${LINGBOT_PORT}" >/dev/null 2>&1 <<'PY'
import sys
import websockets.sync.client

host = sys.argv[1]
port = int(sys.argv[2])
with websockets.sync.client.connect(
    f"ws://{host}:{port}",
    compression=None,
    max_size=None,
    ping_interval=None,
    close_timeout=2,
    open_timeout=2,
) as ws:
    ws.recv(timeout=2)
PY
  then
    echo "[FAIL] LingBot server is not listening on ${LINGBOT_HOST}:${LINGBOT_PORT}." >&2
    exit 2
  fi
}

start_services() {
  "${BASE_RUNTIME}" services
}

start_tmux() {
  check_lingbot_app
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
    "bash -lc 'PYTHONPATH=\"${ROOT}/third_party/A1_SDK_runtime/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:\${PYTHONPATH:-}\" uv run --project \"${ROOT}\" python \"${ROOT}/scripts/apps/lingbot/lingbot_va_ee_bridge.py\" --host \"${LINGBOT_HOST}\" --port \"${LINGBOT_PORT}\" --prompt \"${PROMPT}\" --step-actions --execute-frames 1 --max-model-calls 0 --orientation-mode hold-current --execute --cam1-device \"${WRIST_CAMERA}\" ${BRIDGE_EXTRA_ARGS}; rc=\$?; echo BRIDGE_EXIT=\$rc; exec bash'"
  sleep 4
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[FAIL] tmux session exited during startup." >&2
    exit 2
  fi
  tmux capture-pane -pt "${SESSION}" -S -40
}

doctor() {
  local args=("$@")
  "${BASE_RUNTIME}" doctor "${args[@]}"
  PYTHONPATH="${ROOT}/third_party/A1_SDK_runtime/install/lib/python3/dist-packages:${ROOT}/.cache/ros1_python_overlay:${PYTHONPATH:-}" \
    uv run --project "${ROOT}" python "${ROOT}/scripts/apps/lingbot/a1_lingbot_doctor.py" \
      --lingbot-host "${LINGBOT_HOST}" \
      --lingbot-port "${LINGBOT_PORT}" \
      --wrist-camera "${WRIST_CAMERA}" \
      "${args[@]}"
}

stop_runtime() {
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  "${BASE_RUNTIME}" stop
  echo "LingBot A1 bridge stopped."
}

status() {
  "${BASE_RUNTIME}" status
  echo
  echo "tmux:"
  tmux list-sessions 2>/dev/null | grep "${SESSION}" || echo "${SESSION}: not running"
}

case "${1:-help}" in
  start)
    check_lingbot_app
    start_services
    start_tmux
    echo
    echo "Attach with: tmux attach -t ${SESSION}"
    ;;
  services)
    start_services
    ;;
  tmux)
    start_tmux
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
    "${BASE_RUNTIME}" logs
    ;;
  *)
    cat <<EOF
Usage: $0 <start|services|tmux|stop|doctor|status|logs>

  start     Start A1 base runtime, then open the interactive LingBot bridge tmux
  services  Start only the decoupled A1 base runtime
  tmux      Start only the interactive LingBot bridge
  stop      Stop the LingBot tmux and A1 base runtime
  doctor    Run base runtime checks plus LingBot app checks
  status    Base runtime status plus tmux state
  logs      Tail base runtime logs

Environment:
  A1_LINGBOT_BRIDGE_EXTRA_ARGS  Advanced flags passed directly to lingbot_va_ee_bridge.py.
EOF
    ;;
esac
