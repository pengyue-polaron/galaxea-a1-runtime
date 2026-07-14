#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH="${ROOT}/configs/teleop/a1_so100.toml"
SESSION="a1-camera-web"

if [[ "${1:-}" == "--config" ]]; then
  CONFIG_PATH="$2"
  shift 2
fi

eval "$(
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" uv run --project "${ROOT}" \
    python -m galaxea_a1_runtime.teleop.config --repo-root "${ROOT}" --shell "${CONFIG_PATH}"
)"

start_viewer() {
  if ss -H -ltn "sport = :${WEB_PREVIEW_PORT}" | grep -q .; then
    echo "[FAIL] Camera web port ${WEB_PREVIEW_PORT} is already in use. Stop the active Teleop/ACT/LingBot/viewer owner first." >&2
    exit 2
  fi
  tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
  local command=(uv run --project "${ROOT}" python "${ROOT}/scripts/apps/cameras/a1_camera_web.py" --config "${CONFIG_PATH}")
  local command_q
  printf -v command_q "%q " "${command[@]}"
  tmux new-session -d -s "${SESSION}" -c "${ROOT}" "${command_q}; rc=\$?; echo CAMERA_WEB_EXIT=\$rc; exec bash"
  sleep 4
  local pane
  pane="$(tmux capture-pane -pt "${SESSION}" -S -80 2>/dev/null || true)"
  printf '%s\n' "${pane}"
  if grep -q "CAMERA_WEB_EXIT=" <<<"${pane}"; then
    echo "[FAIL] Camera web viewer exited during startup." >&2
    exit 2
  fi
  echo "Camera viewer started in tmux ${SESSION}."
}

case "${1:-start}" in
  start) start_viewer ;;
  stop)
    tmux kill-session -t "${SESSION}" >/dev/null 2>&1 || true
    echo "Camera viewer stopped."
    ;;
  status)
    tmux list-sessions 2>/dev/null | grep "${SESSION}" || echo "${SESSION}: not running"
    ss -H -ltn "sport = :${WEB_PREVIEW_PORT}" || true
    ;;
  logs) tmux capture-pane -pt "${SESSION}" -S -160 ;;
  *)
    echo "Usage: $0 [--config path] <start|stop|status|logs>" >&2
    exit 2
    ;;
esac
