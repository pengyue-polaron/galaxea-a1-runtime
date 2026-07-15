#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_PATH=""
SESSION="a1-camera-web"
source "${ROOT}/scripts/runtime/a1_config.sh"
source "${ROOT}/scripts/runtime/a1_tmux.sh"

if [[ "${1:-}" == "--config" ]]; then
  if [[ -z "${2:-}" ]]; then
    a1_fail "--config requires a path."
    a1_usage "$0 --config <path> <start|stop|status|logs>" >&2
    exit 2
  fi
  CONFIG_PATH="$2"
  shift 2
fi

start_viewer() {
  local config_args=(--repo-root "${ROOT}" --shell)
  if [[ -n "${CONFIG_PATH}" ]]; then
    config_args+=("${CONFIG_PATH}")
  fi
  a1_load_shell_config env \
    PYTHONPATH="${ROOT}:${PYTHONPATH:-}" uv run --project "${ROOT}" \
    python -m galaxea_a1_runtime.configuration.system "${config_args[@]}"
  a1_tmux_stop "${SESSION}"
  local command=(uv run --project "${ROOT}" python "${ROOT}/scripts/apps/cameras/a1_camera_web.py")
  if [[ -n "${CONFIG_PATH}" ]]; then
    command+=(--config "${CONFIG_PATH}")
  fi
  local command_q
  printf -v command_q "%q " "${command[@]}"
  a1_tmux_start "${SESSION}" "${ROOT}" \
    "${command_q}; rc=\$?; echo CAMERA_WEB_EXIT=\$rc; exec bash"
  a1_tmux_verify_startup \
    "${SESSION}" "CAMERA_WEB_EXIT=" "Camera web viewer" "${TMUX_STARTUP_GRACE_S}"
  a1_success "Camera viewer started in tmux ${SESSION}."
}

case "${1:-start}" in
  start) start_viewer ;;
  stop)
    a1_tmux_stop "${SESSION}"
    a1_success "Camera viewer stopped."
    ;;
  status)
    a1_tmux_status "${SESSION}"
    ;;
  logs) a1_tmux_capture "${SESSION}" 160 ;;
  help|-h|--help)
    a1_usage "$0 [--config path] <start|stop|status|logs>"
    ;;
  *)
    a1_fail "Unknown camera-web command: ${1:-}"
    a1_usage "$0 [--config path] <start|stop|status|logs>" >&2
    exit 2
    ;;
esac
