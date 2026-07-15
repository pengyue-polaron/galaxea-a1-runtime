#!/usr/bin/env bash
# Shared tmux lifecycle primitives. Callers own shell options and startup policy.

a1_tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

a1_tmux_stop() {
  tmux kill-session -t "$1" >/dev/null 2>&1 || true
}

a1_tmux_start() {
  local session="$1"
  local working_directory="$2"
  local command="$3"
  a1_tmux_stop "${session}"
  tmux new-session -d -s "${session}" -c "${working_directory}" "${command}"
}

a1_tmux_capture() {
  local session="$1"
  local lines="${2:-80}"
  tmux capture-pane -pt "${session}" -S "-${lines}" 2>/dev/null
}

a1_tmux_status() {
  local session="$1"
  tmux list-sessions 2>/dev/null | grep -F "${session}" || echo "${session}: not running"
}
