#!/usr/bin/env bash
# Shared tmux lifecycle primitives. Callers own shell options and startup policy.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a1_console.sh"

A1_MANAGED_TMUX_OPTION='@galaxea_a1_runtime_managed'

a1_tmux_has_session() {
  tmux has-session -t "$1" 2>/dev/null
}

a1_tmux_stop() {
  tmux kill-session -t "$1" >/dev/null 2>&1 || true
}

a1_tmux_start() {
  if (( $# != 3 )); then
    a1_fail "a1_tmux_start expects <session> <working-directory> <command>."
    return 2
  fi
  local session="$1"
  local working_directory="$2"
  local command="$3"
  if ! command -v tmux >/dev/null 2>&1; then
    a1_fail "tmux is not installed."
    return 2
  fi
  if [[ ! -d "${working_directory}" ]]; then
    a1_fail "tmux working directory does not exist: ${working_directory}"
    return 2
  fi
  if [[ -z "${session}" || -z "${command}" ]]; then
    a1_fail "tmux session and command must not be empty."
    return 2
  fi
  a1_tmux_stop "${session}"
  tmux new-session -d -s "${session}" -c "${working_directory}" "${command}"
  if ! tmux set-option -t "${session}" "${A1_MANAGED_TMUX_OPTION}" true; then
    a1_tmux_stop "${session}"
    a1_fail "Could not mark tmux session as A1-runtime managed: ${session}"
    return 2
  fi
}

a1_tmux_capture() {
  local session="$1"
  local lines="${2:-80}"
  tmux capture-pane -pt "${session}" -S "-${lines}" 2>/dev/null
}

a1_tmux_verify_startup() {
  if (( $# < 4 || $# > 5 )); then
    a1_fail "a1_tmux_verify_startup expects <session> <exit-marker> <label> <grace-s> [log-lines]."
    return 2
  fi
  local session="$1"
  local exit_marker="$2"
  local label="$3"
  local grace_s="$4"
  local log_lines="${5:-80}"
  if [[ ! "${grace_s}" =~ ^[1-9][0-9]*$ ]]; then
    a1_fail "tmux startup grace must be a positive integer, got: ${grace_s}"
    return 2
  fi
  local deadline=$((SECONDS + grace_s))
  local pane=""
  while (( SECONDS < deadline )); do
    if ! a1_tmux_has_session "${session}"; then
      a1_fail "${label} tmux session exited during startup."
      return 2
    fi
    pane="$(a1_tmux_capture "${session}" "${log_lines}" || true)"
    if grep -Fq "${exit_marker}" <<<"${pane}"; then
      printf '%s\n' "${pane}"
      a1_fail "${label} exited during startup."
      return 2
    fi
    sleep 0.2
  done
  pane="$(a1_tmux_capture "${session}" "${log_lines}" || true)"
  [[ -z "${pane}" ]] || printf '%s\n' "${pane}"
}

a1_tmux_wait_for_http_health() {
  if (( $# < 5 || $# > 6 )); then
    a1_fail "a1_tmux_wait_for_http_health expects <session> <health-url> <exit-marker> <label> <timeout-s> [log-lines]."
    return 2
  fi
  local session="$1"
  local health_url="$2"
  local exit_marker="$3"
  local label="$4"
  local timeout_s="$5"
  local log_lines="${6:-80}"
  if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
    a1_fail "HTTP health timeout must be a positive integer, got: ${timeout_s}"
    return 2
  fi
  local deadline=$((SECONDS + timeout_s))
  local pane=""
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 1 "${health_url}" >/dev/null 2>&1; then
      return 0
    fi
    if ! a1_tmux_has_session "${session}"; then
      a1_fail "${label} tmux session exited during startup."
      return 2
    fi
    pane="$(a1_tmux_capture "${session}" 20 || true)"
    if grep -Fq "${exit_marker}" <<<"${pane}"; then
      a1_tmux_capture "${session}" "${log_lines}" >&2 || true
      a1_fail "${label} process exited during startup."
      return 2
    fi
    sleep 1
  done
  a1_tmux_capture "${session}" "${log_lines}" >&2 || true
  a1_fail "${label} did not become healthy within ${timeout_s}s."
  return 2
}

a1_tmux_status() {
  local session="$1"
  if a1_tmux_has_session "${session}"; then
    a1_success "$(tmux display-message -p -t "${session}" '#S: #{session_windows} windows (created #{t:session_created})')"
    return 0
  fi
  a1_info "${session}: not running"
  return 1
}

a1_tmux_stop_all_managed() {
  if ! command -v tmux >/dev/null 2>&1; then
    return 0
  fi
  local session marker
  local sessions
  if ! sessions="$(tmux list-sessions -F '#S' 2>/dev/null)"; then
    # No tmux server is the normal fully-stopped state.
    return 0
  fi
  local failed=0
  while IFS= read -r session; do
    [[ -n "${session}" ]] || continue
    marker="$(
      tmux show-options -v -t "${session}" "${A1_MANAGED_TMUX_OPTION}" 2>/dev/null || true
    )"
    if [[ "${marker}" == "true" ]]; then
      if ! tmux kill-session -t "${session}" >/dev/null 2>&1; then
        a1_fail "Could not stop managed tmux session: ${session}"
        failed=1
      fi
    fi
  done <<<"${sessions}"
  return "${failed}"
}
