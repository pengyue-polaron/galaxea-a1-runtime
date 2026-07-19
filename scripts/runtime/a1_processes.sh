#!/usr/bin/env bash
# Marked host-process lifecycle helpers for non-tmux runtime services.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a1_console.sh"

A1_PROCESS_MARKER_ENV='GALAXEA_A1_MANAGED_PROCESS'
A1_PROCESS_STATE_ROOT="${A1_PROCESS_STATE_ROOT:-${XDG_RUNTIME_DIR:-/tmp}/galaxea-a1-runtime-${UID}}"
A1_STARTED_PROCESS_PID=""

_a1_process_validate_name() {
  local name="$1"
  if [[ -z "${name}" || ! "${name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    a1_fail "Invalid managed process name: ${name:-<empty>}"
    return 2
  fi
}

a1_process_state_file() {
  local name="$1"
  _a1_process_validate_name "${name}" || return
  printf '%s/%s.pid\n' "${A1_PROCESS_STATE_ROOT}" "${name}"
}

_a1_process_read_pid() {
  local name="$1"
  local state_file
  state_file="$(a1_process_state_file "${name}")" || return
  [[ -f "${state_file}" ]] || return 1
  local pid
  IFS= read -r pid <"${state_file}" || true
  if [[ ! "${pid}" =~ ^[1-9][0-9]*$ ]]; then
    return 1
  fi
  printf '%s\n' "${pid}"
}

_a1_process_pid_is_marked() {
  local name="$1"
  local pid="$2"
  [[ -r "/proc/${pid}/environ" ]] || return 1
  tr '\0' '\n' <"/proc/${pid}/environ" 2>/dev/null |
    grep -Fxq "${A1_PROCESS_MARKER_ENV}=${name}"
}

a1_process_is_running() {
  local name="$1"
  local pid
  pid="$(_a1_process_read_pid "${name}")" || return 1
  kill -0 "${pid}" 2>/dev/null && _a1_process_pid_is_marked "${name}" "${pid}"
}

a1_process_start() {
  if (( $# < 4 )); then
    a1_fail "a1_process_start expects <name> <working-directory> <log-file> <command> [args...]."
    return 2
  fi
  local name="$1"
  local working_directory="$2"
  local log_file="$3"
  shift 3
  _a1_process_validate_name "${name}" || return
  if [[ ! -d "${working_directory}" ]]; then
    a1_fail "Managed process working directory does not exist: ${working_directory}"
    return 2
  fi
  if ! command -v setsid >/dev/null 2>&1; then
    a1_fail "setsid is required for managed host processes."
    return 2
  fi
  local state_file
  state_file="$(a1_process_state_file "${name}")" || return
  if a1_process_is_running "${name}"; then
    a1_fail "Managed process is already running: ${name}"
    return 2
  fi
  mkdir -p "${A1_PROCESS_STATE_ROOT}" "$(dirname "${log_file}")"
  rm -f "${state_file}"

  (
    cd "${working_directory}"
    exec setsid env "${A1_PROCESS_MARKER_ENV}=${name}" "$@"
  ) >"${log_file}" 2>&1 &
  local pid=$!
  local state_tmp="${state_file}.tmp.$$"
  printf '%s\n' "${pid}" >"${state_tmp}"
  mv "${state_tmp}" "${state_file}"

  local deadline=$((SECONDS + 2))
  while ! _a1_process_pid_is_marked "${name}" "${pid}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      rm -f "${state_file}"
      a1_fail "Managed process exited during startup: ${name}"
      return 2
    fi
    if (( SECONDS >= deadline )); then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      rm -f "${state_file}"
      a1_fail "Managed process did not acquire its ownership marker: ${name}"
      return 2
    fi
    sleep 0.02
  done

  local process_group
  process_group="$(ps -o pgid= -p "${pid}" | tr -d '[:space:]')"
  if [[ "${process_group}" != "${pid}" ]]; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    rm -f "${state_file}"
    a1_fail "Managed process did not become a process-group leader: ${name}"
    return 2
  fi
  A1_STARTED_PROCESS_PID="${pid}"
}

a1_process_stop() {
  if (( $# != 2 )); then
    a1_fail "a1_process_stop expects <name> <timeout-s>."
    return 2
  fi
  local name="$1"
  local timeout_s="${2%.*}"
  _a1_process_validate_name "${name}" || return
  if [[ ! "${timeout_s}" =~ ^[1-9][0-9]*$ ]]; then
    a1_fail "Managed process stop timeout must be a positive integer: ${2}"
    return 2
  fi
  local state_file
  state_file="$(a1_process_state_file "${name}")" || return
  local pid
  if ! pid="$(_a1_process_read_pid "${name}")"; then
    rm -f "${state_file}"
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${state_file}"
    return 0
  fi
  if ! _a1_process_pid_is_marked "${name}" "${pid}"; then
    a1_fail "Refusing to stop unmarked PID ${pid} from ${state_file}."
    return 2
  fi

  kill -TERM -- "-${pid}" 2>/dev/null || true
  local deadline=$((SECONDS + timeout_s))
  while kill -0 -- "-${pid}" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      a1_warn "Managed process ${name} did not stop in ${timeout_s}s; sending KILL."
      kill -KILL -- "-${pid}" 2>/dev/null || true
      break
    fi
    sleep 0.1
  done
  wait "${pid}" 2>/dev/null || true
  rm -f "${state_file}"
}

a1_process_status() {
  local name="$1"
  local pid
  if a1_process_is_running "${name}"; then
    pid="$(_a1_process_read_pid "${name}")"
    a1_success "${name}: running (pid=${pid})"
    return 0
  fi
  a1_info "${name}: not running"
  return 1
}

a1_process_stop_all_managed() {
  local timeout_s="${1:-5}"
  shift || true
  local excluded_names=("$@")
  [[ -d "${A1_PROCESS_STATE_ROOT}" ]] || return 0
  local state_file name excluded excluded_name
  local status=0
  shopt -s nullglob
  local state_files=("${A1_PROCESS_STATE_ROOT}"/*.pid)
  shopt -u nullglob
  for state_file in "${state_files[@]}"; do
    name="$(basename "${state_file}" .pid)"
    excluded=false
    for excluded_name in "${excluded_names[@]}"; do
      if [[ "${name}" == "${excluded_name}" ]]; then
        excluded=true
        break
      fi
    done
    if [[ "${excluded}" == "true" ]]; then
      continue
    fi
    a1_process_stop "${name}" "${timeout_s}" || status=1
  done
  return "${status}"
}
