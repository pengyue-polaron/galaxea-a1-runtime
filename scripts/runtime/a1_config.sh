#!/usr/bin/env bash
# Shared fail-closed loader for Python-rendered shell configuration.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/a1_console.sh"

a1_load_shell_config() {
  if (( $# == 0 )); then
    a1_fail "a1_load_shell_config requires a renderer command."
    return 2
  fi

  local rendered
  if rendered="$("$@")"; then
    :
  else
    local rc=$?
    a1_fail "Configuration renderer failed (exit ${rc}): $1"
    return "${rc}"
  fi
  if [[ -z "${rendered}" ]]; then
    a1_fail "Configuration renderer returned no shell assignments: $1"
    return 2
  fi
  eval "${rendered}"
}
