#!/usr/bin/env bash
# Shared semantic terminal colors for every shell entrypoint.

if [[ -n "${A1_CONSOLE_LOADED:-}" ]]; then
  return 0
fi
readonly A1_CONSOLE_LOADED=1

if [[ -t 1 && -z "${NO_COLOR:-}" ]]; then
  readonly A1_COLOR_INFO=$'\033[1;34m'
  readonly A1_COLOR_STEP=$'\033[1;36m'
  readonly A1_COLOR_SUCCESS=$'\033[1;32m'
  readonly A1_COLOR_RESET=$'\033[0m'
else
  readonly A1_COLOR_INFO=''
  readonly A1_COLOR_STEP=''
  readonly A1_COLOR_SUCCESS=''
  readonly A1_COLOR_RESET=''
fi

if [[ -t 2 && -z "${NO_COLOR:-}" ]]; then
  readonly A1_COLOR_WARNING=$'\033[1;33m'
  readonly A1_COLOR_FAILURE=$'\033[1;31m'
  readonly A1_COLOR_ERROR_RESET=$'\033[0m'
else
  readonly A1_COLOR_WARNING=''
  readonly A1_COLOR_FAILURE=''
  readonly A1_COLOR_ERROR_RESET=''
fi

a1_info() {
  echo "${A1_COLOR_INFO}[INFO]${A1_COLOR_RESET} $*"
}

a1_step() {
  echo "${A1_COLOR_STEP}[STEP]${A1_COLOR_RESET} $*"
}

a1_success() {
  echo "${A1_COLOR_SUCCESS}[PASS]${A1_COLOR_RESET} $*"
}

a1_warn() {
  echo "${A1_COLOR_WARNING}[WARN]${A1_COLOR_ERROR_RESET} $*" >&2
}

a1_fail() {
  echo "${A1_COLOR_FAILURE}[FAIL]${A1_COLOR_ERROR_RESET} $*" >&2
}

a1_cleanup() {
  echo "${A1_COLOR_WARNING}[CLEANUP]${A1_COLOR_ERROR_RESET} $*" >&2
}

a1_usage() {
  echo "${A1_COLOR_INFO}Usage:${A1_COLOR_RESET} $*"
}
