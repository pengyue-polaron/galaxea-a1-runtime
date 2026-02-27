#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <status-file> <pid-file> <log-file>" >&2
  exit 2
fi

STATUS_FILE="$1"
PID_FILE="$2"
LOG_FILE="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "$(dirname "${STATUS_FILE}")"
mkdir -p "$(dirname "${LOG_FILE}")"
rm -f "${STATUS_FILE}" "${PID_FILE}"

py="$(scripts/collect_data/dragdatacoach.sh which-python || true)"
if [[ -z "${py}" || "${py}" == "NOT_FOUND" ]]; then
  echo "Could not find a usable DataCoach python interpreter." | tee -a "${LOG_FILE}" >&2
  echo "Set DATACOACH_PYTHON explicitly, or prepare env with hydra/zmq/cv2." | tee -a "${LOG_FILE}" >&2
  printf '1\n' > "${STATUS_FILE}"
  exit 1
fi

set +e
DATACOACH_PYTHON="${py}" "${py}" "${PROJECT_ROOT}/scripts/collect_data/run_drag_replay_collection.py" \
  < /dev/tty > >(tee -a "${LOG_FILE}") 2>&1 &
child_pid=$!
printf '%s\n' "${child_pid}" > "${PID_FILE}"
wait "${child_pid}"
rc=$?
set -e

printf '%s\n' "${rc}" > "${STATUS_FILE}"
exit "${rc}"
