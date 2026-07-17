#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${ROOT}/scripts/runtime/a1_console.sh"

CONFIG="${ROOT}/configs/evaluation/fruit_placement_offline.toml"
RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
PYTHON="${ROOT}/.venv/bin/python"
LINGBOT_RUNTIME="${ROOT}/scripts/apps/lingbot/a1_lingbot_runtime.sh"
PI05_RUNTIME="${ROOT}/scripts/apps/pi05/a1_pi05_runtime.sh"

cleanup() {
  "${LINGBOT_RUNTIME}" server-stop >/dev/null 2>&1 || true
  "${PI05_RUNTIME}" server-stop >/dev/null 2>&1 || true
}
trap cleanup EXIT

a1_step "Starting hardware-free sequential LingBot Teacher Forcing"
"${LINGBOT_RUNTIME}" server
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON}" \
  -m galaxea_a1_runtime.apps.eef_policy_teacher_forcing lingbot \
  --repo-root "${ROOT}" --config "${CONFIG}" --run-id "${RUN_ID}"
"${LINGBOT_RUNTIME}" server-stop

a1_step "Starting hardware-free sequential Pi0.5 Teacher Forcing"
"${PI05_RUNTIME}" server
PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON}" \
  -m galaxea_a1_runtime.apps.eef_policy_teacher_forcing pi05 \
  --repo-root "${ROOT}" --config "${CONFIG}" --run-id "${RUN_ID}"
"${PI05_RUNTIME}" server-stop

PYTHONPATH="${ROOT}:${PYTHONPATH:-}" "${PYTHON}" \
  -m galaxea_a1_runtime.apps.eef_policy_teacher_forcing summarize \
  --repo-root "${ROOT}" --config "${CONFIG}" --run-id "${RUN_ID}"

trap - EXIT
cleanup
a1_success "Sequential Teacher Forcing complete: outputs/offline_evaluation/fruit_placement/${RUN_ID}/TEACHER_FORCING_REPORT.md"
