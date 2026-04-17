#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OFFICIAL_SDK_ROOT="${1:-${PROJECT_ROOT}/third_party/A1_SDK}"
OVERLAY_SDK_ROOT="${2:-${PROJECT_ROOT}/third_party/A1_SDK}"
RUNTIME_SDK_ROOT="${A1_SDK_RUNTIME_ROOT:-${PROJECT_ROOT}/third_party/A1_SDK_runtime}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] ${label} not found: ${path}"
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] ${label} not found: ${path}"
    exit 1
  fi
}

require_dir "${OFFICIAL_SDK_ROOT}" "Official A1 SDK root"
require_dir "${OFFICIAL_SDK_ROOT}/install/lib" "Official A1 SDK runtime libraries"
require_dir "${OVERLAY_SDK_ROOT}/tools" "DragDataCoach A1 tools overlay"
require_dir "${OVERLAY_SDK_ROOT}/install/share/mobiman/auto_generated" "DragDataCoach mobiman auto_generated overlay"

mkdir -p "${RUNTIME_SDK_ROOT}"

echo "[INFO] Syncing official arm SDK into runtime root:"
echo "       ${OFFICIAL_SDK_ROOT} -> ${RUNTIME_SDK_ROOT}"
rsync -a --delete --exclude '.git' --exclude '__pycache__' "${OFFICIAL_SDK_ROOT}/" "${RUNTIME_SDK_ROOT}/"

echo "[INFO] Overlaying DragDataCoach tools and launch files..."
mkdir -p "${RUNTIME_SDK_ROOT}/tools"
mkdir -p "${RUNTIME_SDK_ROOT}/data/records"
mkdir -p "${RUNTIME_SDK_ROOT}/install/share/mobiman/auto_generated"

rsync -a "${OVERLAY_SDK_ROOT}/tools/" "${RUNTIME_SDK_ROOT}/tools/"
rsync -a "${OVERLAY_SDK_ROOT}/install/share/mobiman/auto_generated/" \
  "${RUNTIME_SDK_ROOT}/install/share/mobiman/auto_generated/"

overlay_files=(
  "install/share/signal_arm/launch/single_arm_node.launch"
  "install/share/mobiman/launch/simpleExample/jointTrackerdemo.launch"
  "install/share/mobiman/launch/simpleExample/eeTrackerdemo.launch"
  "install/share/mobiman/launch/simpleExample/eeTrajTrackerdemo.launch"
  "install/share/mobiman/launch/simpleExample/ee_record_only.launch"
)

for rel_path in "${overlay_files[@]}"; do
  src="${OVERLAY_SDK_ROOT}/${rel_path}"
  dst="${RUNTIME_SDK_ROOT}/${rel_path}"
  require_file "${src}" "Overlay file"
  mkdir -p "$(dirname "${dst}")"
  install -m 0644 "${src}" "${dst}"
done

echo "[INFO] Runtime SDK ready:"
echo "       ${RUNTIME_SDK_ROOT}"
