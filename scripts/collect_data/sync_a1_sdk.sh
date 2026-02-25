#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SRC_ROOT="${1:-/home/eric/A1_SDK}"
DST_ROOT="${PROJECT_ROOT}/third_party/A1_SDK"

if [[ ! -d "${SRC_ROOT}" ]]; then
  echo "Source A1_SDK not found: ${SRC_ROOT}"
  exit 1
fi

mkdir -p "${DST_ROOT}"
rsync -a --delete --exclude '.git' "${SRC_ROOT}/" "${DST_ROOT}/"
echo "Synced A1_SDK:"
echo "  from: ${SRC_ROOT}"
echo "  to  : ${DST_ROOT}"
