#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Resolve the checked-in lock without machine-wide uv index/backend selectors.
exec env -u UV_DEFAULT_INDEX -u UV_TORCH_BACKEND \
  "${ROOT}/scripts/apps/lingbot/a1_lingbot_runtime.sh" \
  --config "${ROOT}/configs/deployments/diffusion2one/fruit_blocks_eef.toml" "$@"
