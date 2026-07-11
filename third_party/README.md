# Third-Party Vendor Policy

This directory is for reproducible vendor snapshots only. It is not where A1
app behavior, safety policy, teleop mapping, or LingBot integration should live.

## Current Vendors

Vendor source metadata is tracked in `vendors.toml`.

- `A1_SDK/`: Galaxea ROS1 A1 SDK snapshot used by Docker/ROS runtime scripts.
- `lerobot/`: LeRobot v0.6.0 source snapshot, pinned in root `pyproject.toml`.

## Rules

- Keep vendor code boring. Prefer first-party adapters under
  `galaxea_a1_runtime/` or `scripts/apps/` over editing vendor files.
- If a vendor patch is unavoidable, make it small, document the reason here,
  and keep a test proving why it exists.
- Do not use nested vendor `.git` directories as the source of truth. The parent
  repository tracks the snapshot that will be reviewed and committed.
- Update vendor snapshots intentionally, in their own commit, and update
  `vendors.toml` in the same change.

## A1-Specific Adapters

- SO leader six-axis A1 wiring lives in
  `galaxea_a1_runtime.teleop.a1_so_leader`.
- LingBot runtime integration lives in `scripts/apps/lingbot/` and
  `galaxea_a1_runtime.apps.lingbot`.
- Safe ROS relay and tracker wrappers live in `scripts/runtime/`.
