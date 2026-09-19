# Galaxea A1 Runtime Agent Guide

Use [RUNBOOK](docs/RUNBOOK.md) for commands, [SAFETY](docs/SAFETY.md) for live
control, and [ARCHITECTURE](docs/ARCHITECTURE.md) for ownership and data contracts.
Read the relevant document; tracked configuration and executable code are authoritative.

## Working loop

1. Inspect `git status` and `git diff`; preserve user edits and submodule work.
2. Find the caller with `rg` and edit the owning layer.
3. Validate configuration before opening hardware or starting services.
4. Run static configuration validation and `git diff --check`; use `just check`
   for code/build changes. This repository does not maintain test files. Do not
   add tests or duplicate editable configuration values in assertions.
5. Report the result, checks, and whether hardware was touched.

## Running workflows

- A user's explicit request to run, restart, collect, or infer authorizes the
  standard workflow, including its documented startup/reset sequence. Proceed
  after preflight; do not ask again for power or workspace confirmation.
- Use `just collect EXPERIMENT "EXACT PROMPT"` for collection and keep its terminal
  open; it starts the backend and the guarded Operator Session/Foxglove controls
  own the recording actions. Use `just collect-cli` when the terminal should
  accept the episode keys instead.
  `just cameras start` starts cameras and observation services only.
- One process owns each driver, tracker, camera, serial bus, and command publisher.
  After a partial startup failure, run `just stop` before retrying.
- Publish configured staged targets through the fail-closed relay. Preserve its
  freshness, finite-value, status, alignment, limit, and ownership gates.
  Direct host-topic debugging requires an explicit request and `just stop`.
- Never delete datasets, recordings, checkpoints, weights, or user files without
  explicit authorization.

## Repository maintenance

- The Web panel runs tracked workflows and guarded inputs; repository/configuration
  editing belongs in the CLI. Register prompts with `just prompt-register` or
  `just prompt-catalog-create` as documented in RUNBOOK, preserving exact text and
  marking `train` only for checkpoint training prompts.
- Create configs with `config template`, `config validate`, and `config create`.
  Keep one owner per value, require operator-owned behavior keys, and reject
  unknown keys. Fixed implementation details belong to named code constants;
  disabled features carry no inactive tuning values. Update affected loaders, consumers, and docs with contract changes;
  do not shadow configuration with CLI/environment defaults or hidden clamps.
- Ownership flows `scripts -> apps -> runtime/hardware/policies -> config/schema/safety`.
  Entrypoints are thin; state lives in `galaxea_a1_runtime/apps/`. Shared operations
  belong in pinned `external/embodied-ops`; A1 adapters and hardware stay here.
  Do not patch `third_party/lerobot` for A1 behavior.
- Use `configure_ros1_python` before ROS1 imports. Shared runtime modules remain
  Python 3.11-parseable; heavy optional imports stay lazy.
- Joint vectors must be named, finite, complete, duplicate-free, and reordered.
  Gripper values are normalized `0..1` and map once to System physical stroke.
  Keep safety/config/mapping decisions pure and ROS-free.
- Collection commits canonical LeRobot v3 episodes atomically. Keep datasets in
  `data/`, results in `outputs/`, external code in `external/`, and weights in
  `models/`; never add Git LFS.
