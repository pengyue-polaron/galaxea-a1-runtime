# Galaxea A1 Runtime Agent Guide

Use this file as the short decision index. Operator commands live in
[`docs/RUNBOOK.md`](docs/RUNBOOK.md), live-control invariants in
[`docs/SAFETY.md`](docs/SAFETY.md), and ownership/data contracts in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Read only the document relevant
to the change; tracked configuration and executable code are authoritative.

## Working loop

1. Inspect `git status` and `git diff`; preserve unrelated and submodule work.
2. Locate the current caller with `rg` and change the single owning layer.
3. Validate configuration before starting ROS, cameras, serial devices, Docker,
   tmux, or publishers.
4. Run the smallest hardware-free test that proves the change, then
   `git diff --check`. Run full `just check` for executable/runtime contract
   changes or when explicitly requested; documentation and prompt-only changes
   do not require the full suite.
5. State which checks were static and whether any hardware was touched.

## Operator and repository workflows

- The A1 Web panel is a runtime surface only: it may select and run tracked
  workflows, show cameras/status, and forward guarded input. Do not add config
  editors, Prompt registration, or other repository writes to it.
- Normal collection is `just collect EXPERIMENT "EXACT PROMPT"`; keep that
  terminal open and use the Foxglove **Galaxea A1 Operations** layout for
  Start/Save/Reset/Discard/Stop. `just cameras start` runs persistent cameras
  and Foxglove without powering or probing the arm.
- Repository content is agent-maintained through the CLI. Register a prompt
  without hand-writing JSON:

  ```bash
  just prompts
  just prompt-register \
    configs/tasks/<catalog>/catalog.json <task-id> "<exact prompt>" \
    train|ood
  ```

  Use a stable lowercase id, preserve the exact single-line prompt, and mark
  `train` only when it belongs to the checkpoint/training set. The command is
  create-only and validates the complete catalog; review and commit its one new
  JSON file.
- Create new tracked configurations with `config template`, `config validate`,
  and `config create`. Existing configuration changes must update their owning
  loader, consumers, tests, and documentation together.

## Non-negotiable live boundaries

- Treat ROS publishers and hardware handles as live. Do not run motion or
  execution doctors unless the user confirms power and a clear workspace.
- Applications publish only configured staged targets. Never bypass or weaken
  the fail-closed relay, its freshness/finite/status/alignment/limit gates, or
  its exclusive command ownership.
- One process owns each driver, tracker, camera, serial bus, and command
  publisher. After partial startup failure, run `just stop` before retrying.
- Direct host-topic debugging requires an explicit request and `just stop`.
  Never delete datasets, recordings, checkpoints, weights, or user files
  without explicit authorization.

## Ownership and contracts

```text
scripts -> apps -> runtime / hardware / policies -> configuration / schema / safety
```

- `scripts/runtime/` is app-agnostic lifecycle; `scripts/apps/` contains thin
  entrypoints; stateful behavior lives under `galaxea_a1_runtime/apps/`.
- Generic collection, evaluation, artifact, and Operator Panel behavior belongs
  in pinned `external/embodied-ops`; this repository owns A1 adapters, values,
  validation, ROS, hardware, and safety. Do not patch `third_party/lerobot` for
  A1 behavior.
- One semantic value has one tracked config owner. Schemas require all
  behavior-affecting keys, reject unknown keys, and must not be shadowed by CLI
  flags, environment overrides, launch defaults, or hidden clamps.
- Use `configure_ros1_python` before ROS1 imports. Keep shared runtime modules
  parseable on Python 3.11 and keep optional heavy dependencies lazy.
- Named joint vectors must be finite, complete, duplicate-free, and explicitly
  reordered. Gripper state/action is normalized `0..1` above hardware and maps
  exactly once to System-owned physical stroke.
- Formal collection writes the canonical LeRobot v3 contract directly and
  commits episodes atomically. Keep datasets in `data/`, results in `outputs/`,
  external code in `external/`, and weights in `models/`; never add Git LFS.
- Keep safety/config/mapping decisions pure and ROS-free. Test each public
  contract once at its purest boundary; add higher-level wiring tests only when
  they prove distinct behavior.
