# Galaxea A1 Runtime Agent Guide

This file contains only constraints for code agents. Operator procedures belong
in [docs/RUNBOOK.md](docs/RUNBOOK.md), live-control rules in
[docs/SAFETY.md](docs/SAFETY.md), and design/configuration contracts in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Read the relevant document before
changing that area; tracked configuration and executable code remain the source
of truth.

## Safety

- This repository controls a real arm. Treat every ROS publisher and hardware
  handle as live unless the user explicitly confirms otherwise.
- Prefer static checks. When the arm is off, do not run execution doctors. Use
  `--require-execution` only after the user confirms power and a clear workspace.
- Parse and fully validate tracked configuration before ROS initialization,
  camera/serial access, Docker/tmux creation, or publication.
- Normal applications publish only the staged target topics documented in
  `docs/SAFETY.md`; they never publish host motor commands directly.
- Preserve the locked, fail-closed relay. Do not weaken freshness, finite-value,
  motor-status, alignment, limit, or ownership checks to make startup pass.
- Direct host-topic debugging requires an explicit user request and `just stop`
  first. Never leave two drivers, trackers, camera readers, serial owners, or
  command publishers competing for one device.
- After partial startup failure, stop repository-owned resources before retrying.
  Emergency cleanup may target marked repository resources only.

## Boundaries

Dependency direction is:

```text
scripts -> apps -> runtime / hardware / policies -> configuration / schema / safety
```

- `scripts/runtime/` owns app-agnostic ROS, driver, tracker, relay, and shared
  process lifecycle. It must not depend on Teleop, ACT, or LingBot.
- `scripts/apps/` contains thin entrypoints. Stateful behavior lives under
  `galaxea_a1_runtime/apps/`; reusable runtime, hardware, collection, and policy
  logic lives in its focused package.
- Keep Teleop collection, inference, and conversion independent of LingBot.
- Do not patch `third_party/lerobot` for A1 behavior. Put integrations in the
  first-party package.
- Reuse `galaxea_a1_runtime.runtime.ros1_env.configure_ros1_python` before ROS1
  imports; do not duplicate path surgery.

## Configuration and live contracts

- Follow the single-owner configuration graph in `docs/ARCHITECTURE.md`. Never
  duplicate a semantic value across configs, dataclasses, shell exports, or
  fallback defaults.
- Tracked schemas are strict: require every behavior-affecting key, reject
  unknown keys, and ensure every tracked key is consumed.
- App entrypoints may accept a tracked config path and lifecycle/experiment
  identity. Do not add CLI flags or environment overrides for hardware, safety,
  cameras, collection, or deployment behavior.
- Load the owning typed config directly. Shell exports are narrow lifecycle APIs,
  not alternate configuration objects.
- Host command topic literals may occur only in System config, explicit debug
  tools, and documentation.
- Decode named joint vectors using `joint_safety.names`; reject duplicate,
  missing, or non-finite values and reorder explicitly. Positional fallback is
  allowed only for truly unnamed feedback; command messages must carry names.
- Do not add hidden clamps, scaling, thresholds, or policy-output rewrites.
  Required limits belong in tracked config or a named safety module.

## Data and models

- Gripper state/actions are continuous normalized `0..1` above hardware and map
  exactly once to the System-owned physical stroke. Use
  `/gripper_stroke_host` as feedback; never reinterpret joint-state element 7.
- Formal collection accepts only `galaxea_a1_teleop_raw_v3`. Do not add schema
  fallbacks or migration unless the user explicitly requests recovery.
- Collection must record reproducibility metadata and fresh joint, EEF, action,
  gripper, and paired-camera samples. Enforce configured camera skew and sample
  freshness.
- Write raw episodes and converted datasets to hidden sibling staging paths;
  expose them only by atomic rename after validation. Preserve the previous
  complete output on conversion failure. Crash leftovers must block reuse until
  inspected.
- Dataset keys, state/action names, camera order, and protocol channels come from
  one schema module, never repeated literals or positional slicing.
- Keep datasets under `data/`, durable run results under `outputs/`, external
  checkouts under `external/`, and deployment weights under `models/` as defined
  in `docs/ARCHITECTURE.md`. Never commit weights or add Git LFS.
- Do not delete datasets, recordings, checkpoints, or user files without explicit
  authorization.

## Implementation quality

- Keep configuration, validation, mapping, clamps, and safety decisions in pure,
  ROS-free modules. Hardware code adapts those decisions to external APIs.
- A hardware family has one config-driven constructor. Every physical resource
  has one explicit owner and shutdown order.
- Keep optional heavy dependencies lazy so config validation and pure tests do
  not require ROS, cameras, Torch, serial devices, or model checkouts.
- Give one concept one public name. Prefer typed result dataclasses over mixed
  tuples, insertion-order dictionaries, or list-tail conventions.
- Share an abstraction only after two real callers have the same semantic
  contract. When a shared path becomes authoritative, delete its duplicates.
- Before retaining compatibility code or an apparently unused module, find a
  current caller with `rg`; delete dead branches and superseded wrappers.
- Python CLIs use `galaxea_a1_runtime.console`; shell CLIs source
  `scripts/runtime/a1_console.sh`. Preserve the shared INFO/STEP/PASS/WARN/FAIL
  semantics and machine-readable output without ANSI.
- Keep lifecycle CLIs verb-oriented. Shared supervision belongs in
  `a1_tmux.sh` or `a1_services.sh`; help and static diagnostics must not open
  hardware.

## Change hygiene

- Inspect `git status` and `git diff` first. Preserve unrelated dirty changes;
  never reset or rewrite user work.
- Use `rg` for search and `apply_patch` for manual edits.
- Add the smallest regression test at the purest boundary. Tests should assert
  behavior or public contracts, not source layout. CI must not require hardware.
- A tracked contract change includes its loader, validation, consumers,
  metadata, behavioral tests, and affected documentation in the same change.
- Keep commits reviewable; separate behavior, configuration migration,
  mechanical cleanup, and refactors.
- Before handoff run `just check` and `git diff --check`. For hardware-adjacent
  work, state which checks were static and which touched real hardware.
