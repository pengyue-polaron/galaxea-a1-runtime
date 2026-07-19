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
  process lifecycle. It must not depend on Teleop, LingBot, or OpenPI.
- `scripts/apps/` contains thin entrypoints. Stateful behavior lives under
  `galaxea_a1_runtime/apps/`; reusable runtime, hardware, collection, and policy
  logic lives in its focused package.
- Keep Teleop collection, inference, and conversion independent of LingBot.
- Do not patch `third_party/lerobot` for A1 behavior. Framework-neutral contracts
  and the two LeRobot adapters live in the pinned first-party repositories under
  `external/`; runtime-specific ROS and safety behavior stays in this package.
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
- First-party ROS launch files must require their topic arguments. Runtime
  entrypoints render them from System config; fallback topics would create an
  unsafe second source of truth.
- Decode named joint vectors using `joint_safety.names`; reject duplicate,
  missing, or non-finite values and reorder explicitly. Positional fallback is
  allowed only for truly unnamed feedback; command messages must carry names.
- Do not add hidden clamps, scaling, thresholds, or policy-output rewrites.
  Required limits belong in tracked config or a named safety module.
- Preserve verified Teleop observation, action, mapping, and reset semantics.
  An intentional change must update the owning config, behavioral tests, and
  affected documentation in the same change; compatibility must not drift
  implicitly.

## Data and models

- Gripper state/actions are continuous normalized `0..1` above hardware and map
  exactly once to the System-owned physical stroke. Use
  `/gripper_stroke_host` as feedback; never reinterpret joint-state element 7.
- Formal collection writes the canonical `galaxea_a1_lerobot_dataset_v3_v2`
  contract directly. Raw v3 is a read-only legacy migration source and must
  never be reintroduced as a new-collection intermediate.
- Collection must record reproducibility metadata and fresh joint, EEF, action,
  gripper, and paired-camera samples. Enforce configured camera skew and sample
  freshness.
- Append each direct LeRobot episode through a hidden sibling dataset snapshot;
  expose it only by atomic rename after LeRobot finalization and validation.
  Preserve the previous complete dataset on failure. Crash leftovers must block
  reuse until inspected.
- Dataset keys, state/action names, camera order, and protocol channels come from
  one schema module, never repeated literals or positional slicing.
- The canonical direct dataset stores absolute EEF pose, six measured joints in
  radians, normalized gripper state, absolute joint targets in radians,
  normalized gripper action, task text, and configured camera observations.
- Name intentional Joint/EEF or version derivatives by their stored
  representation and LeRobot version, never by a consuming model. A derivative
  may read the canonical direct dataset or an explicitly legacy Raw v3 import;
  one final derivative must never be the source of another final derivative.
- Legacy Raw v3 migration applies its Dataset-owned boundary trim once before
  Joint/EEF fan-out and records `[start, end)` in `meta/trim.json`. Do not add
  this historical transformation to direct collection implicitly.
- Keep datasets under `data/`, durable run results under `outputs/`, external
  checkouts under `external/`, and deployment weights under `models/` as defined
  in `docs/ARCHITECTURE.md`. Never commit weights or add Git LFS.
- Do not delete datasets, recordings, checkpoints, or user files without explicit
  authorization.

## Implementation quality

- Keep configuration, validation, mapping, clamps, and safety decisions in pure,
  ROS-free modules. Hardware code adapts those decisions to external APIs.
- Give every safety-critical config-to-runtime mapping an exhaustive,
  hardware-free unit test. Doctors and generated reports must derive displayed
  values from loaded config rather than hardcoding a second expected value.
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
- Keep one authoritative test at the purest boundary for each public contract.
  Higher layers get one wiring smoke test instead of repeating lower-layer edge
  cases. Tests assert behavior or public contracts, not source layout.
- Ordinary features default to one happy-path test and one meaningful failure;
  bug fixes add one minimal regression, and behavior-preserving refactors add no
  tests. Extend an existing table or test module before creating a new file.
- Do not retain tests for removed options or legacy behavior unless that
  compatibility is an explicit current requirement. CI must not require hardware.
- A tracked contract change includes its loader, validation, consumers,
  metadata, behavioral tests, and affected documentation in the same change.
- Keep commits reviewable; separate behavior, configuration migration,
  mechanical cleanup, and refactors.
- Before handoff run `just check` and `git diff --check`. For hardware-adjacent
  work, state which checks were static and which touched real hardware.
