# Galaxea A1 Runtime

LeRobot-native runtime for the Galaxea A1 arm.

This repository is being rebuilt around:

- a fail-closed A1 execution runtime,
- LeRobot v0.6.0,
- LeRobotDataset v3.0,
- SO leader teleoperation collection,
- LingBot-VA, FastWAM, and GR00T N1.7 policy profiles,
- clean module boundaries for safety, hardware IO, datasets, and policies.

The arm may be powered and reachable while this repo is open. Treat every ROS
publish path as live hardware.

## Safe Command Path

Normal EEF apps and inference code must use:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay_v2.py
  -> /arm_joint_command_host
```

The relay starts locked. Direct `/arm_joint_command_host` publishing is only for
explicit hardware debug after stopping the safe runtime.

Teleop collection uses the same relay with staged jointTracker output:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay_v2.py
  -> /arm_joint_command_host
```

## Current Main Commands

Static, hardware-free commands:

```bash
just runtime plan
just runtime doctor
just runtime dry-run --profile static
just runtime dry-run --profile safe
just runtime profiles
just runtime safety
just runtime test
```

Dataset migration planning:

```bash
just dataset migration-plan --kind lerobot-v2.1 --repo-id galaxea/a1_task
just dataset convert-raw --dry-run \
  --source-root data/raw/a1_task \
  --target-root data/processed/a1_task \
  --repo-id galaxea/a1_task
```

Safe hardware runtime:

```bash
just a1-runtime doctor
just a1-runtime services
just a1-runtime status
just a1-runtime stop
```

Teleoperation collection:

```bash
just a1-teleop doctor
just collect teleop pick_cube
just a1-teleop stop
```

Use `--require-execution` only after the arm is powered on and positioned
safely.

## New Package Layout

```text
galaxea_a1_runtime/
  safety.py           # pure fail-closed validation and limiters
  schema.py           # LeRobot v3 state/action/camera contracts
  config.py           # typed runtime and dataset config
  hardware/           # IO protocol, EEF helpers, ROS1 safe adapter
  collection/         # teleop state/action schema and episode helpers
  teleop/             # SO leader to A1 joint mapping helpers
  lerobot/            # Robot adapter, dataset writer, recorder, migration
  policies/           # action normalization and policy profiles
  apps/               # reusable app helpers and app-specific transforms
  runtime/            # static doctor and dry-run supervisor
```

## Policy Targets

```bash
just runtime profiles
```

Profiles currently cover:

- LingBot-VA: `policy.type=lingbot_va`
- FastWAM: `policy.type=fastwam`
- GR00T N1.7: `policy.type=groot`

All policy outputs are normalized into the same A1 runtime action contract
before execution.

## LeRobot Robot Adapter

`GalaxeaA1Robot` exposes a LeRobot-style robot interface. Its generic ROS1
hardware adapter supports EEF translation/delta actions by reading
`/end_effector_pose`, publishing `/a1_ee_target`, and enabling the safe relay.
Joint-space arm execution is not implemented in the generic adapter.

## Teleoperation Collection

`just collect teleop <experiment>` starts the staged joint teleop runtime,
records camera frames plus A1 state/action data, and keeps the old episode loop:
Enter starts recording, Enter saves, `d` discards, and `q` exits.

Runtime parameters live in [configs/teleop/a1_so100.toml](configs/teleop/a1_so100.toml).
Edit that tracked file when the SO leader port, cameras, topics, state mode,
FPS, joint mapping, or gripper stroke range changes. The normal collection
entrypoint does not take per-run collector flags.

The raw teleop schema records configurable state modes:

- `eef`: EEF pose plus gripper
- `joint`: six arm joints plus gripper, the default to match the old working
  teleop collector
- `eef_joint`: both

Teleop actions are recorded as `joint_absolute` targets from
`/arm_joint_target_position`. Each saved episode contains `frames.csv`,
`metadata.json`, `cam0/`, and `cam1/`; the metadata records the state topics,
action topics, and staged relay control path used for that episode.

## Dependency Baseline

The main Python environment is pinned to official LeRobot v0.6.0:

```text
30da8e687a6dfc617fcd94afc367ac7071c376ce
```

Python baseline is `>=3.12,<3.13`.

`third_party/lerobot` is also replaced with the official v0.6.0 checkout at the
same commit.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Runbook](docs/RUNBOOK.md)
- [Safety](docs/SAFETY.md)

## Legacy Systems

The previous ZMQ/OpenPI/LeRobot v2.1 mainline stack has been removed. Old data
migration remains one-way under `just dataset ...`.
