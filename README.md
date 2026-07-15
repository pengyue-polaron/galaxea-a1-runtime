# Galaxea A1 Runtime

Fail-closed runtime, SO leader teleoperation, data collection, dataset
conversion, and policy deployment for a real Galaxea A1 arm. The current
baseline is Python 3.12, LeRobot v0.6, and LeRobotDataset v3.

The arm may be powered and reachable while this repository is open. Treat every
ROS publish path as live hardware and read [Safety](docs/SAFETY.md) before
running motion commands.

## Start here

Create the environment and run static validation:

```bash
just setup
just check
```

Then follow the [Runbook](docs/RUNBOOK.md) for hardware acceptance, reset,
Teleop collection, conversion, recovery, and deployment. It labels every
command that can move the robot.

## Scope

This repository provides:

- isolated ROS driver/tracker runtimes behind a validating command relay;
- six-axis SO leader Teleop with continuous gripper control;
- atomic raw episode collection with camera and sample-freshness checks;
- deterministic LeRobot and policy-specific dataset packaging;
- fail-closed ACT joint and LingBot EEF deployment entrypoints;
- strict tracked configuration for hardware, apps, poses, datasets, and models.

First-party implementation lives under `galaxea_a1_runtime/`; `scripts/`
contains thin lifecycle entrypoints, `configs/` contains tracked contracts, and
`third_party/` contains pinned vendor snapshots. Runtime data, results, external
checkouts, and weights live in the ignored `data/`, `outputs/`, `external/`, and
`models/` roots respectively.

## Documentation

Each document has one responsibility:

| Document | Owns |
| --- | --- |
| [Runbook](docs/RUNBOOK.md) | operator commands, expected results, and recovery |
| [Safety](docs/SAFETY.md) | live control paths, relay invariants, status handling, and direct debug |
| [Architecture](docs/ARCHITECTURE.md) | layers, configuration ownership, runtime/data contracts, and artifact layout |
| [Environment setup](docs/SETUP_ENV.md) | Python environment and dependency baseline |
| [udev setup](docs/SETUP_UDEV.md) | persistent A1 serial permissions and device alias |
| [Model registry](models/README.md) | registering local deployment weights |
| [Agent guide](AGENTS.md) | constraints for code-writing agents |

Mechanical assets and vendor-specific notes remain with their directories:
[SO-100 leader CAD](assets/cad/so100_leader/README.md) and
[third-party policy](third_party/README.md).
