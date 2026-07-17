# Environment setup

This document owns the Python environment and dependency baseline. Hardware
setup and operation are covered by the [Runbook](RUNBOOK.md).

## Install

The repository uses one `uv`-managed Python 3.12 environment:

```bash
just setup
```

This installs the locked project from `pyproject.toml` and `uv.lock`. Use `just`
recipes or `${PWD}/.venv/bin/python` for first-party tools; do not invoke app
entrypoints with the older system Python.

Verify the environment without opening hardware:

```bash
just check
```

## Host and container boundary

The host does not need Ubuntu 20.04 or a native ROS Noetic installation. Build
the execution image once:

```bash
docker compose -f docker-compose.a1-noetic.yml build a1-noetic
```

The image is based on `ros:noetic-ros-base-focal`; it owns ROS Noetic and the
A1 SDK execution environment. Runtime orchestration starts isolated containers
for roscore, the A1 driver, the selected tracker, and the validating relay.

Python 3.12 applications, RealSense cameras, serial-device discovery, and
optional GPU drivers remain host-side responsibilities. Ubuntu 22.04 is the
currently verified host. Ubuntu 24.04 can also be used with a working Docker
Engine and compatible device/GPU drivers; it does not need native ROS packages.

## Baseline

- Python `>=3.12,<3.13`
- LeRobot v0.6.0 at `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- LeRobotDataset v3.0 writer and reader
- first-party LeRobotDataset v2.1 exporter, validated through LeRobot's
  official v2.1-to-v3.0 migrator

The runtime no longer uses the old OpenPI/TFP, ZMQ, or DataCoach environments.
Collection writes the current raw-v3 contract; conversion emits generic Joint
and EEF datasets in both v3.0 and v2.1.

## Model inference environments

LingBot and OpenPI pi0.5 are isolated from the first-party runtime and from each
other because their CUDA/model stacks have different Python constraints. A
tracked backend pins source and dependency-lock content; a separate model
descriptor pins the exact weight revision. Setup creates the environment under
the ignored external checkout:

```bash
just lingbot-setup
just pi05-setup
```

The LingBot backend uses its locked Python 3.12 environment. The pinned OpenPI
backend uses the Python 3.11 environment resolved by its committed `uv.lock`.
Neither setup modifies the repository environment, initializes ROS, opens
cameras, or enables arm execution. See the [model
registry](../models/README.md) for artifact verification and the
[Runbook](RUNBOOK.md) for deployment commands.
