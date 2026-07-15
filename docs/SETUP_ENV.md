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
Collection writes the current raw-v3 contract; conversion emits the base A1
dataset in both v3.0 and v2.1 and does the same for the LingBot EEF package.
