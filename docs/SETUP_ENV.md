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

## Baseline

- Python `>=3.12,<3.13`
- LeRobot v0.6.0 at `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- LeRobotDataset v3.0

The runtime no longer uses the old OpenPI/TFP, ZMQ, or DataCoach environments.
One offline LeRobot v2.1 export remains solely for LingBot dataset
compatibility.
