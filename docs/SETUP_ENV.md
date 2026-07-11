# Environment Setup

The refactored Galaxea A1 Runtime uses one Python 3.12 environment managed by
`uv`.

## Main Environment

```bash
just setup
```

This installs the locked project environment from `pyproject.toml` and
`uv.lock`.

Check it with:

```bash
just check
```

## Dependency Baseline

- Python: `>=3.12,<3.13`
- LeRobot: official v0.6.0 tag
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`
- Dataset target: LeRobotDataset v3.0

The old OpenPI/TFP, ZMQ, DataCoach, and LeRobot v2.1 paths have been removed
from the main runtime.

## Hardware Setup

Install udev rules:

```bash
just udev
```

Static runtime checks:

```bash
just check
```

Camera and EEF hardware acceptance after power-on:

```bash
just cameras
just eef-test
```
