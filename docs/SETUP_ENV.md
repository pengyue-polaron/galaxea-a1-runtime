# Environment Setup

The refactored Galaxea A1 Runtime uses one Python 3.12 environment managed by
`uv`.

## Main Environment

```bash
just setup-main
```

This installs the locked project environment from `pyproject.toml` and
`uv.lock`.

Check it with:

```bash
just runtime doctor
just runtime test
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
just udev-install
```

Static runtime checks:

```bash
just runtime doctor
```

Safe hardware runtime checks:

```bash
just a1-runtime doctor
```

Only use execution checks after the arm is powered and placed safely:

```bash
just a1-runtime doctor --require-execution
```
