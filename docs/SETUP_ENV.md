# DragDataCoach Environment Setup

## Recommended layout

This repository now uses three host-side Python environments plus a Dockerized ROS1 runtime:

| Component | Python | Purpose |
|-----------|--------|---------|
| `.venv` | 3.12 | DataCoach host runtime (`hydra`, `zmq`, `cv2`, etc.) |
| `.venv-camera` | 3.10 | RealSense/OpenCV camera services on arm64 |
| `third_party/lerobot/.venv` | 3.12 | SO leader teleoperation (`lerobot[feetech]`) |
| `a1-noetic` Docker | Ubuntu 20.04 + ROS Noetic | A1 ROS1 driver + trackers |

On Ubuntu 22.04 / Jetson, this split is intentional:
- ROS1 Noetic stays isolated in Docker.
- `pyrealsense2` on arm64 works reliably with Python 3.10, so camera services use a dedicated `.venv-camera`.
- `lerobot` requires Python 3.12, so teleop uses its own venv plus a small ROS Python overlay.

The ROS backend is dual-compatible:
- `A1_ROS_BACKEND=auto` (default): use host ROS Noetic when available, otherwise Docker.
- `A1_ROS_BACKEND=host`: force host ROS Noetic.
- `A1_ROS_BACKEND=docker`: force Docker Noetic.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| `uv` | Python/version manager and installer |
| `just` | Task runner used by this repo |
| Docker | Required for ROS1 / A1 runtime |
| A1 SDK runtime | Prepared under `third_party/A1_SDK_runtime/` |

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify: `uv --version`

## 2. Clone the repo

```bash
git clone <repo-url>
cd DragDataCoach
```

## 3. Build host environments

```bash
just setup-main
just setup-camera
just setup-teleop
```

These commands create:
- `.venv` for the host runtime
- `.venv-camera` for camera services
- `third_party/lerobot/.venv` for teleop

Verify:

```bash
just doctor
just which-python
just which-camera-python
```

## 4. ROS1 / A1 runtime

Use the Dockerized Noetic runtime for ROS1 nodes:

```bash
just launch driver /dev/a1
just launch tracker
just launch ee-record /dev/a1
```

You do not need a host `/opt/ros/noetic` for those commands.

On Ubuntu 20.04 machines that already have a working host ROS Noetic + `third_party/A1_SDK/install`, the same commands will use the host runtime by default. To force Docker on those machines:

```bash
A1_ROS_BACKEND=docker just launch driver /dev/a1
```

## 5. Notes on `pyrealsense2` for arm64

The arm64-compatible `pyrealsense2` wheels usable on Ubuntu 22.04 are currently tied to CPython 3.10. For that reason:
- camera services use `.venv-camera`
- teleop stays on Python 3.12
- the host runtime `.venv` does not need `pyrealsense2`

## 6. External policy dependencies

Inference and training recipes still expect an `openpi` checkout and policy checkpoints. Update `OPENPI_ROOT` and checkpoint paths at runtime if your setup differs:

```bash
OPENPI_ROOT=/path/to/openpi just policy /path/to/checkpoint
```
