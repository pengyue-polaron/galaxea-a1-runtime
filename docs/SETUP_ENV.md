# DragDataCoach Environment Setup

This project now supports two setup modes:

1. **Use the local repo conda env** (recommended)
2. **Reuse an existing shared conda env** (fallback)

## 1) Local repo env (recommended)

A local env has been created at:

- `.conda/envs/dragdatacoach`

Core runtime dependencies are installed:
- `hydra-core`
- `pyzmq`
- `opencv-python`
- `numpy`
- `scipy`

Verify:

```bash
scripts/collect_data/dragdatacoach.sh which-python
scripts/collect_data/dragdatacoach.sh doctor
```

Expected output includes local path:
- `DATACOACH_PYTHON=.../.conda/envs/dragdatacoach/bin/python`
- `Missing required: <none>`

## 2) Shared env fallback

If local env is unavailable, the script can fall back to shared envs, e.g.:
- `/home/jolia/miniconda3/envs/datacoach`

## Recreate local env (if needed)

If you need to rebuild local env:

```bash
mkdir -p .conda/envs
CONDA_NO_PLUGINS=true conda create -y --solver classic -p .conda/envs/dragdatacoach python=3.10 pip
CONDA_NO_PLUGINS=true conda run -p .conda/envs/dragdatacoach python -m pip install hydra-core pyzmq opencv-python numpy scipy
```

Optional packages:

```bash
# needed only for conversion to LeRobot dataset format
CONDA_NO_PLUGINS=true conda run -p .conda/envs/dragdatacoach python -m pip install lerobot

# needed only if you use realsense in this python env
CONDA_NO_PLUGINS=true conda run -p .conda/envs/dragdatacoach python -m pip install pyrealsense2
```

Then force this interpreter:

```bash
export DATACOACH_PYTHON="$PWD/.conda/envs/dragdatacoach/bin/python"
scripts/collect_data/dragdatacoach.sh doctor
```

## Runtime prerequisites (all modes)

Before running collection scripts, source ROS and A1 SDK in that terminal:

```bash
source /opt/ros/noetic/setup.bash
source third_party/A1_SDK/install/setup.bash
```

## Why `lerobot` is optional

- **Collection/replay pipeline** (`run_drag_replay_collection.py`) does **not** require `lerobot`.
- **Data conversion** (`scripts/process_data/convert_data_to_lerobot.py`) requires `lerobot`.

If you only collect raw/processed demos, you can skip `lerobot`.
