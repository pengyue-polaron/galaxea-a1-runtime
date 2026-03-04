# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**DragDataCoach** is an end-to-end robotics imitation learning pipeline for the A1 robot arm. It covers: drag-based demonstration recording → data processing → policy training (LeRobot/OpenPI) → live inference.

## Task Runner (just)

All common operations use `just` (see `Justfile`). Key commands:

```bash
# Environment
just doctor                           # Check runtime dependencies
just which-python                     # Show active Python interpreter

# Data collection workflow
just drag start / stop                # Enable/disable drag mode on robot
just record start <tag> / stop        # Record bag file during drag demo
just replay [bag] [rate]              # Replay a bag file
just collect                          # Collect data (run during replay)
just drag-collect [options]           # All-in-one: record + replay + collect

# Hardware control
just launch <roscore|camera-server|ee-tracker|a1-server|driver>
just gripper <start|open|close|stop>
just camera test
just print joints [count] [unit]
just bag <latest|info> [bag]

# Inference and debugging
just replay-infer <input> [options]   # Replay with policy inference
just policy [policy_dir]             # Start policy server
just debug camera [options]          # Dump model input frames
```

## Architecture

The system operates as a **multi-process pipeline** coordinated by shell scripts and tmux:

```
1. RECORDING (drag mode):  A1 Robot → ROS topics → rosbag file
2. REPLAY + COLLECTION:    Bag → A1ReplayBridge → ROS
                           CameraServer → ZMQ(cam, port 5558)
                           DataCollector ← ZMQ streams → disk
3. PROCESSING:             align_timestamps.py → data_converter.py → LeRobot dataset
4. TRAINING:               train.py (Hydra + JAX/Flax) → policy checkpoint
5. INFERENCE:              ZMQPolicyServer (port 5559) → A1Server → robot
```

**Communication layers:**
- **ROS Noetic** — arm/gripper control at 50 Hz (`/end_effector_pose`, `/gripper_stroke_host`)
- **ZMQ** — camera frames, robot state, commands, policy actions
  - Port 5556: commands, 5557: state, 5558: camera, 5559: policy actions
- **tmux** — multi-pane session management for all-in-one collection

## Key Source Locations

| Path | Purpose |
|------|---------|
| `datacoach/` | Core Python package |
| `datacoach/constants.py` | System constants (FPS, ZMQ ports, A1 scaling) |
| `datacoach/data_collection/` | Runtime collection (CameraServer, DataCollector, A1ReplayBridge) |
| `datacoach/data_processing/` | Dataset alignment and LeRobot conversion |
| `datacoach/training/` | Training configs, data loaders, policy definitions |
| `datacoach/inference/` | ZMQPolicyServer and policy serving |
| `scripts/collect_data/` | Shell + Python orchestration scripts |
| `scripts/train/train.py` | Main training entry point (Hydra) |
| `scripts/inference/my_serve_policy.py` | Policy server (loads OpenPI checkpoint) |
| `scripts/process_data/` | Data conversion scripts |
| `configs/` | Hydra YAML configs (`collect_data.yaml`, `drag_replay.yaml`, `train.yaml`, `process_data.yaml`) |
| `third_party/openpi/` | OpenPI submodule (foundation model policies) |
| `third_party/A1_SDK/` | A1 robot ROS packages |

## Data Layout

```
data/
├── raw_data/<task_name>/demo_<idx>_<timestamp>/
│   ├── cam_0_rgb_video.mp4          # RealSense camera (640x480)
│   ├── cam_1_rgb_video.mp4          # USB camera (640x480)
│   ├── *.metadata                   # Timestamp files
│   ├── states.pkl                   # Robot state history
│   ├── commanded_states.pkl
│   └── trajectory.csv
└── processed_data/<task_name>/      # LeRobot-format dataset
```

## Environment Setup

- **Package manager**: `uv` (path hardcoded in Justfile as `/home/jolia/.local/bin/uv`)
- **Conda env**: `.conda/envs/dragdatacoach/bin/python` (preferred) or `~/miniconda3/envs/datacoach/bin/python`
- **Python version**: 3.11+
- **ROS**: Noetic (`/opt/ros/noetic/setup.bash` sourced at runtime)
- See `docs/SETUP_ENV.md` for full setup, `docs/SETUP_UDEV.md` for serial device rules

## Key Constants (`datacoach/constants.py`)

- `ROBOT_FPS = 50`, `CAM_FPS = 20`
- ZMQ ports: 5556 (cmd), 5557 (state), 5558 (camera), 5559 (policy)
- A1 coordinate scaling/offset parameters

## Config System

Uses **Hydra** with YAML configs in `configs/`. Training configs support Hydra overrides:
```bash
python scripts/train/train.py dataset.path=data/processed_data/my_task model.lr=1e-4
```

## Tech Stack

- **Training framework**: JAX + Flax (functional, compiled)
- **Policies**: LeRobot (`lerobot>=0.4.0`) + OpenPI (foundation models, in `third_party/`)
- **Data I/O**: Pickle, MP4, LeRobot tensor format
- **Vision**: OpenCV, `pyrealsense2` (RealSense), PyTorch/torchvision
