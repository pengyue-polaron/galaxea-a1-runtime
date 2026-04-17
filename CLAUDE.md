# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A1 robot arm imitation learning pipeline: data collection (drag-teach & SO leader teleop) → LeRobot v2.1 dataset → OpenPI pi0.5 training → live inference.

## Task Runner (just)

All common operations use `just` (see `Justfile`). Key commands:

```bash
# Data collection (teleop)
just collect teleop --task "pick up the block"

# Data conversion → LeRobot v2.1 (7D joint space)
just convert

# Training
cd scripts/train && python compute_norm_stats.py pi05_a1_joint_lora --data.local_data_dir=../../data/a1_lerobot
python scripts/train/train.py pi05_a1_joint_lora --data.local_data_dir=data/a1_lerobot --exp_name=my_run

# Inference
just policy <checkpoint_dir>
just policy-ros-bridge

# Hardware
just teleop / just teleop stop
just launch <roscore|camera-server|joint-tracker|ee-tracker|a1-server|driver>
just gripper <start|open|close|stop>
just test camera
just print joints [count] [unit]
```

## Architecture

```
1. COLLECTION (teleop):  SO leader → jointTracker bridge → A1 + cameras → JPEG/CSV → disk
2. CONVERSION:           convert_episodes_to_lerobot_v21.py → LeRobot v2.1 parquet + images
3. TRAINING:             pi0.5 LoRA fine-tuning (7D joint, DeltaActions, quantile norm)
4. INFERENCE:            Policy server (WebSocket) → policy_ros_bridge → ROS → A1 robot
```

**Communication layers:**
- **ROS Noetic** (via Docker) — arm/gripper control (`/joint_states_host`, `/arm_joint_target_position`)
- **ZMQ** — camera frames (5558), robot state (5557), policy actions (5559)

## Key Source Locations

| Path | Purpose |
|------|---------|
| `a1/` | Core Python package |
| `a1/constants.py` | System constants (FPS, ZMQ ports) |
| `a1/data_collection/` | CameraServer, DataCollector, A1ReplayBridge |
| `a1/training/config.py` | Training configs including `pi05_a1_joint_lora` |
| `a1/training/a1_policy.py` | A1Inputs/A1JointInputs/A1Outputs transforms |
| `scripts/collect_data/` | Shell + Python orchestration scripts |
| `scripts/process_data/` | Data conversion (episodes → LeRobot v2.1) |
| `scripts/train/` | Training entry point (train.py, compute_norm_stats.py) |
| `scripts/inference/` | Policy server + ROS/ZMQ bridges |
| `configs/` | Hydra YAML configs |
| `third_party/A1_SDK/` | A1 robot ROS packages |
| `third_party/lerobot/` | LeRobot SDK (modified for v2.1 compat) |

## Data Layout

```
data/
├── a1/episode_<timestamp>/                  # Raw teleop collection
│   ├── cam0/*.jpg, cam1/*.jpg
│   ├── frames.csv                           # [frame_index, timestamps, joint1..6, gripper]
│   └── metadata.json
└── a1_lerobot/                              # LeRobot v2.1 dataset (after conversion)
    ├── data/chunk-000/episode_000000.parquet # [state(7), action(7), cam_0, cam_1, ...]
    ├── meta/{info.json, tasks.jsonl, episodes.jsonl, stats.json}
    └── images/chunk-000/episode_000000/{cam_0,cam_1}/*.jpg
```

## Training Pipeline Details

- **State/action**: 7D joint space `[arm_joint1..6, gripper]`
- **Action definition**: `action[t] = state[t+1]` (absolute joint targets)
- **DeltaActions**: Automatically applied during training (6 joint dims → delta, gripper → absolute)
- **Normalization**: Quantile normalization (q01/q99), computed by `compute_norm_stats.py`
- **Config name**: `pi05_a1_joint_lora` (Pi0.5 with LoRA, action_dim=7)

## Key Constants (`a1/constants.py`)

- `ROBOT_FPS = 50`, `CAM_FPS = 20`
- ZMQ ports: 5556 (cmd), 5557 (state), 5558 (camera), 5559 (policy)
