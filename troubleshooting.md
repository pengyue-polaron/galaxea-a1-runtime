# Troubleshooting Log

This document is used to track issues, their causes, and resolutions.  
Each entry is versioned by date for easy reference.

---

## Motor Calibration: Why is the Max Position Extremely Large (Over 30,000)?


**Symptom**
When recording motor positions during calibration, you see a very large max value:
wrist_roll: min=0, max=33057

…but mechanically, the motor only moves a small fraction of a turn.

Cause
The calibration script assumes the motor starts in the middle of its physical range.
Continuous rotation joints (like shoulder_pan or wrist_roll) are assigned a homing offset = 2047.
If you start calibration away from the middle, the offset does not match the mechanical center.
Small movements can then produce very large software positions (30,000+) due to encoder wrap accumulation.

Solution
Always start calibration with the motor in the middle of its range.
The script will assign the homing offset correctly.
Recorded positions will then reflect the true mechanical motion.



## A1 Robot Arm Motor Zero Calibration

This guide explains how to recalibrate the motor zero point for the A1 robot arm to ensure the physical zero position matches the software zero.

## Overview

To fix a motor offset issue, connect to the A1 CAN bus, disable the motor, manually move the arm to the physical zero position, perform the motor zero calibration, and then re-enable the motor.

## Steps

1. **Connect to CAN Communication**

```bash
a1env      # if alias is set
roslaunch signal_arm single_arm_node.launch
```
Verify CAN service is available:
```bash
rosservice list # Look for "/iarm_node_single_arm/function_frame"
```

2. **Motor Zero Calibration Sequence**
This procedure ensures the robot arm's software zero aligns with the physical zero position.
```bash
rosservice call /iarm_node_single_arm/function_frame 2   # Disable motors
# Manually move the robot arm to physical zero position
rosservice call /iarm_node_single_arm/function_frame 3   # Motor zero calibration
rosservice call /iarm_node_single_arm/function_frame 4   # Clear error data (if any)
rosservice call /iarm_node_single_arm/function_frame 1   # Enable motors
```

3. **Verify Zero Position** # TODO: needs re-verification after new calibration

```bash
roslaunch mobiman eeTrackerdemo.launch # The robot arm should move to correct zero position
rostopic echo /joint_states # After calibration, the joint should read approximately [0, 0, 0, 0, 0, 0] when its at physical zero position
```

Remain to solve:
since like after running `roslaunch mobiman eeTrackerdemo.launch` the robot arm will move to [0, 0, 0, -1.67, 0, 1.67] 

4. **Process dataset**
receive A1 data --> align timestamps --> lerobot dataset --> feed into VLA 


## uv sync times out (large packages like rerun-sdk fail to download)

Increase the HTTP timeout:

```bash
UV_HTTP_TIMEOUT=120 uv sync
```

Or skip LFS files:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

If it still fails, install in two steps — sync everything except openpi first, then install openpi separately:

```bash
# 1. Temporarily comment out the openpi dependency in pyproject.toml, then sync the rest
GIT_LFS_SKIP_SMUDGE=1 uv sync
# 2. Install openpi editable
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ./third_party/openpi
```


## Data version issue
Make sure data collected in Lerobot version 2.1 
https://docs.phospho.ai/learn/lerobot-dataset


## NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation

**Resolved**: `torch==2.7.1` from PyPI (`+cu126`) natively supports Blackwell SM_120.
No nightly build required — `uv sync` handles it.

Key points:
- Compatibility depends on the **driver** CUDA version (shown in `nvidia-smi`), not the `nvcc` toolkit version.
- Driver ≥ 12.6 runs `cu126` wheels without issue.
- This machine: driver 570.x, CUDA 12.8 — fully compatible.