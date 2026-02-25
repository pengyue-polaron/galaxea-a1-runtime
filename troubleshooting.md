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

3. **Verify Zero Position** # !!!! Need to double check: because 无论如何在新的标定下

```bash
roslaunch mobiman eeTrackerdemo.launch # The robot arm should move to correct zero position
rostopic echo /joint_states # After calibration, the joint should read approximately [0, 0, 0, 0, 0, 0] when its at physical zero position
```

Remain to solve:
since like after running `roslaunch mobiman eeTrackerdemo.launch` the robot arm will move to [0, 0, 0, -1.67, 0, 1.67] 

4. **Process dataset**
receive A1 data --> align timestamps --> lerobot dataset --> feed into VLA 


## Set up uv environment --- timeout
Issue: 
jolia@nyushrobo5090:~/DataCoach$ GIT_LFS_SKIP_SMUDGE=1 uv sync
warning: The `tool.uv.dev-dependencies` field (used in `third_party/openpi/packages/openpi-client/pyproject.toml`) is deprecated and will be removed in a future release; use `dependency-groups.dev` instead
Resolved 212 packages in 3ms
  × Failed to download `rerun-sdk==0.23.1`
  ├─▶ Failed to extract archive: rerun_sdk-0.23.1-cp39-abi3-manylinux_2_31_x86_64.whl
  ├─▶ I/O operation failed during extraction
  ╰─▶ Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT (current value: 30s).
  help: `rerun-sdk` (v0.23.1) was included because `datacoach-env` (v0.0.1) depends on `openpi` (v0.1.0) which depends on `lerobot` (v0.1.0) which depends on
        `rerun-sdk`

Since the official uv environment is under DataCoach/third_party/openpi while we want to use this uv to run scripts under DataCoach/scripts/, we need to register a uv env which is under DataCoach and it can import openpi where the package is sourced from third_party/openpi. 

But sometimes if you run  GIT_LFS_SKIP_SMUDGE=1 uv sync, it easily timeout. Then you leave  dependencies blank in `/DataCoach/pyproject.toml`, which is like
```bash
    dependencies = [
   #"openpi @ file:/home/jolia/DataCoach/third_party/openpi"
 ]

```
First register the uv environment under DataCoach with 
```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```
Then register openpi under this uv
```bash
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ./third_party/openpi
```

What I basically do is just split original script (`Datacoach/pyproject.toml`) into two steps: register uv, and register third_party/openpi to uv separately.


## Data version issue
Make sure data collected in Lerobot version 2.1 
https://docs.phospho.ai/learn/lerobot-dataset


## NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation

credit to https://github.com/lllyasviel/Fooocus/issues/3862

To use PyTorch for Linux x86_64 and Linux SBSA on NVIDIA 5080, 5090 Blackwell RTX GPUs use the latest nightly builds, or the command below.
```bash
pip uninstall -y torch torchvision torchaudio
pip cache purge
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```