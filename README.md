# DragDataCoach

`DragDataCoach` uses an **offline drag-record + replay-collect** workflow:
1. Drag the A1 arm and record a ROS bag.
2. Replay that bag.
3. During replay, DataCoach captures:
   - `cam_0` third-person video (RealSense)
   - `cam_1` hand-eye video (regular video device)
   - arm trajectory/state streams

Each processed demo outputs:
- two aligned videos (`cam_0_rgb_video.mp4`, `cam_1_rgb_video.mp4`)
- aligned trajectory/state files (`states.pkl`, `commanded_states.pkl`, `trajectory.csv`)

## Third-party A1_SDK

Copy external SDK into this repo (as pure third-party, decoupled from DataCoach logic):

```bash
scripts/collect_data/sync_a1_sdk.sh /home/eric/A1_SDK
```

It syncs into:

```bash
third_party/A1_SDK
```

## Environment

Use:

```bash
scripts/collect_data/dragdatacoach.sh which-python
scripts/collect_data/dragdatacoach.sh doctor
```

By default it now selects local env:

```bash
.conda/envs/dragdatacoach/bin/python
```

Detailed setup guide:

- [Environment Setup](docs/SETUP_ENV.md)
- [A1 Serial Setup (udev)](docs/SETUP_UDEV.md)

`lerobot` is optional unless you run `scripts/process_data/convert_data_to_lerobot.py`.

## Quick Start (Your Workflow)

### 0) Terminal prerequisites (must pass before running)

In terminals that run DataCoach Python scripts, ensure:

```bash
# optional: inspect selected interpreter
scripts/collect_data/dragdatacoach.sh which-python

# ROS + A1 message path
source /opt/ros/noetic/setup.bash
source third_party/A1_SDK/install/setup.bash
```

If not sourced, `run_drag_replay_collection.py` will fail to import ROS/A1 message modules.

### 1) Start A1 driver

```bash
scripts/collect_data/dragdatacoach.sh launch-driver
```

Equivalent raw command:

```bash
source third_party/A1_SDK/install/setup.bash
roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=/dev/ttyACM0
```

### 2) Start drag mode

```bash
scripts/collect_data/dragdatacoach.sh drag-start
```

### 3) Start keyboard gripper

```bash
scripts/collect_data/dragdatacoach.sh gripper-keyboard
```

### 4) Start/stop bag recording while dragging

```bash
scripts/collect_data/dragdatacoach.sh record-start drag_demo
# ... finish dragging ...
scripts/collect_data/dragdatacoach.sh record-stop
scripts/collect_data/dragdatacoach.sh drag-stop
```

### 5) Start tracker launch for replay

```bash
scripts/collect_data/dragdatacoach.sh launch-tracker
```

Equivalent raw command:

```bash
source third_party/A1_SDK/install/setup.bash
roslaunch mobiman eeTrackerdemo.launch
```

### 6) Start DataCoach replay collection

Open a new terminal:

```bash
source /opt/ros/noetic/setup.bash
source third_party/A1_SDK/install/setup.bash
scripts/collect_data/dragdatacoach.sh collect
```

Press Enter to start recording in DataCoach.

### 7) Run replay

Open a new terminal:

```bash
scripts/collect_data/dragdatacoach.sh replay --bag /home/eric/A1_SDK/data/records/a1_eef_drag_20260226_043052.bag --gripper-mode position --rate 1.0
```

Use `--gripper-mode position` for this pipeline.

After replay completes, go back to DataCoach terminal and press `Ctrl+C` to save.

Raw data is saved under:

```bash
data/raw_data/<task_name>/demo_<index>/
```

## Post-processing

Set `task_name`/paths in `configs/process_data.yaml`, then:

```bash
PY=$(scripts/collect_data/dragdatacoach.sh which-python)
$PY scripts/process_data/align_timestamps.py
$PY scripts/process_data/convert_data_to_lerobot.py
```

`convert_data_to_lerobot.py` requires `lerobot`; see [Environment Setup](docs/SETUP_ENV.md).

Processed output:

```bash
data/processed_data/<task_name>/demo_<index>/
```

Includes:
- `cam_0_rgb_video.mp4`
- `cam_1_rgb_video.mp4`
- `states.pkl`
- `commanded_states.pkl`
- `trajectory.csv`
