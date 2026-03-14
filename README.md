# DragDataCoach

End-to-end pipeline for the A1 robot arm:
1. Drag the arm to record a rosbag demonstration.
2. Replay the bag to reconstruct the trajectory.
3. During replay, collect dual-camera video and arm state.

Each demo produces two video streams and a trajectory state file.

## 1. Install `just`

```bash
sudo snap install just --classic
just --version
```

## 2. One-time setup

Install the Python environment (see `docs/SETUP_ENV.md`), then verify:

```bash
just doctor
just which-python
```

## 3. Command reference

```bash
just drag start
just drag stop

just launch driver /dev/a1
just launch ee-record /dev/a1
just launch tracker
just ee-tracker
just ee-tracker -drag

just gripper start
just gripper open
just gripper close
just gripper stop

just record start drag_demo
just record stop

just replay
just replay /path/to/demo.bag 1.0 position

just collect
just drag-collect --serial /dev/a1 --tag drag_demo

just test camera
just test camera-raw --config configs/drag_replay.yaml

just bag latest
just bag info /path/to/demo.bag
```

See all commands:

```bash
just --list
```

## 4. Camera check

```bash
just test camera
```

Enumerates devices and checks connectivity. Does not open cameras or save frames.

`just replay` without a bag argument automatically uses the latest bag in `third_party/A1_SDK/data/records/`.

`just replay` checks `cam_0` and `cam_1` before starting playback and exits if either is unavailable. `just drag-collect` runs the same check once before launching `collect`.

## 5. Standard manual workflow

**Recording (drag) phase:**

```bash
just launch ee-record /dev/a1
just drag start
just gripper start               # optional
just record start drag_demo
# after dragging:
just record stop
just drag stop
just gripper stop
```

To quickly set a target pose and send it to `ee-tracker`:

```bash
just ee-tracker -drag
```

This will:
- Launch `eeTrackerdemo` (`rviz:=false`)
- Wait for you to manually move the arm to the target pose and press Enter
- Read the current `/end_effector_pose` and publish it to `/a1_ee_target`
- Keep the tracker alive (auto-cleanup on failure); use `--no-keep-tracker` to force exit

**Replay + collection phase (3 terminals):**

```bash
just launch driver /dev/a1
just launch tracker
just collect
just replay /path/to/demo.bag 1.0 position
```

## 6. Audio-triggered gripper (local mic → remote host)

If your microphone is on a local machine and the control code runs on a remote host over SSH, run the volume-threshold listener locally and trigger `just gripper open/close` remotely via SSH.

The remote host supports one-shot gripper commands:

```bash
just gripper open
just gripper close
```

Local listener script:

```bash
scripts/collect_data/gripper_audio_threshold.py
```

Install audio dependencies on your local machine:

```bash
python3 -m pip install numpy sounddevice
```

List local microphone devices:

```bash
python3 scripts/collect_data/gripper_audio_threshold.py --ssh-host <your-ssh-host> --list-devices
```

Toggle open/close on volume threshold:

```bash
python3 scripts/collect_data/gripper_audio_threshold.py \
  --ssh-host <your-ssh-host> \
  --threshold-db -24 \
  --trigger-mode toggle
```

Open only when volume exceeds threshold:

```bash
python3 scripts/collect_data/gripper_audio_threshold.py \
  --ssh-host <your-ssh-host> \
  --threshold-db -24 \
  --trigger-mode open
```

Key parameters:

- `--threshold-db`: trigger threshold in dBFS; closer to `0` = harder to trigger.
- `--reset-db`: re-arm threshold, defaults to 8 dB below `--threshold-db`.
- `--cooldown-seconds`: cooldown after trigger to prevent rapid re-firing.
- `--initial-state close`: in `toggle` mode, the first trigger executes `open`.
- `--device <name-or-index>`: select local microphone.

## 7. All-in-one script

Run the full record → replay → collect pipeline in a single command:

```bash
just drag-collect --serial /dev/a1 --tag drag_demo
```

This calls `scripts/collect_data/dragdatacoach_all_in_one.sh`, which creates a `tmux` session with multiple windows. During recording, `just gripper start` opens in the current terminal for keyboard gripper control; press Enter to stop recording.

If a session with the same name already exists, the script restarts it by default. Use `--on-existing` to change this behavior.

Replay-only (skip recording):

```bash
just drag-collect \
  --skip-record \
  --bag /path/to/demo.bag \
  --serial /dev/a1
```

Optional flags:

```
--rate <float>              replay speed multiplier
--gripper-mode <mode>       gripper mode during replay
--session <name>            tmux session name
--no-gripper-keyboard       skip keyboard gripper control during recording
--no-auto-stop              do not auto-stop the replay launch window when done
--on-existing <policy>      ask|restart|attach|new|abort
```

## 8. Output paths

Raw data is saved to:

```
data/raw_data/<task_name>/demo_<index>_<YYYYMMDD_HHMMSS>/
```

To rename timestamped folders to sequential indices:

```bash
cd data/raw_data/<task_name>
i=0
for dir in */; do
    mv "$dir" "demo_$i"
    ((i++))
done
```

Typical files per demo:
- `cam_0_rgb_video.mp4`
- `cam_1_rgb_video.mp4`
- `states.pkl`
- `commanded_states.pkl`
- `trajectory.csv`

Process into LeRobot format:

```bash
cd scripts/process_data
uv run python align_timestamps.py
uv run python process_data.py
```
