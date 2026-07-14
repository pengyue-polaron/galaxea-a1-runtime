# Runbook

This file is the short operator path. Edit tracked config files when hardware
settings change; avoid ad hoc per-run flags for data collection.

## Daily Check

No hardware motion:

```bash
just check
just hardware
```

## Hardware Acceptance

Use this after reconnecting all hardware and clearing the workspace.

1. Stop stale processes and inspect hardware enumeration:

```bash
just stop
just hardware
```

`just hardware` checks the tracked A1 serial, SO leader by-id serial, RealSense,
wrist camera, and config consistency without moving the robot. If it reports a
serial as busy, stop the runtime before running `just reset` or `just teleop`.

2. Inspect cameras:

```bash
just cameras
```

Check the printed `cam0_front`, `cam1_wrist`, `contact_sheet`, and, when
enabled, `cam0_depth`/`cam0_depth_preview` image paths.

3. Test EEF control:

```bash
just eef-test
```

The tool holds the current EEF pose, then waits for Enter before each
`x+/x-/y+/y-/z+/z-` step. Confirm the EEF moves in the printed direction.

4. Test SO leader teleop:

```bash
just teleop-test
just logs
```

Confirm the bridge log prints `leader_keys=['joint0.pos', ..., 'joint5.pos']`.
Move one leader joint at a time and verify the A1 direction matches the old
working behavior. Test gripper open/close.

5. Return to the tracked initial pose:

```bash
just reset
```

The target pose is tracked in `configs/poses/a1_initial.toml`.
It includes both the A1 joint pose and the SO leader/LeRobot start pose.
Both grippers are closed as part of this command.

6. Stop:

```bash
just stop
```

## Daily Teleop Recording Flow

Use this path for normal dataset collection after the hardware is connected,
powered, and the workspace is clear.

1. Clear stale runtime state and run static checks:

```bash
just stop
just check
just hardware
```

2. Confirm cameras from the tracked teleop config:

```bash
just cameras
```

Check `cam0_front`, `cam1_wrist`, and `contact_sheet`. If depth is enabled in
the tracked config, also check `cam0_depth`/`cam0_depth_preview`. The command
prints `cam0_usb`, `cam0_front_fps`, and `cam1_wrist_fps`. The default config
is USB2-compatible RGB-only.

3. Restore the collection start pose:

```bash
just reset
```

This reads `configs/poses/a1_initial.toml`, moves the A1 arm to the tracked
joint pose, closes the A1 gripper, moves the SO leader to its tracked pose,
commands the SO leader gripper closed, disables leader torque, and stops the
runtime. The A1 and leader move concurrently. If the hardware start pose or
reset speed changes intentionally, update and commit that config file.
During Reset, the terminal shows one compact `A1 xx% | Leader xx%` progress
line. Interactive collection uses color for setup, recording, saved, rejected,
and reset states; set `NO_COLOR=1` when plain output is required.

4. Start recording:

```bash
just teleop pick_cube
```

Replace `pick_cube` with the experiment name. The command reads
`configs/teleop/a1_so100.toml`; it starts ROS, the A1 driver, the staged joint
tracker, the fail-closed relay, the SO leader bridge, and the interactive
recorder. No extra collector flags are accepted on the normal path.

5. Use the episode loop:

```text
first run only: enter task prompt
each episode : Enter=start recording
recording    : Enter=save, d+Enter=discard, q+Enter=quit
```

After Enter=save, the collector validates joint-action continuity before
writing metadata. A step larger than
`collection.max_joint_action_step_rad` rejects and deletes the whole episode,
prints the exact frame/joint/value transition, reuses the episode index, and
automatically homes both devices. Only episodes that pass are reported as
`saved`.

Saved episodes are appended under `data/raw/<experiment>/`. Discarded episodes
are deleted immediately. Exiting the recorder stops the teleop runtime.
After every successful save, the recorder pauses the leader bridge, resets the
A1 and leader concurrently, restarts the bridge, and only then offers the next
episode. A reset failure stops collection with the bridge off. Set
`collection.auto_reset_after_save = false` in the tracked teleop config only
when this behavior is intentionally unwanted.
After saving, compare the printed frame count against the real action length:
at the default 30 FPS, 10 seconds should be about 300 frames. If the printed
nominal duration is much shorter than the action you performed, discard or
delete that episode and investigate camera readiness before continuing.
If either camera stops producing fresh samples while recording, the collector
aborts that episode and deletes the partial folder.

6. Stop manually if needed:

```bash
just stop
```

## What Teleop Records

Teleop behavior is locked by `configs/teleop/a1_so100.toml`: leader port,
cameras, state mode, FPS, topics, joint mapping, limits, and gripper range.
The default front RealSense config records RGB only and accepts USB2.1. Depth
capture remains supported, but enable it intentionally in the tracked config
after the RealSense is on USB3 or after lowering FPS/resolution for USB2. The
wrist camera uses uncompressed YUYV to avoid corrupt MJPG frames.

Recorded state modes:

- `joint`: `joint_1..joint_6`, `gripper`.
- `eef`: `eef_x/y/z/qx/qy/qz/qw`, `gripper`.
- `eef_joint`: both, the tracked default.

Actions are recorded as `joint_absolute` targets from
`/arm_joint_target_position`.
Saved episodes contain `frames.csv`, `metadata.json`, `cam0/`, `cam1/`, and
`cam0_depth/` when depth is enabled. Raw depth is stored as aligned 16-bit PNG
in millimetres and converts to LeRobot as `observation.images.front_depth`.

The raw frame table contains frame index, wall-clock timestamp, ROS timestamp,
relative image paths, configured state columns, and configured action columns.
Episode metadata stores task text, experiment name, state/action names, topics,
camera settings, FPS target, and the staged relay control path.

## Convert Training Data

```bash
just convert banana_in_the_plate
```

Conversion semantics and paths come only from
`configs/datasets/<experiment>.toml`. LingBot packages use binary gripper
actions: `0=closed`, `1=open`. Each conversion emits EEF v3.0, EEF v2.1, and
joint-action v3.0 packages; never mix their files in one directory. Joint
positions remain absolute targets in radians.

## LingBot-VA

The tracked LingBot command assembles the step-500 model root from the frozen
base components and fine-tuned transformer, then starts the policy server, A1
runtime, and bridge:

```bash
just lingbot
tmux attach -t lingbot-a1
```

Runtime behavior is locked by `configs/inference/lingbot_va_a1.toml`: server,
checkpoint, prompt, cameras, EEF workspace, orientation mode, relay topics,
execution cadence, and gripper mapping.

The A1 step-500 profile is a finite continuous rollout: 36 model calls, four
latent frames per call, four actions per frame, at 30 Hz. The first model frame
is the episode condition, so execution is approximately 19 seconds. Model EEF
poses are episode-relative and are composed onto the measured startup pose
before the absolute A1 workspace clamp. The gripper uses the ACT deployment
mapping: continuous policy values map to 0-80 mm. KV-cache action history uses
the target actually sent to the tracker, matching the checkpoint's training
action contract; measured EEF feedback remains the observation and safety
signal, not the model's past-action token.

When the bridge completes, raises an error, or receives `Ctrl-C`, its process
guard stops the A1 runtime and policy server. `just stop` remains the operator
emergency-stop command.

Start only the model server for a no-ROS load test with:

```bash
scripts/apps/lingbot/a1_lingbot_runtime.sh server
scripts/apps/lingbot/a1_lingbot_runtime.sh server-logs
```

Stop with:

```bash
just stop
```

## ACT Joint Policy

The ACT deployment path is configured by
`configs/inference/act_joint_a1.toml`. The tracked checkpoint is
`outputs/train/act_banana_joint_state_30k/checkpoints/030000/pretrained_model`.

Start in the default dry-run mode:

```bash
just stop
just check
just cameras
just act
tmux attach -t act-a1
```

In dry-run, Enter runs one model inference and prints the first predicted joint
targets without enabling the relay. To move the arm, edit the tracked config and
set `execution.execute = true` after the robot is powered, reset, and clear.
Execution still remains step-gated: the bridge aligns jointTracker output to
current feedback, waits for relay `ACTIVE`, then publishes only
`/arm_joint_target_position` plus binary gripper commands.

Stop with:

```bash
just stop
```
