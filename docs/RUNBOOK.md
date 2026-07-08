# Runbook

## Daily Static Checks

These do not touch hardware:

```bash
just runtime doctor
just runtime safety
just runtime profiles
just runtime dry-run --profile safe
just runtime dry-run --profile collect
just runtime test
```

## Base A1 Runtime

Use this for the safe ROS/A1 stack:

```bash
just a1-runtime doctor
just a1-runtime services
just a1-runtime doctor --require-execution
just a1-runtime status
just a1-runtime logs
just a1-runtime stop
```

Only use `--require-execution` after the arm is powered on and in a clear
workspace.

## Teleop Collection

Use this for SO leader demonstration collection:

```bash
just a1-teleop doctor
just collect teleop pick_cube
```

First run asks for the task prompt and writes `data/raw/pick_cube/task.txt`.
Each episode uses the same interaction:

```text
Enter=start recording
Enter=save
d+Enter=discard
q+Enter=quit
```

The teleop runtime starts the A1 driver, staged joint tracker, relay, SO leader
bridge, and recorder. It stops those processes when collection exits.

Teleop runtime parameters are locked by the tracked config file:

```text
configs/teleop/a1_so100.toml
```

Edit that file when camera devices, leader port, gripper stroke range, state
mode, FPS, topics, joint mapping, or limits change. The normal collection
command intentionally rejects per-run collector flags so dataset semantics do
not drift between episodes.

State modes:

- `eef`: `eef_x/y/z/qx/qy/qz/qw`, `gripper`
- `joint`: `joint_1..joint_6`, `gripper`, default to match the old working
  teleop collector
- `eef_joint`: both

Actions are recorded as `joint_absolute` targets from
`/arm_joint_target_position`. Joint values are radians; gripper is normalized
`0..1` using the old teleop bridge's default `0..200mm` gripper stroke range.
Each `metadata.json` records the state topics, action topics, and staged relay
path for the saved episode.

Stop a teleop session manually with:

```bash
just a1-teleop stop
```

For an explicitly tracked alternate setup, create another config file and run:

```bash
A1_TELEOP_CONFIG=configs/teleop/my_setup.toml just collect teleop pick_cube
```

## LingBot-VA

Start the LingBot server separately, then:

```bash
just a1-lingbot doctor
just a1-lingbot start
tmux attach -t lingbot-a1
```

The bridge is step-gated:

- `INFERENCE #N READY`: Enter runs one new model inference.
- `Next=publish this EE step`: Enter publishes one already predicted EEF step.
- Cache update messages are synchronization, not new inference.

Stop with:

```bash
just a1-lingbot stop
```

## LingBot Parameters

Normal startup does not apply a per-step XYZ hard clamp. The model's XYZ target
is executed as predicted, subject only to explicit workspace bounds, orientation
mode, gripper range mapping, and relay safety.

Advanced bridge flags can be passed with `A1_LINGBOT_BRIDGE_EXTRA_ARGS`, for
example:

```bash
A1_LINGBOT_BRIDGE_EXTRA_ARGS="--eef-servo-gain 1.4 --eef-servo-settle 0.4 --eef-servo-corrections 1" \
  just a1-lingbot tmux
```

Advanced flags:

- `--eef-servo-gain`
  Extra target gain for the official EE tracker. Default `1.0` means off. Values
  above `1` intentionally overshoot the tracker target to compensate observed
  under-tracking.
- `--eef-servo-max-extra`
  Maximum extra overshoot distance in meters when servo gain is enabled.
- `--eef-servo-settle`
  Seconds to wait after a publish to measure actual EEF tracking error.
- `--eef-servo-tolerance`
  XYZ error tolerance in meters for settle/correction logic.
- `--eef-servo-corrections`
  Number of extra correction publishes after settle if the actual EEF is still
  far from the policy target.
- `--orientation-mode hold-current`
  Default. Ignore model quaternion channels and hold the current orientation.
- `--orientation-mode model-quat`
  Use model quaternion channels. Use only when you trust orientation outputs.
- `--cache-actual-feedback`
  Default. Feed measured `/end_effector_pose` back into LingBot cache state
  instead of pretending the command was reached exactly.
- `--gripper-stroke-scale 60`
  Maps normalized gripper `0..1` to `0..60mm`.

## Real-Hardware Checklist

Run these manually when you are ready to test the physical arm:

1. Static preflight:

```bash
just runtime doctor
just runtime test
```

2. Stop stale runtime:

```bash
just a1-runtime stop
```

3. Powered-off/optional doctor:

```bash
just a1-runtime doctor
```

4. Power the arm, clear the workspace, then start safe runtime:

```bash
just a1-runtime services
just a1-runtime doctor --require-execution
```

5. Generic ROS1 EEF smoke test:
   - Send a tiny EEF translation, 1 cm or less, through `GalaxeaA1Robot`.
   - Confirm app output is `/a1_ee_target` plus `/a1_arm_motion_enable`.
   - Confirm no app publisher appears on `/arm_joint_command_host`.
   - Confirm `/a1_arm_relay_status` becomes `ACTIVE` only after validation.

6. LingBot test:

```bash
just a1-lingbot doctor --require-execution
just a1-lingbot start
tmux attach -t lingbot-a1
```

Read the preview before each publish step. Stop with `just a1-lingbot stop`.

7. Teleop test:

For a short smoke run, set `collection.max_duration_s = 5.0` in
`configs/teleop/a1_so100.toml`, commit or keep that local config change as
appropriate, then run:

```bash
just a1-teleop doctor
just collect teleop smoke_test
```

Confirm that the relay becomes `ACTIVE`, the arm follows the leader, and a saved
episode contains `frames.csv`, `metadata.json`, `cam0/`, and `cam1/`.

## Dataset Migration

Raw episode migration:

```bash
just dataset convert-raw --dry-run \
  --source-root data/raw/a1_task \
  --target-root data/processed/a1_task \
  --repo-id galaxea/a1_task
```

LeRobot v2.1 migration plan:

```bash
just dataset migration-plan --kind lerobot-v2.1 --repo-id galaxea/a1_task
```

## Hardware Notes

- Motor zero calibration still uses the vendor service flow:
  disable motors with `/iarm_node_single_arm/function_frame 2`, move to physical
  zero, calibrate with `3`, clear errors with `4`, then enable with `1`.
- If `uv sync` times out on large packages, retry with `UV_HTTP_TIMEOUT=120`.
- PyTorch `2.7.1` supports the RTX 5090 generation used here; driver CUDA must
  be new enough, but local `nvcc` version is not the deciding factor.
