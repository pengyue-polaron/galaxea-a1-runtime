# Runbook

This file is the short operator path. Edit tracked config files when hardware
settings change; avoid ad hoc per-run flags for data collection.

## Daily Check

No hardware motion:

```bash
just check
```

## Hardware Acceptance

Use this after reconnecting all hardware and clearing the workspace.

1. Stop stale processes and inspect cameras:

```bash
just stop
just cameras
```

Check the printed `cam0_front`, `cam1_wrist`, and `contact_sheet` image paths.

2. Test EEF control:

```bash
just eef-test
```

The tool holds the current EEF pose, then waits for Enter before each
`x+/x-/y+/y-/z+/z-` step. Confirm the EEF moves in the printed direction.

3. Test SO leader teleop:

```bash
just teleop-test
just logs
```

Confirm the bridge log prints `leader_keys=['joint0.pos', ..., 'joint5.pos']`.
Move one leader joint at a time and verify the A1 direction matches the old
working behavior. Test gripper open/close.

4. Stop:

```bash
just stop
```

## Teleop Collection

```bash
just cameras
just teleop pick_cube
```

First run asks for the task prompt and writes `data/raw/pick_cube/task.txt`.
Each episode uses:

```text
Enter=start recording
Enter=save
d+Enter=discard
q+Enter=quit
```

Teleop behavior is locked by `configs/teleop/a1_so100.toml`: leader port,
cameras, state mode, FPS, topics, joint mapping, limits, and gripper range.

Recorded state modes:

- `joint`: `joint_1..joint_6`, `gripper`, default.
- `eef`: `eef_x/y/z/qx/qy/qz/qw`, `gripper`.
- `eef_joint`: both.

Actions are recorded as `joint_absolute` targets from
`/arm_joint_target_position`.

## LingBot-VA

Start the LingBot server separately, then:

```bash
just lingbot
tmux attach -t lingbot-a1
```

The bridge is step-gated:

- `INFERENCE #N READY`: Enter runs one new model inference.
- `Next=publish this EE step`: Enter publishes one already predicted EEF step.

Stop with:

```bash
just stop
```

## Dataset Conversion

```bash
just convert-raw --dry-run \
  --source-root data/raw/a1_task \
  --target-root data/processed/a1_task \
  --repo-id galaxea/a1_task
```

Remove `--dry-run` after inspecting the planned conversion.
