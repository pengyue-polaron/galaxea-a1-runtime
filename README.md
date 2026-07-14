# Galaxea A1 Runtime

LeRobot-native runtime for the Galaxea A1 arm.

This repository is being rebuilt around:

- a fail-closed A1 execution runtime,
- LeRobot v0.6.0,
- LeRobotDataset v3.0,
- SO leader teleoperation collection,
- ACT joint-state policy deployment,
- LingBot-VA, FastWAM, and GR00T N1.7 policy profiles,
- clean module boundaries for safety, hardware IO, datasets, and policies.

SO-100 leader printable parts are kept under
[`assets/cad/so100_leader/`](assets/cad/so100_leader/) instead of the repository
root.

The arm may be powered and reachable while this repo is open. Treat every ROS
publish path as live hardware.

## Safe Command Path

Normal EEF apps and inference code must use:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

The relay starts locked. Direct `/arm_joint_command_host` publishing is only for
explicit hardware debug after stopping the safe runtime.

Teleop collection and ACT joint-state inference use the same relay with staged
jointTracker output:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

## Current Main Commands

Daily local checks, no hardware motion:

```bash
just check
```

Hardware enumeration check, no arm motion:

```bash
just hardware
```

Camera check, no arm motion:

```bash
just cameras
```

LAN dual-camera web preview, no arm motion:

```bash
just camera-web
```

Open `http://<robot-lan-ip>:8088` directly. Stop the standalone camera owner
with `just camera-web-stop`; the same preview endpoint is embedded in Teleop,
ACT, and LingBot, so the LAN URL remains useful while one of those apps owns
the cameras.

EEF hardware acceptance:

```bash
just eef-test
```

SO leader teleop:

```bash
just reset              # restore tracked A1 + SO leader start pose
just teleop-test        # manual leader-to-A1 check
just teleop pick_cube   # record episodes
just stop
```

LingBot:

```bash
just models
just lingbot
tmux attach -t lingbot-a1
just stop
```

LingBot runtime parameters live in
[configs/inference/a1_lingbot_va.toml](configs/inference/a1_lingbot_va.toml).
The tracked command starts the managed deployment policy server before the A1
runtime. The checked-in profile is currently fail-closed until a new checkpoint,
prompt, and dataset quantiles are registered. Edit that file when the
checkpoint, server, prompt, cameras, EEF workspace, execution cadence, or
gripper mapping changes.

Deployment weights are registered under the ignored local `models/` directory;
see [models/README.md](models/README.md). `just models` validates every configured
weight, catches tracked files over 100 MiB, and reports stale Git pack garbage.

ACT joint policy:

```bash
just act
tmux attach -t act-a1
just stop
```

ACT runtime parameters live in
[configs/inference/a1_act_joint.toml](configs/inference/a1_act_joint.toml).
It starts dry-run and step-gated by default. Set `execution.execute = true` in
that tracked file only after static checks, camera checks, and a clear robot
workspace.

Collection, ACT, and LingBot share the same AgentView contract: the D455 is
captured at 640x480 and only `(x=103, y=0, width=480, height=480)` is recorded
or passed to a policy. Camera Web keeps the full view visible and outlines that
actual policy region in red.

Dataset conversion:

```bash
just convert banana_in_the_plate
```

Each experiment has a tracked conversion contract at
`configs/datasets/<experiment>.toml`. One command emits separate EEF LeRobot
v3.0 and v2.1 packages plus a joint-action v3.0 package.

Use motion commands only after the arm is powered on and positioned safely.

## New Package Layout

```text
galaxea_a1_runtime/
  safety.py           # pure fail-closed validation and limiters
  schema.py           # LeRobot v3 state/action/camera contracts
  config.py           # typed runtime and dataset config
  hardware/           # IO protocol, EEF helpers, ROS1 safe adapter
  collection/         # teleop state/action schema and episode helpers
  teleop/             # SO leader to A1 joint mapping helpers
  lerobot/            # Robot adapter, dataset writer, recorder, migration
  policies/           # action normalization and policy profiles
  apps/               # reusable app helpers and app-specific transforms
  runtime/            # static doctor and safety disclosure
assets/cad/            # versioned robot/leader mechanical assets
configs/               # tracked runtime, inference, pose, and dataset contracts
scripts/               # operator entrypoints grouped by runtime/app responsibility
```

## Policy Targets

Profiles currently cover:

- ACT joint-state: `policy.type=act`
- LingBot-VA: `policy.type=lingbot_va`
- FastWAM: `policy.type=fastwam`
- GR00T N1.7: `policy.type=groot`

All policy outputs are normalized into the same A1 runtime action contract
before execution.

## LeRobot Robot Adapter

`GalaxeaA1Robot` exposes a LeRobot-style robot interface. Its generic ROS1
hardware adapter supports EEF translation/delta actions by reading
`/end_effector_pose`, publishing `/a1_ee_target`, and enabling the safe relay.
Joint-space arm execution is not implemented in the generic adapter.

## Original SDK Coverage

The vendored A1 SDK exposes mobiman tracker demos for EEF targets
(`/a1_ee_target`), EEF trajectories (`/arm_target_trajectory`), joint targets
(`/arm_joint_target_position`), gripper topics, FK feedback, and RViz
visualization. This runtime wraps the EEF and joint target paths through the
fail-closed relay. It does not currently expose the upstream EEF trajectory
demo as a first-class `just` command.

There is no standard MoveIt `move_group`/MoveIt config path in this repo. The
Docker/runtime baseline includes RViz and TRAC-IK/mobiman pieces, not a MoveIt
planning stack.

## Teleoperation Collection

Daily recording flow:

```bash
just stop
just check
just hardware
just cameras
just reset
just teleop pick_cube
```

`just reset` moves the A1 and the SO leader to the tracked collection start pose
in [configs/poses/a1_so100_collection_start.toml](configs/poses/a1_so100_collection_start.toml), closes both
grippers, resets both devices concurrently, disables leader torque, and stops
the runtime. `just teleop
<experiment>` then starts the staged joint teleop runtime, records front RGB,
wrist RGB, and A1 state/action data, and keeps the old episode loop:
Enter starts recording, Enter saves, `d` discards, and `q` exits.
After each successful save, it pauses the bridge, automatically restores both
devices, restarts the bridge, and waits for the next episode. Discarding an
episode does not trigger a reset.

Runtime parameters live in [configs/teleop/a1_so100.toml](configs/teleop/a1_so100.toml).
Edit that tracked file when the SO leader port, cameras, topics, state mode,
FPS, automatic post-save reset, joint mapping, or gripper stroke range changes.
The normal collection entrypoint does not take per-run collector flags.

The A1 leader adapter lives in
[galaxea_a1_runtime/teleop/a1_so_leader.py](galaxea_a1_runtime/teleop/a1_so_leader.py):
leader actions use six arm axes `joint0.pos..joint5.pos` plus an independent
`gripper.pos`. The bridge intentionally rejects upstream SO arm names so a
miswired leader cannot treat gripper input as an A1 arm joint. Vendored LeRobot
source stays on the official v0.6.0 baseline.

The raw teleop schema records configurable state modes:

- `eef`: EEF pose plus gripper
- `joint`: six arm joints plus gripper
- `eef_joint`: EEF pose, six arm joints, and gripper; the tracked default

Teleop actions are recorded as `joint_absolute` targets from
`/arm_joint_target_position`. Each saved episode contains `frames.csv`,
`metadata.json`, `cam0/`, and `cam1/`; `cam0_depth/` is present only when
RealSense depth is enabled in the tracked config. The metadata records the
state topics, action topics, cameras, and staged relay control path used for
that episode.

The default tracked teleop config is USB2-compatible RGB-only. Depth capture is
still supported, but it should be enabled intentionally in
`configs/teleop/a1_so100.toml` after the RealSense is on a stable USB3 link or
after lowering the camera FPS/resolution for USB2. During recording, stale
camera samples abort the episode and delete the partial folder instead of
saving bad data.

AgentView is captured at 640x480 and cropped to the tracked square ROI
`x=103, y=0, width=480, height=480` before `cam0` is written. The LAN preview
keeps the full frame and draws the recorded area in red; the overlay is not
saved. Wrist frames remain uncropped.
The collector rejects appending 480x480 AgentView frames to an older raw
experiment containing 640x480 frames; use a new experiment name or migrate the
whole existing experiment first.

Enter requests a save; it does not make the episode durable immediately. The
collector first checks joint-action continuity using the tracked
`collection.max_joint_action_step_rad` threshold. A failed check prints the
frame, joint, values, and limit, deletes the episode, reuses its index, and
homes both devices before the next attempt.

Raw episodes are written under `data/raw/<experiment>/episode_NNN_timestamp/`.
Convert a selected source dataset to its tracked training package with
`just convert <experiment>`.

The tracked rig binds the agent D455 and wrist D405 by explicit RealSense
serial number. `just cameras` captures `cam0_front.jpg`, optional `cam0_depth.png` plus
`cam0_depth_preview.jpg`, `cam1_wrist.jpg`, and a contact sheet from the same
tracked camera config without moving the arm. It also runs a short sustained
FPS probe and prints the RealSense USB link type.

The shared web preview listens on the host's LAN interfaces at port `8088` and
has no ROS or robot-control endpoints. It has no login, so do not port-forward
this unencrypted HTTP service to the public Internet. Only one process may own the
RealSense devices: use standalone `just camera-web` when no robot app is
running; Teleop, ACT, and LingBot reuse their existing readers for preview.

## Dependency Baseline

The main Python environment is pinned to official LeRobot v0.6.0:

```text
30da8e687a6dfc617fcd94afc367ac7071c376ce
```

Python baseline is `>=3.12,<3.13`.

`third_party/lerobot` is also replaced with the official v0.6.0 checkout at the
same commit.

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Runbook](docs/RUNBOOK.md)
- [Safety](docs/SAFETY.md)

## Legacy Systems

The previous ZMQ/OpenPI/LeRobot v2.1 mainline stack has been removed. Dataset
conversion remains one-way under `just convert <experiment>`.
