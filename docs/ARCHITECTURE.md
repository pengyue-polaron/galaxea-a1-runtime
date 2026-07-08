# Architecture

The repo is now a small Galaxea A1 runtime around LeRobot v0.6.0 and
LeRobotDataset v3.0.

## Layers

1. Core package: `galaxea_a1_runtime`
   - `schema.py`: LeRobotDataset v3 state/action/camera contracts.
   - `safety.py`: pure relay and limit checks.
   - `collection/`: teleop raw state/action schema and episode helpers.
   - `teleop/`: pure SO leader to A1 joint mapping helpers.
   - `hardware/`: IO protocol, EEF helpers, ROS1 safe adapter, shared camera IO.
   - `policies/`: normalized action contract and policy profile metadata.
   - `apps/`: reusable app helpers plus app-specific transforms.
   - `lerobot/`: `GalaxeaA1Robot`, writer helpers, passive recorder, migration.
   - `runtime/`: doctor, safety report, dry-run plans.

2. Base runtime: `scripts/runtime`
   - Owns ROS master, A1 driver, isolated EE tracker, and safe relay.
   - Must stay app-agnostic and LingBot-free.

3. App runtime: `scripts/apps/<app>`
   - Owns model servers, cameras, prompts, teleop leaders, and UI loops.
   - Reuses shared EEF bridge helpers instead of publishing directly to host
     joint commands.

4. Data tools: `scripts/process_data`
   - Migration and diagnostic utilities only.

## Safe Control Path

Normal apps must use:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay_v2.py
  -> /arm_joint_command_host
```

Direct `/arm_joint_command_host` publishing is only for explicit direct-debug
hardware work after stopping the safe runtime.

Teleop joint control uses the same relay, but with a joint tracker:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay_v2.py
  -> /arm_joint_command_host
```

## Reusable Policy Bridge Design

The LingBot bridge used to be too large because it did four jobs in one file:

- model protocol: LingBot WebSocket and KV-cache updates
- observation IO: RealSense and wrist cameras
- policy action transforms: LingBot 8D action cleanup
- A1 execution: EEF target publishing, relay enable, gripper publish, feedback

The reusable parts now live in package modules:

- `galaxea_a1_runtime.apps.eef_bridge`: relay status parsing, EEF feedback
  helpers, state conditioning shape helper, direction formatting, and ROS-like
  EEF/gripper command publisher.
- `galaxea_a1_runtime.apps.lingbot.actions`: LingBot-specific 8D action
  sanitation, workspace bounds, orientation behavior, gripper mapping, and
  optional tracker compensation.
- `galaxea_a1_runtime.hardware.cameras`: shared RealSense/OpenCV color camera
  wrappers used by teleop collection, camera snapshots, and LingBot.

Future FastWAM/GR00T app scripts should reuse `apps.eef_bridge` and only provide
their own model IO plus action conversion into the normalized A1 EEF contract.

## Teleop Collection Design

Teleop is the built-in demonstration collection mode.

- `scripts/apps/teleop/so100_joint_bridge.py`: reads an SO leader, maps it to
  A1 joint targets, publishes `/arm_joint_target_position`, and arms the relay.
- `scripts/apps/teleop/teleop_collect.py`: records synchronized front/wrist
  images, A1 state, and target actions. It does not command the robot. Episode
  metadata records the state topics, action topics, and staged relay path.
- `scripts/apps/teleop/camera_snapshot.py`: captures front/wrist snapshots from
  the same tracked config without starting ROS or moving the robot.
- `scripts/apps/teleop/a1_teleop_runtime.sh`: starts/stops ROS, driver, staged
  joint tracker, relay, bridge, and recorder.
- `configs/teleop/a1_so100.toml`: tracked runtime contract for leader port,
  cameras, topics, joint mapping, gripper range, state mode, and FPS.

The episode interaction is:

```text
first run: enter task prompt
each episode: Enter=start, Enter=save, d=discard, q=quit
```

The default recorded state is joint-space to match the old working collector.
EEF and combined EEF+joint state are explicit opt-in modes.
Those options are changed in the tracked config file, not through ad hoc
per-run collection flags.

## Current Capabilities

- Static checks: `just runtime doctor`, `just runtime safety`, `just runtime test`.
- Safe hardware runtime: `just a1-runtime services|doctor|status|logs|stop`.
- Manual EEF acceptance: `just a1-runtime eef-nudge --execute`.
- Teleop collection: `just collect teleop <experiment>`.
- Teleop camera check: `just a1-teleop cameras`.
- Generic LeRobot robot adapter: safe EEF translation/delta actions through
  `/a1_ee_target`; rejects `joint_absolute`.
- LingBot app: step-gated inference and publishing, relay guard, EEF state
  conditioning, workspace validation, linear gripper mapping.
- Dataset: LeRobotDataset v3 contract, writer helpers, passive episode recorder,
  teleop raw migration, v2.1 migration plan.

## Intentionally Not Done Yet

- FastWAM and GR00T have profiles but not dedicated A1 app scripts yet.
- LingBot still keeps model protocol and the interactive loop in one script; the
  camera IO, action transforms, and A1 EEF execution pieces are reusable.
