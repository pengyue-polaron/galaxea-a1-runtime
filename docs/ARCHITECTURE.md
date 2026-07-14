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
   - `runtime/`: doctor and safety report.

2. Base runtime: `scripts/runtime`
   - Owns ROS master, A1 driver, isolated EE/joint trackers, and safe relay.
   - Must stay app-agnostic and LingBot-free.

3. App runtime: `scripts/apps/<app>`
   - Owns model servers, cameras, prompts, teleop leaders, and UI loops.
   - Reuses shared EEF bridge helpers instead of publishing directly to host
     joint commands.

4. Data tools: `scripts/process_data`
   - Migration and diagnostic utilities only.

5. Mechanical assets: `assets/cad/<device>`
   - Versioned printable/source-export geometry, separated from runtime code.
   - `assets/cad/so100_leader/` contains the leader-arm STL set and an origin
     filename manifest.

## Infra Policy

- `Justfile` exposes the small operator surface; scripts can stay detailed, but
  daily commands should remain short.
- Runtime behavior composes one tracked physical contract in
  `configs/system/a1.toml` with app or model contracts under `configs/teleop/`
  and `configs/deployments/`.
- `third_party/` contains reproducible vendor snapshots listed in
  `third_party/vendors.toml`. A1-specific behavior belongs in
  `galaxea_a1_runtime/` or `scripts/apps/`, not vendor patches.
- `just check` is the hardware-free gate for docs, vendor boundaries, pure
  imports, safety logic, and static app structure.

## Safe Control Path

Normal apps must use:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

Direct `/arm_joint_command_host` publishing is only for explicit direct-debug
hardware work after stopping the safe runtime.

Teleop joint control and ACT joint-policy inference use the same relay, but
with a joint tracker:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

All normal gripper actions share one staged path:

```text
/a1_gripper_target
  -> safe_arm_command_relay.py
  -> /gripper_position_control_host
```

The physical `0..100 mm` range, both topics, and relay timing live only in
`configs/system/a1.toml`. Apps and converters receive those typed values; they
do not define fallback stroke ranges.

## Reusable Policy Bridge Design

The LingBot bridge used to be too large because it did four jobs in one file:

- model protocol: LingBot WebSocket and KV-cache updates
- observation IO: RealSense color, optional RealSense depth, and wrist color cameras
- policy action transforms: LingBot 8D action cleanup
- A1 execution: EEF target publishing, relay enable, staged gripper target, feedback

The reusable parts now live in package modules:

- `galaxea_a1_runtime.apps.eef_bridge`: relay status parsing, EEF feedback
  helpers, state conditioning shape helper, direction formatting, and ROS-like
  EEF/gripper command publisher.
- `galaxea_a1_runtime.apps.lingbot.actions`: LingBot-specific 8D action
  sanitation, workspace bounds, orientation behavior, gripper mapping, and
  optional tracker compensation.
- `galaxea_a1_runtime.apps.lingbot.config`: tracked LingBot runtime config
  loading and conversion into the bridge arguments used by `just lingbot`.
- `galaxea_a1_runtime.apps.act.config`: tracked ACT joint runtime config
  loading and conversion into the bridge arguments used by `just act`.
- `galaxea_a1_runtime.hardware.cameras`: shared RealSense color/depth and
  OpenCV color camera wrappers used by teleop collection, camera snapshots, and
  LingBot.
- `galaxea_a1_runtime.hardware.web_preview`: read-only LAN MJPEG
  encoding and HTTP service. It consumes existing latest-frame readers and
  never opens cameras or imports ROS. Teleop, ACT, and LingBot therefore expose
  preview without creating a second RealSense owner.

Future FastWAM/GR00T app scripts should reuse `apps.eef_bridge` and only provide
their own model IO plus action conversion into the normalized A1 EEF contract.

ACT is the built-in joint-state policy deployment path. It loads a local
LeRobot ACT checkpoint, reads front/wrist RGB plus six A1 joints and continuous
normalized gripper state, predicts absolute joint targets, and publishes only
`/arm_joint_target_position`. The isolated jointTracker stages motor commands,
then the relay guards the final host command topic. The bridge starts dry-run
and step-gated from `configs/deployments/act_joint.toml`.

## Teleop Collection Design

Teleop is the built-in demonstration collection mode.

- `scripts/apps/teleop/so100_joint_bridge.py`: reads an SO leader, maps it to
  A1 joint targets, publishes `/arm_joint_target_position` and
  `/a1_gripper_target`, and arms the relay.
- `galaxea_a1_runtime.teleop.a1_so_leader`: A1-specific SO leader motor layout
  (`joint0..joint5` plus gripper) built on LeRobot motor primitives without
  patching vendored LeRobot source.
- `scripts/apps/teleop/teleop_collect.py`: records synchronized front RGB,
  optional front depth, wrist RGB, A1 state, and target actions. It does not
  command the robot. Camera streams are read by background latest-frame readers
  so a blocking camera read cannot silently stop state/action recording. Each
  frame requires fresh front/wrist samples; stale cameras abort and delete the
  partial episode. Enter=save runs the pure collection quality checks before
  metadata is committed; discontinuous joint actions reject and delete the
  episode. Episode metadata records the state topics, action topics, cameras,
  quality thresholds, and staged relay path.
- `scripts/apps/cameras/a1_camera_diagnostics.py`: captures front/wrist snapshots from
  the same tracked config without starting ROS or moving the robot, and probes
  sustained camera FPS plus the RealSense USB link type.
- `scripts/apps/cameras/a1_camera_web.py`: standalone owner for the tracked D455
  agent camera and D405 wrist camera when no Teleop/inference app is running.
  It serves the same shared preview module used inside all three app chains
  and draws the configured AgentView collection ROI on the full frame. Teleop
  uses the same pure ROI helper to crop RGB and aligned depth before writing.
- `scripts/apps/teleop/a1_teleop_runtime.sh`: starts/stops ROS, driver, staged
  joint tracker, relay, bridge, and recorder. It also owns the post-episode
  sequence that pauses the bridge, homes both devices, and resumes the bridge.
- `configs/system/a1.toml`: shared physical contract for cameras, topics,
  relay timing, workspaces, joint limits, and gripper stroke.
- `configs/teleop/a1_so100.toml`: teleop-only leader, joint mapping, and
  collection contract.
- `configs/deployments/`: checkpoint-specific model and rollout contracts.

`SystemConfig` is the only typed physical configuration. The former parallel
`RuntimeConfig/TopicConfig/SafetyConfig` stack has been removed. Dataset writer
settings remain a data-layer `DatasetConfig`; they contain no hardware topics,
limits, or gripper range.

The episode interaction is:

```text
first run: enter task prompt
each episode: Enter=start, Enter=save and auto-reset, d=discard, q=quit
```

The tracked default records combined EEF pose and joint state. Joint-only and
EEF-only collection remain explicit alternatives. These options are changed in
the tracked config file, not through ad hoc per-run collection flags.

The recording data flow is intentionally split by responsibility:

```text
SO leader
  -> so100_joint_bridge.py
  -> /arm_joint_target_position
  -> staged jointTracker + safe relay
  -> A1 driver

A1 feedback + cameras
  -> teleop_collect.py
  -> data/raw/<experiment>/episode_NNN_timestamp/
  -> convert_raw.py
  -> LeRobotDataset v3 output
```

The bridge is the only module that commands teleop motion during recording. The
recorder reads ROS state, the latest joint target action, front RealSense RGB,
optional depth, and wrist RGB, then writes synchronized raw episode files. Once
a save is durable, it asks the runtime shell to perform the shared reset
workflow; it does not implement or publish reset commands itself.

The shared reset implementation uses `configs/poses/a1_so100_collection_start.toml` to restore
the A1 and SO leader concurrently and close both grippers. `just reset` starts
the required services, runs that implementation, and stops the runtime. The
post-save path reuses it while keeping the services and cameras alive, then
restarts the bridge for the next episode.

## Original SDK Feature Coverage

The vendored Galaxea SDK exposes:

- EEF target tracking with `mobiman eeTrackerdemo.launch` and `/a1_ee_target`.
- EEF trajectory tracking with `mobiman eeTrajTrackerdemo.launch` and
  `/arm_target_trajectory`.
- Joint target tracking with `mobiman jointTrackerdemo.launch` and
  `/arm_joint_target_position`.
- RViz visualization, FK `/end_effector_pose`, and gripper command/status
  topics.

The current runtime wraps the EEF target path and joint target path through the
staged relay. The EEF trajectory demo and RViz launch remain SDK/debug
capabilities, but are not first-class daily `just` commands. Standard MoveIt
`move_group` support is not present in the current repo or Docker baseline.

## Current Capabilities

- Static checks: `just check`, or `just test` for tests only.
- Camera check: `just cameras`.
- Manual EEF acceptance: `just eef-test`.
- Teleop collection: `just teleop <experiment>`.
- Manual teleop acceptance: `just teleop-test`, `just logs`, `just stop`.
- LingBot app: `just lingbot` manages the registered deployment policy server and A1 bridge;
  inspect them with `tmux attach -t lingbot-va-server` and
  `tmux attach -t lingbot-a1`.
- ACT joint policy: `just act`, then `tmux attach -t act-a1`.
- Dataset conversion: `just convert <experiment>` using
  `configs/datasets/<experiment>.toml`; each run emits independent LeRobot
  EEF v3.0, EEF v2.1, and joint-action v3.0 packages.
- Generic LeRobot robot adapter: safe EEF translation/delta actions through
  `/a1_ee_target`; rejects `joint_absolute`.
- LingBot app: finite continuous inference and publishing, relay guard,
  episode-relative EEF state conditioning, absolute workspace validation, and
  shared continuous gripper execution.
- ACT app: dry-run default, local checkpoint loading, action preview, staged
  jointTracker alignment, relay guard, joint limit/step checks, and continuous
  normalized gripper execution.
- Dataset: LeRobotDataset v3 contract, writer helpers, passive episode recorder,
  teleop raw migration, v2.1 migration plan, raw RealSense depth capture, and
  `observation.images.front_depth` conversion for depth-enabled teleop episodes.

## Intentionally Not Done Yet

- FastWAM and GR00T have profiles but not dedicated A1 app scripts yet.
- LingBot still keeps model protocol and the interactive operator loop in one
  script; config loading, camera IO, action transforms, and A1 EEF execution
  pieces are reusable.
