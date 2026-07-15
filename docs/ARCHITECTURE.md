# Architecture

The repo is now a small Galaxea A1 runtime around LeRobot v0.6.0 and
LeRobotDataset v3.0.

## Layers

1. Core package: `galaxea_a1_runtime`
   - `schema.py`: LeRobotDataset v3 state/action/camera contracts.
   - `safety.py`: pure relay and limit checks.
   - `collection/`: teleop raw state/action schema and episode helpers.
   - `teleop/`: pure SO leader to A1 joint mapping helpers.
   - `hardware/`: IO protocol, EEF helpers, shared camera IO, and web preview.
   - `policies/`: normalized runtime action contract.
   - `apps/`: managed implementations split by app and responsibility.
   - `lerobot/`: explicit-IO `GalaxeaA1Robot`, writer helpers, passive recorder,
     migration, and shared deterministic package primitives.
   - `runtime/`: doctor, safety report, shared relay/feedback caches, and the
     ROS1 Python-path bootstrap.

2. Base runtime: `scripts/runtime`

   - `a1_services.sh` is the thin, app-agnostic owner of repeated Docker,
     ROS-master, driver, feedback-wait, and locked-relay startup primitives.
   - `a1_tmux.sh` owns only repeated tmux start, stop, status, and log-capture
     primitives; app scripts still decide startup checks and cleanup policy.
   - `a1_runtime.sh`, `a1_joint_runtime.sh`, and the teleop runtime retain their
     own tracker choice, lifecycle, doctors, UI, and failure policy.
   - Owns ROS master, A1 driver, isolated EE/joint trackers, and safe relay.
   - Must stay app-agnostic and LingBot-free.

3. App runtime: `scripts/apps/<app>`
   - Provides thin operator/process-lifecycle entrypoints at stable paths.
   - Dispatches stateful Python work to `galaxea_a1_runtime/apps/<app>/`, where
     protocol, camera, ROS state, recording, policy, and CLI responsibilities
     are separated and testable.

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

The former LingBot entrypoint was too large because it did four jobs in one
file:

- model protocol: LingBot WebSocket and KV-cache updates
- observation IO: AgentView and wrist color cameras plus preview
- policy action transforms: LingBot 8D action cleanup
- A1 execution: EEF target publishing, relay enable, staged gripper target, feedback

The reusable parts now live in package modules:

- `galaxea_a1_runtime.runtime.relay` and `.ros_feedback`: app-agnostic,
  thread-safe relay, joint, gripper, and staged-command feedback monitors used
  by ACT, Teleop, and LingBot.
- `galaxea_a1_runtime.apps.eef_bridge`: EEF feedback helpers, state conditioning
  shape helper, direction formatting, and ROS-like EEF/gripper publisher.
- `galaxea_a1_runtime.apps.lingbot.actions`: LingBot-specific 8D action
  sanitation, workspace bounds, orientation behavior, gripper mapping, and
  optional tracker compensation.
- `galaxea_a1_runtime.apps.lingbot.client`: WebSocket/MessagePack protocol.
- `galaxea_a1_runtime.apps.policy_camera`: shared ACT/LingBot exclusive camera
  ownership, freshness, AgentView crop validation, and read-only preview
  lifecycle.
- `galaxea_a1_runtime.apps.lingbot.episode_state`, `.rollout`, and `.review`:
  fresh episode coordinates, validated action-tensor indexing, and operator
  previews without ROS process ownership.
- `galaxea_a1_runtime.apps.lingbot.bridge` and `.cli`: ROS/camera rollout
  orchestration and the config-file entrypoint, respectively.
- `galaxea_a1_runtime.apps.lingbot.config_schema`, `.config`, and
  `.config_runtime`: typed deployment schema, TOML loading/validation, and
  lifecycle values used by `just lingbot`.
- `galaxea_a1_runtime.apps.act.config_schema`, `.config`, and `.config_runtime`:
  typed ACT schema, TOML loading/validation, and shell lifecycle values.
- `galaxea_a1_runtime.apps.act.actions`, `.policy`, `.bridge`, and `.cli`: pure
  model-action validation, checkpoint inference, guarded execution, and
  config loading; feedback caches come directly from the shared runtime
  layer.
- `galaxea_a1_runtime.lerobot.dataset_package`: one implementation of dataset
  tree copying, hard-link fallback, vector statistics, digesting, JSON, and
  atomic archives shared by the joint and LingBot packagers.
- `galaxea_a1_runtime.hardware.cameras`: shared RealSense color/depth and
  OpenCV color camera wrappers used by teleop collection, camera snapshots, and
  LingBot.
- `galaxea_a1_runtime.hardware.web_preview`: read-only LAN MJPEG
  encoding and HTTP service. It consumes existing latest-frame readers and
  never opens cameras or imports ROS. Teleop, ACT, and LingBot therefore expose
  preview without creating a second RealSense owner.
- `galaxea_a1_runtime.filesystem`: process-safe staging and atomic install
  primitives shared by raw collection and offline dataset packaging.

Future policy integrations should reuse `apps.eef_bridge`, keep their operator
scripts thin, and add a tracked deployment only when an executable integration
exists; this repository does not keep speculative profile registries.

ACT is the built-in joint-state policy deployment path. It loads a local
LeRobot ACT checkpoint, reads front/wrist RGB plus six A1 joints and continuous
normalized gripper state, predicts absolute joint targets, and publishes only
`/arm_joint_target_position`. The isolated jointTracker stages motor commands,
then the relay guards the final host command topic. The bridge starts dry-run
and step-gated from `configs/deployments/act_joint.toml`.

## Teleop Collection Design

Teleop is the built-in demonstration collection mode.

- `scripts/apps/teleop/so100_joint_bridge.py`: stable, thin bridge entrypoint.
- `galaxea_a1_runtime.apps.teleop.bridge`: reads an SO leader, maps it to A1
  joint targets, publishes `/arm_joint_target_position` and
  `/a1_gripper_target`, and arms the relay.
- `galaxea_a1_runtime.teleop.a1_so_leader`: A1-specific SO leader motor layout
  (`joint0..joint5` plus gripper) built on LeRobot motor primitives without
  patching vendored LeRobot source.
- `scripts/apps/teleop/teleop_collect.py`: stable, thin collector entrypoint.
- `galaxea_a1_runtime.apps.teleop`: separates fresh ROS state assembly,
  frame recording, camera ownership, single-episode validation/persistence,
  collector orchestration, reset config, A1 reset, and leader reset. The
  collector does not command the robot. Background
  readers prevent a blocking camera read from silently stopping state/action
  recording; stale required streams abort and remove the staged episode.
  Enter=save runs quality and file-count checks before an atomic commit.
  Metadata includes the tracked teleop config path along with topics, cameras,
  freshness thresholds, and the staged relay path.
- `scripts/apps/cameras/a1_camera_diagnostics.py`: captures front/wrist snapshots from
  `configs/system/a1.toml` without starting ROS or moving the robot, and probes
  sustained camera FPS plus the RealSense USB link type. Its output root,
  timeouts, probe duration, and encoding quality are system-config values, not
  CLI defaults.
- `scripts/apps/cameras/a1_camera_web.py`: standalone owner using the same system
  config for the tracked D455
  agent camera and D405 wrist camera when no Teleop/inference app is running.
  It serves the same shared preview module used inside all three app chains
  and draws the configured AgentView collection ROI on the full frame. Teleop
  uses the same pure ROI helper to crop RGB and aligned depth before writing.
- `scripts/apps/teleop/a1_teleop_runtime.sh`: starts/stops ROS, driver, staged
  joint tracker, relay, bridge, and recorder. It also owns the post-episode
  sequence that pauses the bridge, homes both devices, and resumes the bridge.
- `configs/system/a1.toml`: shared physical contract for cameras, topics,
  relay/doctor/startup timing, EEF acceptance steps, workspaces, joint limits,
  and gripper stroke.
- First-party tracker launch files deliberately have no topic defaults. Runtime
  entrypoints pass every endpoint rendered from the system contract, so a topic
  change cannot silently leave a stale launch fallback behind.
- `configs/teleop/a1_so100.toml`: teleop-only leader, joint mapping, and
  collection contract.
- `configs/deployments/`: checkpoint-specific model and rollout contracts.

`SystemConfig` is the only typed physical configuration. ACT, LingBot, and
teleop config objects retain a direct `system: SystemConfig` reference and only
define model/app semantics. App processes load their TOML directly and do not
serialize topics, relay thresholds, cameras, EEF bounds, joint limits, or
gripper range through a second CLI contract.
The shared tmux supervisor receives its startup observation grace explicitly
from `SystemConfig.startup`; it contains no independent timing default.
Camera tables are strict tagged schemas: RealSense entries contain serial,
sensor, and optional depth settings, while V4L2 entries contain device/backend
and pixel-format settings. Inapplicable backend keys are rejected instead of
being retained as empty or ignored values; wrist depth fields are absent
because that path is unsupported.
Crop coordinates, manual exposure/gain/white-balance values, and depth stream
dimensions are likewise valid only when their enabling flag selects that mode;
disabled features must not carry inert fallback numbers.
The former parallel `RuntimeConfig/TopicConfig/SafetyConfig` stack and app-level
physical mirror dataclasses have been removed. Dataset writer settings remain a
data-layer `DatasetConfig`; they contain no hardware topics, limits, or gripper
range.

Repository-owned Docker containers and tmux sessions receive a managed marker.
`just stop` first uses each app's precise lifecycle command and then invokes the
configuration-independent marked-resource fallback, so a temporarily invalid
TOML file cannot prevent shutdown and unrelated user resources remain untouched.
The final fallback is authoritative: `just stop` exits nonzero instead of
claiming success when marked containers or tmux sessions cannot be removed.

Each app config follows the same three-part shape: `config_schema.py` contains
immutable typed values, `config.py` loads and validates TOML, and
`config_runtime.py` exposes only shell-owned lifecycle values such as tmux
names and model paths. Python bridges, cameras, collectors, and doctors receive
the typed config directly.

Operator presentation is also shared rather than app-specific.
`galaxea_a1_runtime.console` owns Python argument errors, help styling, and the
`INFO/STEP/PASS/WARN/FAIL` vocabulary; `scripts/runtime/a1_console.sh` mirrors
that contract for shell. `a1_tmux.sh` owns early-process-exit detection, so app
scripts provide an exit marker instead of copying startup sleeps and pane
inspection. Structured JSON, shell exports, and diagnostic `key=value` output
remain uncolored.

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

The shared reset implementation uses
`configs/poses/a1_so100_collection_start.toml` to restore the A1 and SO leader
concurrently and close both grippers. The pose config contains targets and
motion semantics only; the owning Teleop config injects leader identity and
mapping, while its System reference supplies joint names, limits, topics, and
relay timings. Its orchestrator delegates A1 relay and
joint monitoring to `reset_a1.py`, leader IO to `reset_leader.py`, and display
state to `reset_progress.py`. `just reset` starts the required services, runs
that implementation, and stops the runtime. The post-save path reuses it while
keeping the services and cameras alive, then restarts the bridge for the next
episode.

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
- Generic LeRobot robot wrapper: schema/IO composition only; callers must inject
  an explicit `A1HardwareIO`, and managed apps own all live ROS control paths.
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

- LingBot's ROS publishing and interactive rollout orchestration intentionally
  remain together; protocol, shared policy-camera ownership, episode state,
  tensor indexing, review output, configuration, and CLI are independent
  modules.
