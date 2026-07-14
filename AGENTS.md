# Galaxea A1 Runtime Agent Notes

This repo controls a real Galaxea A1 arm. Prefer boring, explicit, fail-closed
changes over clever shortcuts.

The arm may be physically powered and reachable while you work. Treat every ROS
publish path as potentially live hardware unless the user explicitly says the
arm is disconnected.

## Layering

- Keep `scripts/runtime/a1_runtime.sh` LingBot-free. It owns only ROS,
  the A1 driver, the isolated EE tracker, and the safe relay.
- Keep `scripts/runtime/a1_joint_runtime.sh` app-agnostic. It owns only ROS,
  the A1 driver, the isolated joint tracker, and the safe relay.
- Keep app-specific logic in app scripts. LingBot belongs under
  `scripts/apps/lingbot/a1_lingbot_runtime.sh` and
  `scripts/apps/lingbot/lingbot_va_ee_bridge.py`.
- Teleop collection, inference, and data processing should
  use shared runtime/doctor concepts but should not depend on LingBot being
  installed or running.

## Model Storage

- Inference configs must reference deployment weights through the ignored local
  registry under `models/`; do not point tracked configs directly at a user's
  home directory or a native training output.
- Keep downloaded base models under `models/base/`, trained deployment exports
  under `models/checkpoints/<app>/<run>/`, and generated component assemblies
  under `models/runtime/`.
- Register existing files with `just model-link <slot> <source>`. The registry
  uses symlinks so large weights are not copied.
- Native training jobs may continue writing to `train_out/` or `outputs/train/`.
  Treat those as sources, `outputs/` as run logs/results, and `data/` as datasets.
- Never commit model weights or use Git LFS for them in this repo. Run
  `just models` before inference and after any interrupted large Git operation.

## ROS Control Paths

### Safe Runtime Path

- Normal app/inference code must not publish directly to
  `/arm_joint_command_host`.
- Route EE commands through:

  ```text
  /a1_ee_target
    -> isolated eeTracker
    -> /arm_joint_command_a1_staged
    -> safe_arm_command_relay.py
    -> /arm_joint_command_host
  ```

- The relay enable topic is `/a1_arm_motion_enable`; the relay status topic is
  `/a1_arm_relay_status`.
- The relay starts `LOCKED`. It only publishes to `/arm_joint_command_host`
  after an app explicitly enables motion and validation passes.

### Safe Teleop Path

- Teleop apps publish joint targets to `/arm_joint_target_position`, not host
  motor commands.
- ACT joint-policy inference also publishes joint targets to
  `/arm_joint_target_position`, not host motor commands.
- Route joint-space commands through:

  ```text
  /arm_joint_target_position
    -> isolated jointTracker
    -> /arm_joint_command_a1_staged
    -> safe_arm_command_relay.py
    -> /arm_joint_command_host
  ```

### Teleop Config Contract

- The normal teleop collection entrypoint is:

  ```bash
  just teleop <experiment>
  ```

- Teleop app/data semantics are controlled by the tracked config file:

  ```text
  configs/teleop/a1_so100.toml
  ```

- Do not change teleop behavior with ad hoc per-run collector flags. Camera
  devices, depth, physical gripper range, topics, and joint limits live in
  `configs/system/a1.toml`; leader mapping and collection semantics live in
  `configs/teleop/a1_so100.toml`.
- For an alternate tracked hardware setup, add another TOML file under
  `configs/teleop/` and run the app script with an explicit config path:

  ```bash
  scripts/apps/teleop/a1_teleop_runtime.sh --config configs/teleop/my_setup.toml collect pick_cube
  ```

- The old working teleop behavior is the compatibility baseline:
  - SO leader by-id port
    `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A016967-if00`,
    id `my_leader`.
  - The first-party `A1SOLeader` adapter is intentionally shaped as six arm
    axes `joint0..joint5` plus an independent `gripper`; do not replace it with
    the upstream five-axis `shoulder_*`/`wrist_*` naming unless the hardware is
    deliberately changed.
  - Unknown leader action key layouts should fail loudly instead of falling back
    to sorted `*.pos` keys.
  - Relative leader-to-A1 joint mapping from startup pose.
  - Sign mapping `[-1, 1, 1, -1, 1, -1]`.
  - Default collected state mode `eef_joint` so every frame preserves both EEF
    pose and joint state.
  - Default collection FPS `30`.
  - Gripper state and policy actions are continuous normalized values:
    `0=minimum stroke`, `1=maximum stroke`. The SO leader, collector, dataset
    converters, ACT, and LingBot all use the physical stroke range from
    `configs/system/a1.toml`; do not add per-app thresholds or binary rewrites.
  - `/gripper_stroke_host` is the only collection/inference feedback source.
    Do not reinterpret the seventh `/joint_states_host` value as millimeters.
- If any of those defaults must change, update the owning tracked config,
  docs, and behavioral tests in the same change.

### Official/Original ROS Debug Path

- For explicit hardware debugging, it is acceptable to bypass the relay and use
  the Galaxea-style direct path:

  ```text
  /a1_ee_target
    -> eeTracker_demo_node
    -> /arm_joint_command_host
    -> single_arm_node
  ```

- Before starting the direct path, stop the safe runtime:

  ```bash
  just stop
  ```

- In this repo, the most reliable direct debug launch is the isolated tracker
  launch with its output remapped back to the official command topic:

  ```bash
  roslaunch /workspace/scripts/runtime/ee_tracker_staged.launch \
    staged_command_topic:=/arm_joint_command_host
  ```

  This preserves the original topic semantics without starting the safe relay.
  Mount the repo read-write inside Docker because `mobiman` may write generated
  CppAD files under `third_party/A1_SDK/install/share/mobiman/auto_generated`.

- The upstream official `mobiman eeTrackerdemo.launch` publishes directly to
  `/arm_joint_command_host`, but it may also try to start GUI/RViz pieces. For
  headless debugging, prefer the direct remap above unless the user explicitly
  wants the full official launch.
- The vendored SDK also has `eeTrajTrackerdemo.launch` for
  `/arm_target_trajectory` and `jointTrackerdemo.launch` for
  `/arm_joint_target_position`. This repo does not currently provide a standard
  MoveIt `move_group` path.

## Hardware Safety

- When the arm is powered off, do only static checks or non-execution doctors.
  Use `--require-execution` only after the user confirms the arm is powered on
  and positioned safely.
- If startup fails midway, stop the runtime with `just stop` before retrying.
- If switching between safe runtime and direct debug mode, stop the previous
  containers first. Do not leave two trackers or drivers fighting over the same
  topics/serial device.

## A1 Status Codes Observed Here

- `/arm_status_host` code `64` is bit 6,
  `ACU Feedback: ECU -> ACU Timeout`.
- On this arm, pure `64` can be a normal idle/no-upstream-command condition and
  should not by itself block runtime startup.
- During actual arm execution, the useful ready signal is usually:

  ```text
  arm joints 1-6: 0
  gripper: 64 is acceptable
  ```

- Pure `64` is treated as non-blocking by the runtime/LingBot doctors and relay
  safety core. Error codes with additional bits, such as `68` (`64 + motor
  disconnected`), should still be treated as faults.
- Gripper control has been verified to work through
  `/gripper_position_control_host` even when idle status includes `64`.

## EEF Control Notes

- `/end_effector_pose` is feedback, not a command. It is published by
  `eepose_pub_node` from `/joint_states_host` using FK.
- `/a1_ee_target` is the EE command topic. It uses
  `geometry_msgs/PoseStamped`.
- The observed feedback frame is `base_link`; the tracker accepts targets in
  `world`. The launch publishes an identity static transform from `world` to
  `base_link`, so those frames are effectively aligned in the current setup.
- The official `eeTracker_demo_node` is MPC/IK-style tracking, not a linear
  Cartesian servo. Small `+1 cm` targets can look like no motion; `+3 cm`
  targets produced visible movement.
- Verified direct-debug responses near the current workspace:
  - `z +3 cm` produced about `+1.5 cm` actual z motion.
  - `y +3 cm` produced about `+1.5-1.8 cm` actual y motion with some x/z
    coupling.
  - `x +3 cm` produced about `+1.0 cm` actual x motion with y/z coupling.
- Expect coupling and under-tracking. Do not assume a published EEF target is
  reached exactly.

## EEF Policy Guidance

- A fully EEF-based closed-loop policy is feasible in this repo because
  `/end_effector_pose` feedback and `/a1_ee_target` control are both working.
- Prefer closed-loop EEF servoing over open-loop EEF playback:

  ```text
    camera + current EEF
    -> policy predicts dx, dy, dz, gripper
    -> apply explicit runtime policy
    -> publish /a1_ee_target
    -> read /end_effector_pose
    -> repeat from actual feedback
  ```

- Start with translation-only EEF control and hold the current orientation.
- Use `/gripper_position_control_host` for the gripper; do not send gripper
  actions through the EE tracker.
- Suggested initial policy loop: 2-5 Hz decisions, while holding each target at
  20-30 Hz. Start with `2-3 cm` max EEF deltas and clamp the workspace tightly
  before expanding.

## Common Commands

```bash
just check
just cameras
just reset
just eef-test
just teleop-test
just teleop pick_cube
just lingbot
just act
just convert banana_in_the_plate
just stop
tmux attach -t lingbot-a1
tmux attach -t act-a1
```

Shared physical hardware is configured in `configs/system/a1.toml`. Teleop
leader mapping and collection semantics are configured in
`configs/teleop/a1_so100.toml`.
The default teleop config is RGB-only and binds both RealSense cameras by
serial: D455 agent view `341522300456` and D405 wrist `218622276998`.
Depth capture remains supported, but enable it intentionally in the tracked
system config after the agent RealSense is on USB3 or after lowering FPS/resolution
for USB2. The shared read-only LAN web preview is configured under
`[web_preview]` and must reuse the owning app's camera readers rather than
opening the same RealSense from a second process.

The A1 + SO leader reset pose is configured in
`configs/poses/a1_so100_collection_start.toml`. `just reset` moves A1 to that tracked joint pose
through the staged jointTracker and relay path while moving the SO leader to
its tracked Feetech position, explicitly closes both grippers, disables leader
torque, and stops the runtime. Successful teleop saves reuse the same concurrent
reset implementation before the next episode when
`collection.auto_reset_after_save` is enabled. Update and commit that file when
the operator intentionally changes the collection start pose or reset speed.

LingBot inference semantics are configured in
`configs/deployments/lingbot_va.toml`. Edit and commit that file when the
server, prompt, execution cadence, or model gripper mapping changes.

ACT joint inference semantics are configured in
`configs/deployments/act_joint.toml`. It starts dry-run and step-gated by
default. Edit and commit that file when the checkpoint, execution cadence, or
model gripper mapping changes.

Useful direct-debug checks inside the ROS/Docker environment:

```bash
rostopic echo -n1 /end_effector_pose
rostopic echo -n1 /joint_states_host
rostopic echo -n1 /arm_status_host
rostopic info /a1_ee_target
rostopic info /arm_joint_command_host
```

Open/close gripper:

```bash
rostopic pub /gripper_position_control_host signal_arm/gripper_position_control \
  "{header: {stamp: now}, gripper_stroke: 200.0}"
rostopic pub /gripper_position_control_host signal_arm/gripper_position_control \
  "{header: {stamp: now}, gripper_stroke: 0.0}"
```

## Engineering Rules

- Respect the dirty worktree; do not revert user changes.
- Prefer `rg` for searching.
- Use `apply_patch` for manual edits.
- Add tests for pure safety logic when possible. Hardware behavior should also
  have a dry/static check that can run while the arm is powered off.
- Keep hardware-touching shell entrypoints boring and explicit. They should
  print the tracked config they are using and fail closed when required devices
  are missing.
- Keep app-level scripts out of `scripts/runtime/`. Runtime scripts own ROS,
  the A1 driver, staged trackers, and relays; app scripts own leader/model/camera
  loops.
- Preserve old working teleop semantics unless a change is intentional and
  visible in config, docs, and tests.
- Do not add new hidden clamps, scaling, or policy-output rewrites. If a limit
  is needed, make it explicit in a tracked config or named safety module.
- Normal data collection should write enough metadata to reproduce the run:
  config path, state/action topics, control path, state/action names, FPS, and
  camera settings. Optional RealSense depth is recorded as raw aligned 16-bit
  PNG in `cam0_depth/` and converts to LeRobot as
  `observation.images.front_depth` when enabled. If cameras stop producing
  fresh samples during collection, fail the episode and delete the partial
  folder rather than saving stale frames.
- Enter=save is a validation boundary. Reject and delete an episode when a
  joint action step exceeds the tracked
  `collection.max_joint_action_step_rad`; print the exact frame and joint,
  reuse the episode index, and reset both devices before retrying.
- `third_party/lerobot` is vendored for the LeRobot v0.6 runtime baseline. Do
  not patch it for A1-specific app behavior; put A1 integration code under
  `galaxea_a1_runtime/` or `scripts/apps/`. The A1 SO leader motor layout lives
  in `galaxea_a1_runtime.teleop.a1_so_leader`.
