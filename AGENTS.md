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
- Keep `scripts/apps/**` as thin operator/runtime entrypoints. Stateful app
  implementations belong under `galaxea_a1_runtime/apps/<app>/`; for example,
  LingBot protocol, CLI, and bridge code live under
  `galaxea_a1_runtime/apps/lingbot/`, shared policy-camera ownership lives in
  `galaxea_a1_runtime/apps/policy_camera.py`, while
  `scripts/apps/lingbot/lingbot_va_ee_bridge.py` only dispatches to that package.
- Teleop collection, inference, and data processing should
  use shared runtime/doctor concepts but should not depend on LingBot being
  installed or running.

## Model Storage

- Inference configs must reference deployment weights through the ignored local
  registry under `models/`; do not point tracked configs directly at a user's
  home directory or a native training output.
- Machine-local external source checkouts belong under the ignored `external/`
  directory; tracked deployment configs must use repo-relative paths there.
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
- Normal gripper apps publish `/a1_gripper_target`, never the host gripper
  topic. The same relay validates and forwards fresh targets to
  `/gripper_position_control_host` only while `ACTIVE` and gripper status is
  healthy (`0` or idle `64`).

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
    staged_command_topic:=/arm_joint_command_host \
    joint_states_topic:=/joint_states_host \
    target_topic:=/a1_ee_target \
    ee_pose_topic:=/end_effector_pose \
    tracker_node:=/eeTracker_demo_node
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
- The tracked compatibility exception is gripper bit 3 (`Position Jump`, mask
  `8`): the relay may ignore that bit only when explicitly configured by
  `relay.gripper_ignored_error_mask`. All other additional gripper bits remain
  faults.
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
    -> policy predicts an explicit absolute/relative EEF action plus gripper
    -> apply explicit runtime policy
    -> publish /a1_ee_target
    -> read /end_effector_pose
    -> repeat from actual feedback
  ```

- Start with translation-only EEF control and hold the current orientation.
- Use `/a1_gripper_target` for normal gripper control; do not send gripper
  actions through the EE tracker or bypass the relay.
- Suggested initial policy loop: 2-5 Hz decisions, while holding each target at
  20-30 Hz. Start with `2-3 cm` max EEF deltas and clamp the workspace tightly
  before expanding.

## Common Commands

```bash
just check
just hardware
just cameras
just camera-web
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
system config and add its dimensions/alignment mode after the agent RealSense
is on USB3 or after lowering FPS/resolution for USB2. The shared read-only LAN
web preview is configured under
`[web_preview]` and must reuse the owning app's camera readers rather than
opening the same RealSense from a second process.

The A1 + SO leader reset pose is configured in
`configs/poses/a1_so100_collection_start.toml`. `just reset` moves A1 to that tracked joint pose
through the staged jointTracker and relay path while moving the SO leader to
its tracked Feetech position, explicitly closes both grippers, disables leader
torque, and stops the runtime. Successful teleop saves reuse the same concurrent
reset implementation before the next episode when
`collection.auto_reset_after_save` is enabled. User discards and quality-check
rejections reuse that reset before retry when
`collection.auto_reset_after_discard` is enabled. Update and commit that file when
the operator intentionally changes the collection start pose or reset speed.
The pose file owns only targets and reset motion behavior; leader identity and
mapping are injected by the owning Teleop config, while physical topics,
limits, names, and relay timings derive from that config's System reference.

LingBot inference semantics are configured in
`configs/deployments/lingbot_va.toml`. Edit and commit that file when the
server, prompt, checkpoint statistics, or execution cadence changes. Physical
gripper mapping remains owned by `configs/system/a1.toml`.

ACT joint inference semantics are configured in
`configs/deployments/act_joint.toml`. It starts dry-run and step-gated by
default. Edit and commit that file when the checkpoint, execution cadence, or
model action semantics change. Physical gripper mapping remains owned by
`configs/system/a1.toml`.

Useful direct-debug checks inside the ROS/Docker environment:

```bash
rostopic echo -n1 /end_effector_pose
rostopic echo -n1 /joint_states_host
rostopic echo -n1 /arm_status_host
rostopic info /a1_ee_target
rostopic info /arm_joint_command_host
```

Explicit direct-debug open/close gripper commands (run only after `just stop`):

```bash
rostopic pub /gripper_position_control_host signal_arm/gripper_position_control \
  "{header: {stamp: now}, gripper_stroke: 104.0}"
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
  the A1 driver, staged trackers, and relays. Thin scripts under `scripts/apps/`
  own operator process lifecycle; reusable leader/model/camera/recording logic
  belongs under `galaxea_a1_runtime/apps/` or another focused package module.
- Reuse `galaxea_a1_runtime.runtime.ros1_env.configure_ros1_python` before
  importing `rospy` or A1 message types; do not duplicate ROS1/SDK path surgery
  in each app.
- Preserve old working teleop semantics unless a change is intentional and
  visible in config, docs, and tests.
- Do not add new hidden clamps, scaling, or policy-output rewrites. If a limit
  is needed, make it explicit in a tracked config or named safety module.
- First-party ROS launch files must require their topic arguments. Runtime
  entrypoints render those values from the system config; launch-file fallback
  topics would create a second, unsafe source of truth.
- Normal data collection should write enough metadata to reproduce the run:
  config path, state/action topics, control path, state/action names, FPS, and
  camera settings. Optional RealSense depth is recorded as raw aligned 16-bit
  PNG in `cam0_depth/` and converts to LeRobot as
  `observation.images.front_depth` when enabled. If a required camera, joint,
  EEF, action, or gripper stream stops producing fresh samples during
  collection, fail the episode and delete the partial folder rather than
  saving cached old values.
- Formal collection and conversion support only the current
  `galaxea_a1_teleop_raw_v3` input contract. Do not add old-raw migration,
  schema fallbacks, or inferred compatibility unless the user explicitly asks
  to recover old data. An unsupported raw schema must fail before creating or
  replacing a processed dataset.
- `just convert <experiment>` is the complete tracked pipeline: current raw v3
  to base LeRobot v3, then LingBot EEF continuous v3, its deliberate v2.1
  compatibility export, and joint continuous v3. The expected raw
  state/action/camera contract derives from the referenced Teleop config and
  its System reference; do not duplicate state mode, physical dimensions, or
  camera schema in the dataset TOML.
- Dataset converters must build into sibling staging paths. An overwrite may
  replace an existing dataset or archive only after the new output is complete;
  failures must preserve the previous complete output.
- Raw collection must also write into a hidden sibling staging directory. Only
  atomically install the final `episode_*` directory after frames, metadata,
  quality checks, and expected file counts pass. Crash leftovers must block the
  next collection run until an operator inspects and removes or quarantines
  them; never ignore deletion failures.
- Decode every named ROS joint vector against `joint_safety.names`, reject
  duplicate/missing names and non-finite values, and reorder explicitly. A
  positional fallback is allowed only for a truly unnamed compatibility
  message; command messages must carry names.
- Record both camera sequence numbers and monotonic sample times. Reject a
  collection or policy observation when either frame is stale or their skew
  exceeds `cameras.max_pair_skew_s`.
- Enter=save is a validation boundary. Reject and delete an episode when a
  joint action step exceeds the tracked
  `collection.max_joint_action_step_rad`; print the exact frame and joint,
  reuse the episode index, and reset both devices before retrying.
- `third_party/lerobot` is vendored for the LeRobot v0.6 runtime baseline. Do
  not patch it for A1-specific app behavior; put A1 integration code under
  `galaxea_a1_runtime/` or `scripts/apps/`. The A1 SO leader motor layout lives
  in `galaxea_a1_runtime.teleop.a1_so_leader`.

### Configuration Ownership And Propagation

- Every runtime value must have exactly one tracked owner:
  - `configs/system/` owns physical hardware, ROS topics, camera devices and
    acquisition settings, physical gripper stroke, workspace limits, and relay
    safety settings.
  - `configs/teleop/` owns leader identity and mapping plus collection
    semantics. It references a system config; it must not restate system-owned
    values.
  - `configs/deployments/` owns model locations and inference/execution
    semantics. It references a system config; it must not restate hardware
    limits or camera acquisition settings.
  - `configs/poses/` owns target pose values and reset motion behavior only. It
    is referenced by the relevant app config, whose typed config is injected
    into the pose loader; never copy device identity, topic, or mapping fields
    into the pose file.
  - `configs/datasets/` owns source/output dataset packaging and conversion
    semantics only. It references the Teleop contract used to collect the raw
    data, which already references the System contract, instead of restating
    state mode or observation shapes.
- Do not mirror fields from a typed owner config into another app-specific
  dataclass merely for convenience. Pass the owning typed config through, or
  derive a smaller runtime object in one pure, named mapping function with an
  exhaustive unit test.
- Every tracked TOML key must be loaded, validated, and consumed. An unused,
  ignored, or silently defaulted tracked key is a defect; remove it or wire it
  through in the same change.
- Tracked config schemas are strict. Loaders must reject unknown tables and
  unknown keys so misspellings fail closed. Do not use `.get(key, default)` for
  live hardware, safety, camera, collection, or deployment behavior that the
  tracked config is expected to own; require the key and keep the value
  explicit in TOML.
- Defaults belong in one place. Prefer required tracked values for behavior
  that affects hardware, datasets, or model compatibility. Library fallbacks
  must not disagree with tracked config defaults.
- Do not store the same semantic value in two config sections and then validate
  that they are equal. Keep it once at the owning layer and pass the typed value
  to every consumer.
- Shell-export helpers must compose the canonical system export instead of
  independently serializing the same fields for Teleop, ACT, and LingBot.
- Shell-export helpers are narrow process-lifecycle APIs, not alternate config
  objects. Export only variables consumed by the corresponding shell script;
  Python/ROS processes must load the typed config directly.
- Normal app entrypoints may accept a tracked config path and lifecycle command
  only. Do not add CLI flags or environment overrides for camera parameters,
  safety limits, action mapping, model execution behavior, or collection
  semantics. Diagnostic output flags and offline data-conversion arguments are
  allowed when they do not alter live control behavior.

### Operator CLI And Terminal Semantics

- Python entrypoints must use `galaxea_a1_runtime.console.ArgumentParser` and
  the shared `info`/`step`/`success`/`warning`/`failure` helpers. Shell
  entrypoints must source `scripts/runtime/a1_console.sh`; do not add local ANSI
  constants, ad hoc `[app]` status prefixes, or a second color vocabulary.
- Semantic labels are fixed: blue `[INFO]` for context, cyan `[STEP]` for an
  in-progress operator action, green `[PASS]` for completed work, yellow
  `[WARN]` for recoverable attention, and red `[FAIL]` for a failed command.
  Cleanup notices use yellow and must not imply successful shutdown.
- Color only interactive terminal output. Honor `NO_COLOR`, suppress ANSI when
  stdout/stderr is redirected, and keep JSON, shell assignments, CSV, and
  `key=value` diagnostic output machine-readable.
- Keep lifecycle CLIs verb-oriented and small: one entrypoint plus
  `start|stop|status|logs|doctor`-style subcommands. Do not create a second
  top-level alias for a subcommand such as `foo-stop`; use `foo stop`.
- Shared process supervision belongs in `a1_tmux.sh` or `a1_services.sh`.
  App scripts provide session names, commands, and readiness/exit markers;
  the shared tmux observation grace comes from
  `configs/system/a1.toml [startup]`, and behavior-specific timeouts remain in
  the owning tracked config. They must not copy sleep-then-grep startup loops
  or add fallback timeout constants to the shared library.
- Unknown commands and invalid arguments print a concise colored usage/error
  and exit 2. Help and dry/static diagnostics must not open hardware or start
  ROS, Docker, tmux, cameras, or serial devices.

### Hardware And Side-Effect Boundaries

- Parse and fully validate tracked configuration before `rospy.init_node`,
  opening cameras or serial ports, creating Docker/tmux processes, or
  publishing any ROS message. Configuration errors must fail without touching
  hardware or starting partial runtimes.
- Emergency shutdown must not depend exclusively on successfully parsing app
  configuration. Mark every repository-owned Docker container and tmux session,
  and keep a configuration-independent fallback that stops only those marked
  resources. Never broaden that fallback to unrelated user containers or tmux
  sessions.
- A hardware family must have one config-driven construction path. Collection,
  inference, diagnostics, and standalone web preview must use the same camera
  factory and pass every applicable setting, including backend, serial/device,
  resolution, FPS, USB requirement, exposure, gain, white balance, pixel
  format, crop, depth, warmup, and freshness limits.
- A standalone system-level service, such as camera diagnostics or camera web
  preview, must load `configs/system/` directly. It must not depend on a Teleop
  or model deployment config solely to discover hardware settings.
- Camera diagnostic output, timeout, FPS-probe, and encoding settings belong to
  `[camera_diagnostics]` in the system config. Do not reintroduce per-run flags
  or hidden Python defaults for them.
- Owning apps may share an already-open hardware reader, but two processes must
  not open the same camera, serial bus, tracker, or command publisher
  concurrently. Make ownership and shutdown ordering explicit.
- Keep configuration mapping, validation, clamps, status decoding, and safety
  decisions in ROS-free pure modules. Hardware scripts should only adapt those
  decisions to ROS/Docker/tmux APIs. Every safety-critical config-to-runtime
  mapping needs a hardware-free unit test.
- Configuration and schema modules must not import hardware modules or eagerly
  load OpenCV, NumPy, RealSense, Torch, ROS, or LeRobot. Put pure value objects
  such as ROI and web-preview settings under configuration/schema, then let the
  hardware layer depend on them.
- Host command topic literals may appear only in the system config, explicit
  direct-debug tooling, and documentation. Normal app code and reports must
  read them from `SystemConfig`.

### Module And API Design

- Give one concept one name. Do not define different public classes with the
  same name in separate modules; use names that distinguish schema,
  transformation, limits, and runtime state.
- Split modules by responsibility when they combine configuration parsing,
  process lifecycle, hardware IO, policy logic, serialization, and UI. Avoid
  large `main()`/`run()` functions and nested copies of doctor or lifecycle
  helpers; extract cohesive pure helpers rather than adding forwarding layers.
- Do not create a shared abstraction until at least two real callers have the
  same semantic contract. Once shared, delete the duplicate implementations
  and make the shared path authoritative.
- Prefer typed result dataclasses for records passed across stages. Do not use
  heterogeneous positional tuples or recover semantic fields by slicing a CSV
  row, list tail, or dictionary insertion order.
- Dataset feature keys, state/action names, camera ordering, and protocol channel
  names must come from one schema module. Do not repeat literals such as
  `observation.images.front`, `observation.images.wrist`, or the seven joint
  action names across collectors, policies, and packers.
- Dependency direction is `scripts -> apps -> runtime/hardware/policies ->
  pure configuration/schema/safety`. Core modules must not import app scripts,
  and system/runtime layers must not import Teleop, ACT, or LingBot configs.
- Keep optional heavy dependencies lazy at module boundaries so static doctor,
  config validation, and pure tests do not require cameras, ROS, Torch, or a
  model checkout. Add a top-level dependency only when tracked production code
  imports it directly; transitive-only packages belong to the owning
  dependency.
- Before retaining compatibility code, placeholders, adapters, or an apparently
  unused module, use `rg` to identify a current caller and document the
  supported compatibility contract. Delete dead branches, dead config fields,
  and superseded wrappers instead of preserving them indefinitely.

### Tests, Documentation, And Change Hygiene

- Test behavior and typed contracts, not source layout. Do not add tests that
  merely search Python, shell, TOML, or Markdown text for strings. String
  assertions are appropriate only when the string itself is a public protocol,
  serialized schema, operator-facing safety diagnostic, or command contract.
- Doctors and generated reports validate structure and derive displayed values
  from the loaded config. They must not hardcode the currently selected camera
  crop, stroke range, topic, port, workspace, model cadence, or prompt as a
  second expected value.
- A bug fix must include the smallest regression test at the purest boundary.
  Hardware behavior additionally needs a safe static/dry check; never make CI
  require powered hardware, ROS master, camera access, Docker, or serial ports.
- When a tracked config contract changes, update its loader, validation,
  consumers, metadata, behavioral tests, `AGENTS.md`, and affected docs in the
  same change. Examples must reference current entrypoints and current topic
  paths.
- Run `just check` and `git diff --check` before handing off a change. For
  hardware-adjacent changes, also state explicitly which checks were static and
  which, if any, touched real hardware.
- Keep changes reviewable and reversible. Do not mix repository-wide formatting
  with behavior changes. Separate mechanical formatting, configuration
  migration, safety behavior, and module refactors into intentional commits.
- Respect existing dirty changes. Inspect `git diff` before editing, avoid
  unrelated rewrites, and never delete generated data, datasets, checkpoints,
  or user files unless the user explicitly authorizes that deletion.
