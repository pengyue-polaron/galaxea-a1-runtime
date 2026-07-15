# Architecture

This repository separates physical system policy, reusable runtime logic, app
behavior, and operator lifecycle. Dependency direction points inward toward
typed configuration and pure contracts; live hardware is opened only at the
outer edge.

## Layers

```text
Justfile / scripts
        |
        v
galaxea_a1_runtime.apps
        |
        +----> runtime / hardware / policies
        |                 |
        +-----------------+
                          v
        configuration / schema / safety / collection contracts
```

- `scripts/runtime/` owns ROS, the A1 driver, staged trackers, the relay, and
  shared process lifecycle. It is app-agnostic.
- `scripts/apps/` contains thin operator entrypoints for cameras, Teleop, ACT,
  and LingBot.
- `galaxea_a1_runtime/apps/` implements app orchestration and protocol logic.
- `runtime/` and `hardware/` adapt pure decisions to ROS, RealSense, and process
  APIs.
- `configuration/`, `schema.py`, `safety.py`, and collection contracts remain
  hardware-free and are unit-testable.
- `lerobot/` owns current raw conversion and deterministic derived packages.
- `third_party/` contains pinned vendor snapshots, never A1-specific app logic.

Heavy optional dependencies are lazy. Static config validation and unit tests do
not need ROS, cameras, serial devices, Torch, or a model checkout.

## One Configuration Graph

`configs/system/a1.toml` is the root physical contract. It owns host runtime,
ROS topics, relay settings, joint and EEF safety, gripper stroke, both cameras,
camera diagnostics, and LAN preview.

The remaining configs reference it:

```text
configs/system/a1.toml
  ├── configs/teleop/a1_so100.toml
  │     ├── configs/poses/a1_so100_collection_start.toml
  │     └── configs/datasets/<experiment>.toml
  ├── configs/deployments/act_joint.toml
  └── configs/deployments/lingbot_va.toml
```

Each layer owns only its semantics:

- Teleop: leader identity/mapping and collection behavior.
- Pose: reset targets and reset motion behavior.
- Deployment: model identity and inference/execution behavior.
- Dataset: raw source and output package locations.

Loaders reject missing and unknown keys. Derived runtime objects are built by
named mappings from these typed owners rather than parallel defaults. Shell
export helpers expose only values required for process lifecycle; Python apps
load the same typed config directly.

## Safe Control Plane

EEF target path:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe relay
  -> /arm_joint_command_host
```

Joint target path used by Teleop and ACT:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe relay
  -> /arm_joint_command_host
```

Gripper path:

```text
/a1_gripper_target
  -> safe relay
  -> /gripper_position_control_host
```

The relay starts locked. It validates vector shape, finite values, configured
joint limits, control modes, input freshness, motor status, and initial staged
alignment before forwarding. Code outside the relay does not publish normal
commands to host topics.

The runtime joint action-step guard is explicitly disabled by the tracked
system config. Finite-value and absolute-limit checks remain active. Collection
has a separate post-recording continuity boundary before an episode becomes
durable.

## Teleop Runtime

The normal entrypoint is `just teleop <experiment>`:

1. load and validate Teleop, System, and reset-pose configs;
2. start the isolated joint runtime and locked relay;
3. open the six-axis SO leader and both configured cameras;
4. map leader displacement relative to startup pose into A1 joint targets;
5. require fresh joint, EEF, gripper-feedback, action, and paired-camera data;
6. arm the relay only after readiness checks pass;
7. record episodes into hidden sibling staging directories;
8. validate quality and exact files, then atomically rename a successful save;
9. reset both devices after a successful save or discard when configured.

Unknown leader key layouts, non-finite values, stale samples, camera skew, or a
partial episode fail closed. A rejected episode does not consume its index.

The first-party leader adapter deliberately exposes six arm axes
`joint0..joint5` plus an independent `gripper`; the upstream five-axis naming is
not interchangeable with this physical rig.

## Observation Contract

The System config selects two RealSense cameras by serial. A single owning app
opens them and shares frames with collection/inference and its embedded preview.
A standalone preview is available only when another app is not the owner.

```text
D455 AgentView 640x480
  ├── full frame -> web preview + red ROI overlay
  └── 480x480 ROI -> collection and every policy

D405 wrist 640x480
  └── full frame -> preview, collection, and every policy
```

The red web rectangle is display-only. Optional aligned depth is recorded as
raw 16-bit PNG and converts to `observation.images.front_depth`. Each
observation must contain fresh frames whose monotonic-time skew is within the
system limit.

The gripper contract is continuous: application state and action use normalized
`0..1`, mapped exactly once to the physical `0..104 mm` A1 stroke. The tracked
SO leader input range `0..53.16` maps directly to that interval. Full-open
feedback is about `103.8 mm`, and the only feedback source is
`/gripper_stroke_host`.

## Raw Episode Commit

Current episodes use schema `galaxea_a1_teleop_raw_v3` and contain:

```text
episode_NNN_timestamp/
  metadata.json
  frames.csv
  cam0/                 cropped AgentView RGB
  cam1/                 wrist RGB
  cam0_depth/           optional aligned uint16 depth
```

Metadata records the config path, task, state/action names and topics, control
path, FPS, camera settings, crop, freshness limits, and counts. `frames.csv`
stores continuous state/action vectors plus camera sequence and sample times.

Recording never writes directly to the final episode directory. Enter-to-save
validates state/action dimensions, finite values, joint action continuity,
metadata, frame count, and the exact image set. Only an atomic rename exposes a
complete episode. Stale streams or interrupted writes remove the partial
episode; undeletable crash leftovers block the next run for operator review.

## Dataset Pipeline

`just convert <experiment>` loads `configs/datasets/<experiment>.toml` and runs:

```text
raw v3 episodes
  -> base LeRobotDataset v3
       ├── LingBot EEF continuous v3 + archive
       │     └── LingBot EEF continuous v2.1 + archive
       └── joint continuous v3 + archive
```

The converter supports only current raw v3. It derives expected state/action
and camera shapes from the referenced Teleop config and its System config, so
the dataset config does not duplicate the collection or physical observation
contract. It validates every episode
and requires all episodes to agree on names, cameras, shapes, and FPS.

Every output is built next to its destination and installed atomically. An
existing complete dataset remains intact if generation, encoding, validation,
or archive creation fails. The v2.1 package is an intentional derived LingBot
compatibility export; it is not accepted as raw collector input.

## Deployment Boundaries

ACT and LingBot each have one tracked deployment config. Both reuse the System
camera, gripper, topic, and safety contract.

- ACT predicts joint targets and publishes through the staged joint path.
- LingBot predicts EEF targets and publishes through the staged EEF path.
- Each starts dry-run/step-gated and refuses startup until a new registered
  checkpoint is explicitly marked deployment-ready.
- This checkout does not train models. Weights produced or downloaded elsewhere
  are registered in the ignored `models/` registry before deployment.

`GalaxeaA1Robot` is an injected LeRobot-style adapter and has no implicit live
ROS publisher. Managed live paths remain in the app runtimes so there is one
implementation of each control plane.

## Process Ownership

Runtime resources are repository-marked. Normal app stop paths use their
validated config; the global stop fallback can remove repository-owned
containers and tmux sessions even if config parsing fails. It must never target
unrelated user processes.

The camera web service is read-only HTTP/MJPEG on the LAN. It has no ROS or
motion endpoint, no authentication, and no encryption. It is not a public
Internet service.

## Deliberate Limits

- No standard MoveIt `move_group` path is provided.
- The upstream EEF trajectory demo is not a first-class operator command.
- No old raw-data migration is maintained.
- No deployment is enabled until its newly trained checkpoint contract is
  registered and reviewed.
