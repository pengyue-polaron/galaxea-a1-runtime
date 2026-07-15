# Architecture

This document owns the repository's design, configuration graph, data
contracts, and artifact layout. Live-control invariants are defined in
[Safety](SAFETY.md); operator procedures are defined in the
[Runbook](RUNBOOK.md).

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

- `scripts/runtime/` owns app-agnostic ROS, driver, staged tracker, relay, and
  process lifecycle.
- `scripts/apps/` contains thin operator entrypoints.
- `galaxea_a1_runtime/apps/` implements Teleop, ACT, and LingBot orchestration.
- `runtime/` and `hardware/` adapt pure decisions to ROS, RealSense, serial, and
  process APIs.
- `configuration/`, schema, safety, and collection modules remain hardware-free.
- `lerobot/` owns current raw conversion and deterministic derived packages.
- `third_party/` contains pinned vendor snapshots, not A1-specific behavior.

Heavy dependencies are loaded only at hardware or model boundaries. Static
configuration validation and pure tests do not require ROS, cameras, serial
devices, Torch, or a model checkout.

## Configuration graph

`configs/system/a1.toml` is the physical root. Other tracked configs reference
it instead of copying its values:

```text
configs/system/a1.toml
  ├── configs/teleop/a1_so100.toml
  │     ├── configs/poses/a1_so100_collection_start.toml
  │     └── configs/datasets/<experiment>.toml
  ├── configs/deployments/act_joint.toml
  └── configs/deployments/lingbot_va.toml
```

Ownership is exclusive:

| Config | Owns |
| --- | --- |
| System | devices, ROS topics, cameras, physical limits, relay and startup safety |
| Teleop | leader identity/mapping and collection behavior |
| Pose | reset targets and reset motion behavior |
| Dataset | source/output packaging and conversion policy |
| Deployment | model references and inference/execution behavior |

Schemas require all behavior-affecting keys and reject unknown ones. Python
apps load typed owners directly; shell exports contain only values needed for
process lifecycle. No app-specific config may mirror a physical value from the
System config.

## Runtime composition

Every managed motion path has four roles: an app publishes a high-level target,
an isolated tracker produces a staged driver command, the relay validates it,
and the A1 driver owns the hardware. Exact topics and relay gates are defined in
[Safety](SAFETY.md).

The relay starts locked. An app enables it only after its own inputs and the
shared runtime are ready. Repository-owned Docker containers and tmux sessions
are marked so emergency cleanup can stop them without touching unrelated user
processes.

Each physical resource has one owner. An app that owns cameras shares its open
readers with collection, inference, and embedded preview; the standalone
preview is used only when no app owns those cameras.

## Teleop and observation contract

The first-party `A1SOLeader` exposes six arm axes, `joint0..joint5`, plus an
independent `gripper`. Leader joint motion is mapped relative to both startup
poses using the tracked signs and limits; unknown layouts fail instead of being
sorted heuristically.

The default collection contract contains:

- configured AgentView and wrist RGB observations, plus optional aligned depth;
- EEF pose, six named A1 joints, and continuous gripper state;
- six absolute joint targets and continuous gripper action;
- camera sequence numbers and monotonic sample times;
- configuration, topic, camera, and control-path metadata.

Application gripper state and action are continuous normalized `0..1`. The
leader input maps to that interval, which maps exactly once to the System-owned
physical A1 stroke. `/gripper_stroke_host` is the only gripper feedback source.

An owning app applies the configured camera crop before recording or policy
input. Web preview may show the full image with that crop outlined, but the
overlay is never stored. A valid observation requires fresh frames whose
monotonic-time skew remains within the System limit.

## Episode and dataset commit

Formal collection writes only `galaxea_a1_teleop_raw_v3`:

```text
episode_NNN_timestamp/
  metadata.json
  frames.csv
  cam0/
  cam1/
  cam0_depth/       optional
```

Recording occurs in a hidden sibling staging directory. Save validates vector
dimensions, names, finite values, joint-action continuity, metadata, frame
counts, and exact image sets before an atomic rename exposes the episode.
Rejected saves reuse the episode index; undeletable crash leftovers block the
next run for inspection.

Conversion derives its expected state, action, and camera contract from the
referenced Teleop and System configs:

```text
raw v3
  -> base LeRobotDataset v3
       ├── base LeRobotDataset v2.1
       ├── LingBot EEF continuous v3
       │     └── LingBot EEF continuous v2.1
       └── ACT joint continuous v3
```

Each output and archive is built beside its destination and installed
atomically. Failure preserves the previous complete output. The base A1
joint/state dataset is emitted in both LeRobotDataset v3.0 and episode-based
v2.1; the first-party v2.1 exporter is checked against LeRobot's official
v2.1-to-v3.0 migrator. Both v2.1 packages are derived outputs, never accepted as
collector input.

## Deployment

ACT predicts joint targets through the staged joint runtime. LingBot predicts
EEF targets through the staged EEF runtime. Both reuse the System camera,
gripper, topic, and safety contracts and refuse startup until their registered
checkpoint is explicitly marked ready. Execution remains independently
step-gated in each deployment config.

This checkout does not train models. Reviewed weights produced or downloaded
elsewhere are registered through the local model registry described in
[`models/README.md`](../models/README.md).

## Artifact roots

| Root | Contents |
| --- | --- |
| `data/` | raw episodes, processed datasets, exports, and quarantined legacy data |
| `outputs/` | durable diagnostics, logs, evaluations, and run results |
| `models/` | deployment references and generated runtime assemblies |
| `external/` | machine-local external source checkouts |
| `.cache/` | reproducible disposable caches only |
| `/tmp` | PID files, sockets, and process-lifecycle state |

There is no local training-output root. First-party code must not create
`train_out/`, `outputs/train/`, `artifacts/`, `video_exports/`, or nested
`scripts/**/outputs/` directories.

## Deliberate limits

- No standard MoveIt `move_group` path is provided.
- Old raw-data migration is not maintained.
- No deployment is enabled until its checkpoint contract is registered and
  reviewed.
