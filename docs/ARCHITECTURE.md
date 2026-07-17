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
- `galaxea_a1_runtime/apps/` implements Teleop, LingBot, and OpenPI
  orchestration. Shared EEF-policy state and transforms live directly under
  `apps/`; model-specific packages remain adapters.
- `models/` owns backend-independent artifact identity, provenance, manifests,
  download, and validation. `inference/` owns the shared wire protocol.
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
  ├── configs/deployments/lingbot/<deployment>.toml
  │     ├── configs/inference/backends/lingbot_va.toml
  │     ├── configs/models/lingbot/<model>.toml
  │     └── configs/tasks/<catalog>.toml
  └── configs/deployments/pi05/<deployment>.toml
        ├── configs/inference/backends/openpi_pi05.toml
        ├── configs/models/pi05/<model>.toml
        └── configs/tasks/<catalog>.toml
```

Ownership is exclusive:

| Config | Owns |
| --- | --- |
| System | devices, ROS topics, cameras, physical limits, relay and startup safety |
| Teleop | leader identity/mapping and collection behavior |
| Pose | reset targets and reset motion behavior |
| Dataset | source/output packaging and conversion policy |
| Inference backend | pinned code checkout, dependency lock, and engine behavior |
| Model descriptor and contract | immutable weight revision, full content manifest, and weight-specific tensor/action semantics |
| Task catalog | approved exact prompt strings, stable operator-facing task ids, and train/OOD provenance |
| Deployment | backend/model/task-catalog references, service lifecycle, execution, and run recording behavior |

Schemas require all behavior-affecting keys and reject unknown ones. Python
apps load typed owners directly; shell exports contain only values needed for
process lifecycle. No app-specific config may mirror a physical value from the
System config.

## Runtime composition

Every managed motion path has four roles: an app publishes a named joint target,
the isolated jointTracker produces a staged driver command, the relay validates
it, and the A1 driver owns the hardware. EEF-policy apps first solve their
Cartesian target through the pure, bounded, System-configured URDF IK adapter.
Exact topics and relay gates are defined in [Safety](SAFETY.md).

The relay starts locked. An app enables it only after its own inputs and the
shared runtime are ready. Repository-owned Docker containers and tmux sessions
are marked so emergency cleanup can stop them without touching unrelated user
processes.

Each physical resource has one owner. An app that owns cameras shares its open
readers with collection, inference, and embedded preview; the standalone
preview is used only when no app owns those cameras.

LingBot also shares its owned AgentView reader with an asynchronous H.264 run
recorder; it never opens a second camera handle. Recording writes into a hidden
staging directory and publishes the finalized MP4 plus metadata by atomic rename
under `outputs/` during normal, interrupted, and fail-closed bridge cleanup.

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
  └── validated boundary trim [start, end)
      ├── Joint LeRobotDataset v3
      ├── Joint LeRobotDataset v2.1
      ├── EEF LeRobotDataset v3
      └── EEF LeRobotDataset v2.1
```

The boundary trim removes only contiguous stationary prefixes and suffixes.
It detects departure from stable endpoint medians using the six named joint
targets and continuous gripper, requires stable final action and feedback
anchors before trimming the suffix, and retains configured pre/post rolls. It
never removes interior pauses. Ambiguous endpoints, excessive trimming, or a
too-short result preserve the complete episode. Raw v3 remains immutable, and
every processed package records the shared source-frame bounds and decisions in
`meta/trim.json`.

Published metadata is machine-independent: provenance uses logical dataset IDs,
and external assets use portable names plus content hashes, never host absolute
paths.

One logical processed dataset references a Raw-package config that owns one or
more Raw v3 task roots. All roots must share the same state, action, camera, and
FPS contract, while each episode retains its source root's task text.

Each output and archive is built beside its destination and installed
atomically. Failure preserves the previous complete output. All four outputs
derive independently from the same validated and trimmed Raw v3 view; no final
processed dataset is an input to another. The converter may create disposable
LeRobot v3 staging data while writing v2.1, but those intermediates are removed
automatically and never become a user-managed source of truth.

Joint and EEF are model-agnostic action representations, each emitted in
LeRobotDataset v3.0 and episode-based v2.1. Model-specific channel selection,
normalization rules, and checkpoint assumptions belong to deployment or
training configs, not dataset names or manifests. The first-party v2.1 exporter
is checked against LeRobot's official v2.1-to-v3.0 migrator. Both v2.1 packages
are derived outputs, never accepted as collector input.

## Deployment

LingBot and OpenPI pi0.5 predict EEF targets through a shared first-party IK
adapter and the staged joint runtime.
Both reuse the System camera, gripper, topic, and safety contracts and refuse
startup until their deployment is explicitly marked ready. Execution remains
independently owned in each deployment config. The reviewed fruit-placement
deployments are live-enabled. Before any model, ROS, camera, or hardware process
starts, the operator must select one prompt from their shared tracked task
catalog. The selected session then begins inference and executes the configured
actions without additional prompts. Each checkpoint's episode-relative
quaternion is always composed onto the episode origin and preserved through IK;
there is no translation-only orientation mode. Each deployment exclusively owns
its rollout cadence under `[execution]`; cadence is an operator-reviewed runtime
choice, not a System safety setting. LingBot renders compact run progress on one
terminal line. Each deployment has a finite model-call budget sized to cover the
longest episode in the training data; reaching that budget or stopping the
bridge finalizes any owned recording, locks the relay, and tears down the
runtime.

This checkout does not train models. Reviewed weights produced or downloaded
elsewhere are registered through the local model registry described in
[`models/README.md`](../models/README.md).

Managed model inference is a host-side GPU service separated from the ROS
bridge. Configuration is composed from five exclusive owners:

```text
System + inference backend + immutable model + task catalog + deployment
```

The backend pins source code and its dependency lock. The model descriptor pins
one Hugging Face commit, checkpoint step, artifact format, complete file
manifest, and model-specific contract. Its local directory is derived as
`models/artifacts/<model-id>/<revision>/`; no mutable `latest` alias or
hand-maintained weight path exists. Downloads are validated in a hidden sibling
and exposed by atomic rename only after the exact file set, byte sizes, and
SHA-256 digests pass. The task catalog owns the approved runtime prompt set and
explicitly records whether each prompt is from training or OOD evaluation; the
deployment owns only its catalog reference, service lifecycle, and execution
choices. Runtime input selects a tracked task id and cannot introduce an
unregistered prompt.

At connection time both LingBot and pi0.5 bridges validate a canonical digest
covering code, model, task catalog, camera, state/action, normalization, and
engine contracts before accepting actions. Their shared pure EEF adapter owns episode-relative
pose composition, gripper conversion, review, and explicit safety transforms.
Their shared ROS-free execution coordinator enforces a staged current-joint hold
before relay enable, gripper publication only after `ACTIVE`, configured
feedback correction, and fail-closed cleanup; model bridges only supply rollout
behavior. The IK adapter reads the same URDF as the runtime, uses named System
joint limits, and rejects non-convergence or excessive joint deltas. Model
services reuse the app-agnostic tmux health/exit supervisor. Each live bridge
can publish only staged named-joint/gripper targets; the isolated jointTracker
and locked relay remain the sole path toward host motor commands.

## Artifact roots

| Root | Contents |
| --- | --- |
| `data/` | raw episodes, processed datasets, exports, and quarantined legacy data |
| `outputs/` | durable diagnostics, logs, evaluations, and run results |
| `models/` | immutable, content-verified deployment artifacts |
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
