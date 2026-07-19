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
- The pinned `external/embodied-ops` package owns framework-neutral capability,
  manifest, lifecycle, health, and backend-discovery protocols. It has no ROS or
  LeRobot dependency.
- The pinned `external/lerobot-robot-galaxea-a1` and
  `external/lerobot-teleoperator-galaxea-a1-so-leader` packages own only their
  LeRobot adapters. A1-Research registers the `galaxea_a1_runtime` embodied-ops
  backend and remains the ROS/safety/process composition root.
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
  │     │     └── configs/poses/a1_collection_start.toml
  │     └── configs/datasets/<experiment>.toml
  ├── configs/deployments/lingbot/<deployment>.toml
  │     ├── configs/inference/backends/lingbot_va.toml
  │     ├── configs/models/lingbot/<default-model>.toml
  │     └── configs/tasks/<catalog>.toml
  ├── configs/runs/lingbot/<plan>.toml
  │     ├── configs/deployments/lingbot/<deployment>.toml
  │     └── configs/poses/a1_collection_start.toml
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
| Run plan | ordered task ids, repetitions, and the shared tracked A1 reset pose |

A LingBot deployment names a registered default model so its configuration is
complete without command-line input. `--model` may replace only that model
reference with another strict descriptor registered for the same backend; it
never accepts an artifact path, mutable Hub label, or unregistered weights.
Every process in one run receives the resolved full model id, and durable run
metadata records its full source revision. Batch resume and export require that
exact model identity, so results from two checkpoints cannot fill each other's
slots.

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
shared runtime are ready. Repository-owned Docker containers, host process
groups, and tmux sessions are marked so emergency cleanup can stop them without
touching unrelated user processes. LingBot runs its bridge in the invoking
terminal and uses a marked host process group only for its background policy
server.

Each physical resource has one owner. A marked persistent Camera Bridge owns
both cameras and the read-only Web endpoint for its complete lifetime. It reads
each physical device once and publishes exact raw BGR/depth pairs, source
sequence numbers, and source monotonic timestamps over a per-user local socket.
Inference and collection attach as raw consumers; they never reopen a device or
take over the HTTP port. A separate latest-frame branch encodes the minimal Web
preview at its configured lower rate. Slow browsers or JPEG encoding may drop
preview frames but cannot queue work in, rewrite, or block the raw observation
contract. Web JPEGs are never fed back into policy, recording, or collection.

The generic Operator Panel is a separate localhost-only control plane. The
repository-independent `operator_panel/` package owns HTTP, static rendering,
create-only configuration staging, subprocess supervision, and a small child
input-readiness protocol. It has no Galaxea, ROS, camera, model, topic, or
tracked-config imports and may move to a standalone repository or Git submodule
without moving A1 behavior. The A1 adapter under
`galaxea_a1_runtime/apps/operator_panel/` discovers and fully loads repository
Teleop, LingBot deployment, Batch, model, and reset files, supplies the dynamic
form catalog, and constructs argv-only commands for the existing entrypoints.

One exclusive subprocess owner runs a workflow. Interactive buttons remain
locked until the child explicitly announces its next accepted input set; one
decision consumes that announcement, preventing Web clicks from being queued
through a later safety gate. Configuration creation offers an existing same-kind
template, writes the edited candidate to hidden sibling staging, runs the owning
strict loader, and publishes the new file atomically without overwrite. It is
prohibited while a workflow is active. The page embeds Camera Web MJPEG streams,
while Camera Web remains a read-only service with no control routes.

LingBot shares its bridged raw AgentView reader with an asynchronous H.264 run
recorder; neither component opens another camera handle. One run identity owns hidden
video and log staging paths before model or hardware startup. The recorder
publishes a complete MP4 by atomic directory rename, after which the lifecycle
finalizer adds the scene note, prompt/configuration/Git metadata and that run's
foreground and policy-server logs. The MP4 filename is a portable composition
of scene note, exact input prompt, and start date. A startup failure still
publishes its metadata and logs without exposing an incomplete MP4. A typed IK
or workspace target rejection records `safety_stopped`; batch resume validates
the complete artifact set, exact scene/plan slot, and durable operator
count/discard decision before treating it as finished.

The LingBot batch exporter derives only from those finalized recording roots.
It requires one unambiguous valid run for every tracked plan slot, then writes a
manifest and the selected videos/metadata/logs to a hidden tar staging path
under `outputs/exports/lingbot/` before atomically publishing the completed tar.
Discarded, undecided, incomplete, and duplicate-valid results cannot enter an
export implicitly.

## Teleop and observation contract

The first-party `GalaxeaA1SOLeader` plugin exposes six arm axes,
`joint0..joint5`, plus an independent `gripper`. Leader joint motion is mapped
relative to both startup poses using the tracked signs and limits; unknown
layouts fail instead of being sorted heuristically. The plugin reports truthful
leader units; pair-specific degree-to-radian, gripper, sign, scale, bias, and
limit mapping is a LeRobot processor derived from this repository's tracked
Teleop and System configs.

The default collection contract contains:

- configured AgentView and wrist RGB observations, plus optional aligned depth;
- EEF pose, six named A1 joints, and continuous gripper state;
- six absolute joint targets and continuous gripper action;
- camera sequence numbers and monotonic sample times;
- configuration, topic, camera, and control-path metadata.

Application gripper state and action are continuous normalized `0..1`. The
leader input maps to that interval, which maps exactly once to the System-owned
physical A1 stroke. The System-owned normalized endpoint tolerance absorbs only
the documented LingBot quantile roundoff before this mapping; material
overshoot remains invalid. `/gripper_stroke_host` is the only gripper feedback
source.

The raw consumer applies the configured camera crop before recording or policy
input. The minimal Web preview shows both full unoverlaid images. A valid
observation requires fresh frames whose monotonic-time skew remains within the
System limit.

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

LingBot training summaries normally bind the training code repository and full
revision. For older published artifacts that omit both fields, setup and verify
accept the artifact only when its embedded inference configuration is
byte-identical to the configuration in the pinned backend checkout. This proves
inference compatibility, not the missing training-code provenance, and is
reported explicitly during validation.

At connection time both LingBot and pi0.5 bridges validate a canonical digest
covering code, model, task catalog, camera, state/action, normalization, and
engine contracts before accepting actions. Their shared pure EEF adapter owns episode-relative
pose composition, gripper conversion, review, and explicit bounds validation.
Their shared ROS-free execution coordinator enforces a staged current-joint hold
before relay enable, gripper publication only after `ACTIVE`, and fail-closed
cleanup; model bridges only supply rollout
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
| `external/` | three pinned first-party SDK/plugin submodules plus ignored machine-local external checkouts |
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
