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
- `models/` owns A1 model descriptors, contract parsing, registry values, and
  the thin composition of generic artifact and code-environment workflows.
  `inference/` owns the A1 inference transports.
- `runtime/` and `hardware/` adapt pure decisions to ROS, RealSense, serial, and
  process APIs.
- The pinned `external/embodied-ops` package owns the standard CLI presentation,
  collection interaction, reset-point contract, streaming leading-stillness
  trimming, task selection, evaluation progress, strict Operator
  Panel catalog and Web presentation, contract digests, verified external
  artifact/code-environment workflows, sample timing, normalized camera health,
  transactional artifact primitives, and shared LeRobot v3 validation/v2.1
  format conversion behind an optional extra. It defines no robot API and has
  no ROS dependency.
- The pinned `external/lerobot-robot-galaxea-a1` and
  `external/lerobot-teleoperator-galaxea-a1-so-leader` packages own only their
  LeRobot adapters. The Robot plugin depends on the lightweight protocol/client
  distribution released from `packages/galaxea-a1-runtime-protocol`; this
  Runtime owns the server, sessions, leases, watchdog, ROS, safety, and process
  composition.
- `configuration/`, schema, safety, and collection modules remain hardware-free.
- `lerobot/` owns direct LeRobotDataset recording, atomic episode commits, and
  deterministic derived packages, while `embodied-ops` supplies the shared
  LeRobot file-graph and format mechanics.
- `third_party/` contains pinned vendor snapshots, not A1-specific behavior.

Heavy dependencies are loaded only at hardware or model boundaries. Static
configuration validation and pure tests do not require ROS, cameras, serial
devices, Torch, or a model checkout.

## Reusable workflow boundary

`embodied-ops` is deliberately an operator-workflow standard, not an
embodied-hardware abstraction. Its inputs are ordinary paths, identifiers,
timestamps, episode decisions, task IDs, declarative form catalogs, and narrow
adapter protocols. It standardizes CLI levels, task selection, Enter-to-Start/
Save collection behavior, Web layout and form structure, progress, locked
environment setup, complete artifact publication, and shared dataset-format
mechanics. It does not name joints, interpret actions, open devices, define a
robot schema/provenance policy, or execute policies.

This repository supplies the A1 semantics around those primitives: the
canonical observation/action schema, LeRobotDataset policy and provenance,
camera identities, reset behavior, task-catalog values, model contracts, ROS
ownership, and safety gates. Hardware interoperability uses LeRobot's existing `Robot`
and `Teleoperator` plugin contracts. The A1 wire schema and thin client are a
lightweight distribution in this repository; the server is Runtime-only.

## Configuration graph

`configs/system/a1.toml` is the physical root. Other tracked configs reference
it instead of copying its values:

```text
configs/system/a1.toml
  ├── configs/teleop/a1_so100.toml
  │     └── configs/poses/a1_so100_collection_start.toml
  │           └── configs/poses/a1_collection_start.toml
  ├── configs/datasets/<experiment>_derivatives.toml
  ├── configs/deployments/lingbot/<deployment>.toml
  │     ├── configs/inference/backends/lingbot_va.toml
  │     ├── configs/models/lingbot/<default-model>.toml
  │     └── configs/tasks/**/<catalog>/catalog.json
  ├── configs/runs/lingbot/<plan>.toml
  │     ├── configs/deployments/lingbot/<deployment>.toml
  │     └── configs/poses/a1_collection_start.toml
  └── configs/deployments/pi05/<deployment>.toml
        ├── configs/inference/backends/openpi_pi05.toml
        ├── configs/models/pi05/<model>.toml
        └── configs/tasks/**/<catalog>/catalog.json
```

Ownership is exclusive:

| Config | Owns |
| --- | --- |
| System | devices, ROS topics, cameras, physical limits, relay/startup safety, the A1 Robot service contract, and the Operator Panel/Foxglove observability endpoints |
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

A task registry has one `catalog.json` identity and one create-only
`prompts/<task-id>.json` record per prompt. Prompt records own their exact text,
stable id, train/OOD provenance, and deterministic display order. Registration
creates a new record atomically; it never rewrites the catalog or an existing
record. Top-level task catalogs are collection-facing and provide the panel's
training-prompt suggestions. A nested model-bound catalog may expose the strict
prompt subset and provenance of one checkpoint without reclassifying the
collection experiment's catalog; nested catalogs are not offered as collection
prompt suggestions.

## Runtime composition

```text
LeRobot Robot plugin
  -> galaxea-a1-runtime-protocol client
  -> A1-specific gRPC over System-owned Unix socket
  -> Runtime-owned server/session/lease/watchdog
  -> ROS staged tracker -> locked relay -> host driver
```

The service process is the only owner of A1 ROS resources. `Describe` is static;
hardware attachment occurs only when a session opens. The first session attaches
read-only feedback and relay-status subscribers; it does not create command publishers
or require a `LOCKED` relay. One command session owns an opaque lease at a time. Only
that lease may attach the staged-command subscriber, target publishers, motion gate,
and hold timer, and it requires fresh feedback plus a fresh `LOCKED` relay before
attachment. Read-only observation sessions may coexist throughout.

Commands carry a contiguous sequence and monotonic timestamp. Successful commands
refresh the command-inactivity deadline; heartbeats refresh only session liveness.
Command failure, inactivity expiry, lease expiry, or command-session close disables
and releases command resources. Observation resources remain attached while observers
remain connected. Calibration and reset remain explicit Runtime workflows rather than
remote Robot capabilities. A failed command-resource release is process-fatal, so a
possibly stale publisher or timer can never coexist with a new owner. The socket is
published inside a current-user-owned `0700` directory with mode `0600`; existing paths
are never unlinked on startup. The relay's independent input-freshness and motor-status
gates remain authoritative.

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

Foxglove observability is a side branch of that runtime. Its only control path
terminates at the existing Operator Session rather than at ROS command topics:

```text
existing ROS state/status/command topics + Camera Bridge raw consumer
  -> validating A1 telemetry adapter
  -> named JointState mirrors + CompressedImage + DiagnosticArray
  -> scoped foxglove_bridge WebSocket
  -> Foxglove Desktop/Web

Foxglove exact Trigger service
  -> telemetry adapter phase/run/input-revision validation
  -> current-user Unix Operator Session
  -> embodied-ops one-shot input gate
  -> already-supervised collection child stdin
```

The adapter never opens a camera, creates a target or host-command publisher,
enables the relay, or rewrites a command. Invalid command-shaped messages are
reported and omitted from the display mirrors. The bridge receives exact
subscription, asset, and five collection-service regexes derived from System
config. Parameter access, client publication, client-advertised topics, and all
other services use a no-match allowlist. Its capabilities are
`connectionGraph`, `assets`, and `services`; it deliberately omits
`clientPublish`. The current layout therefore has no Publish panel. The
configured URDF and each referenced mesh are the only remotely retrievable
assets.

The committed Foxglove layout is generated from System topic names, joint names,
and the configured URDF, and a test rejects drift. ROS master, telemetry, and
Foxglove are one shared persistent observation stack rather than per-application
sidecars. The public Camera lifecycle ensures that stack, and execution runtimes
reuse it instead of opening a second port or ROS master. Normal execution
shutdown preserves both Camera Bridge and observation stack so monitoring
survives arm power and application transitions; their explicit public stop
closes them. The standalone stack never starts the A1 driver, tracker, or relay.
Joint feedback and TF appear only when an owning execution runtime supplies
them. Camera panels intentionally use compressed images without invented
calibration or frame transforms.

The committed JSON and generated extension config are the cloud-workspace
sources derived from System config. A path-scoped GitHub Actions workflow
validates them, builds and publishes the private organization extension
`Galaxea A1 Collection Console`, then uses the Foxglove API to upsert the
exact-name layout `Galaxea A1 Operations` in folder `Galaxea A1`. A unique
workflow-run extension version makes retries and updates explicit. The API key
exists only as a repository secret, and publication does not connect to a robot
or ROS master. Exact-name lookup must find zero or one matching layout;
duplicates fail closed instead of updating an ambiguous target.

The A1 Runtime gives the generic Operator Panel application one private,
current-user Unix socket. The telemetry adapter polls that socket at the
System-owned rate, validates the generic status schema, removes child command
argv and terminal logs, and publishes a versioned heartbeat summary on
`/a1/ops/workflow_status`. Workflow identity, lifecycle, progress, semantic
phase, current guarded input choices, and failure state are therefore visible
in Foxglove without importing ROS into `embodied-ops`.

The same adapter advertises exactly five `std_srvs/Trigger` services: start,
save, discard, reset, and stop. Start/save/discard/reset are accepted only when
their action id is present in the active `collect` gate and its semantic phase
matches; the follow-up Unix request carries the exact `run_id` and
`input_revision`, so a race or double click is rejected again by the generic
supervisor. Stop is scoped to the exact active collection run. None of these
services publish a target or motor command. Reset merely selects the collector's
existing tracked reset branch, which continues to use the staged tracker and
locked relay contract.

External OpenRAL deployment uses two versioned, private local-service
boundaries. Camera Bridge protocol `describe` exposes its exact digest and raw
frame shapes before streaming. The LingBot OpenRAL policy gateway loads the
authoritative Runtime deployment/model contract, owns episode-relative EEF
transforms, temporal-cache replay, and bounded URDF IK, and returns only six
absolute joint proposals plus one normalized gripper proposal. The gateway has
no ROS imports, hardware handle, or command publisher. OpenRAL supplies its
narrower ordered joint envelope during handshake, then remains the sole
candidate-action safety and HAL execution owner. Neither side imports the
other repository's source tree.

The generic Operator Panel is a separate control plane. The A1 instance binds
its System-owned endpoint on the trusted LAN; it has request-integrity tokens
but no user authentication or transport encryption and must not be exposed
beyond that LAN. The
repository-independent `embodied_ops.operator_panel` package in the pinned
`external/embodied-ops` repository owns the versioned catalog schema and form
builders, HTTP, packaged static rendering, create-only document staging,
subprocess supervision, and a small child presentation protocol for input
readiness and typed progress. Child events and workflow-status snapshots have
independent versioned envelopes; each workflow run has a stable UUID, monotonic
status and input-gate revisions, semantic input phase/detail, explicit lifecycle
state, and UTC start/finish timestamps. Malformed, stale, replayed, cross-run,
or undeclared events are rejected rather than silently changing available input.
Its minimal adapter owns only catalog values and workflow launch contracts;
camera health and repository-maintenance providers are independent optional
capabilities. Progress is display-only, retained by stable id as latest state,
and excluded from durable logs. The generic core has no Galaxea, ROS, camera,
model, topic, or tracked-config imports. The A1 adapter under
`galaxea_a1_runtime/apps/operator_panel/` deliberately enables only camera
health and validated runtime workflows. Repository configuration and Prompt
maintenance use the unified A1 CLI instead of Web mutation endpoints.

The A1-only reset entrypoint is self-contained: its thin app lifecycle wrapper
validates the System and pose before starting the app-agnostic joint runtime,
runs the shared staged reset implementation, and stops its owned ROS services
on every exit path. The Web panel and unified CLI use this same command.

One exclusive subprocess owner runs a workflow. Interactive buttons remain
locked until the child explicitly announces its next accepted input set; one
decision consumes that announcement, preventing Web clicks from being queued
through a later safety gate. The page embeds Camera Web MJPEG streams,
while Camera Web remains a read-only service with no control routes. Per-camera
preview rate, frame age, freshness, and errors are read from Camera Web's existing
health endpoint through the A1 adapter; the generic panel neither opens cameras nor
redefines camera-health thresholds.

Every submitted workflow value is revalidated against its declared form
immediately before reaching the adapter: unknown keys, missing required fields,
wrong JSON types, and unavailable select options fail closed.
The panel launches each workflow through an ownership supervisor. If the panel
process disappears, the supervisor interrupts the workflow process group and
escalates only that owned group after bounded grace periods, preventing an
orphaned hardware workflow from outliving its control plane.

LingBot shares the Camera Bridge's atomic raw AgentView/wrist pair with one
asynchronous H.264 run recorder; neither component opens another camera handle.
One run identity owns hidden video and log staging paths before model or hardware
startup. The recorder encodes both camera streams at one output cadence, records
the source sequence/timestamp/skew for every output frame, and publishes the two
MP4s, timeline, and camera sidecar in one atomic directory rename. The lifecycle
finalizer then adds the scene note, prompt/configuration/Git metadata and that
run's foreground and policy-server logs. Video filenames are portable
compositions of scene note, exact input prompt, start date, and camera name. A
startup or encoder failure still publishes its metadata and logs without
exposing a partial camera pair as complete. A typed IK or workspace target
rejection records `safety_stopped`; batch resume validates both videos, the
timeline/sidecar, exact scene/plan slot, and durable operator count/discard
decision before treating it as finished.

The local LingBot server adapter can reconstruct a bounded cache-aware,
multi-layer WAM diagnostic on an explicit audit request. It captures all 30
self-attention layers at both the trailing video `t=0` cache commit whose
predicted KV is actually consumed by action inference and the final
scheduler-consumed action denoise. It does not use the trailing action
cache-only call as the returned action's source. Each layer performs the
full-key softmax and retains only the current-query transition and selected
visual-key blocks. Residual rollout exposes action-to-predicted-future,
predicted-future-to-each-actual-history latent anchor, and direct
action-to-each-actual-history latent anchor maps. It also composes the first
two rollouts to report an action-via-predicted-future-to-history diagnostic.
The adapter uses the model-owned temporal scale both to align cached actual
latents to raw observation anchors 0/4 and to align its four generated latent
frames to RGB anchors 0/4/8/12 of the 13-frame decode. It undoes the VAE's
front/wrist width packing for both pixels and tokens, and writes every 8×8 map
over its corresponding actual or predicted image. Action maps average only the
queries for frames the runtime will execute. Text cross-attention, MLP/gating
paths, cached-action paths, and cross-layer paths through other unselected
cache groups remain omitted, so these maps are association diagnostics rather
than causal attribution. Normal inference requests do not pay the
reconstruction or VAE decode cost.

The LingBot batch exporter derives only from those finalized recording roots.
It requires one unambiguous valid run for every tracked plan slot, then writes a
manifest and the selected videos/timeline/sidecar/metadata/logs to a hidden tar staging path
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
Teleop and System configs. The processor accumulates shortest-path degree deltas
so a Feetech `4095 <-> 0` encoder crossing remains continuous, then rejects any
processed per-frame joint step beyond the tracked bridge threshold before the
Robot can submit it to the Runtime.

The production Teleop bridge is the pair composition root. It constructs the
auto-discovered Teleoperator and Robot plugins, then applies LeRobot's
teleoperator-action and robot-action processor ordering. The first processed
action is the exact current A1 joint pose and current normalized gripper state,
even when configured bias or the leader gripper differs. Only the second frame
begins relative mapping and rate checks. The Robot sends the hold through the
lightweight protocol client. The A1 Runtime service alone attaches to ROS and
requests the relay. The previous in-app joint mapping and ROS publishers are not
retained as a second control path. Generic LeRobot 0.6 CLI entrypoints use
identity processors and are not valid for this degree-to-radian hardware pair.

The Robot plugin intentionally exposes the control observation required by the
pair (six measured joints and gripper), rather than making joint Teleop depend
on EEF tracking or cameras. Formal collection is a Runtime composition because
it also observes the command actually sent, EEF feedback, and the sole-owner
Camera Bridge. Both paths consume the same canonical feature-name/schema
module; neither duplicates a physical owner or command publisher.

The default collection contract contains:

- configured AgentView and wrist RGB observations, plus optional aligned depth;
- EEF pose, six named A1 joints, and continuous gripper state;
- six absolute joint targets and continuous gripper action;
- camera sequence numbers and monotonic sample times;
- configuration, topic, camera, and control-path metadata.

Application gripper state and action are continuous normalized `0..1`. The
leader input maps to that interval, which maps exactly once to the System-owned
physical A1 stroke. LingBot's model contract owns a gripper-only latent
projection: values inside the tracked training envelope are projected to the
quantile interval before de-normalization, while non-finite or farther-out
outputs are rejected. The System-owned normalized endpoint tolerance then
absorbs only quantile arithmetic roundoff before the physical mapping;
material protocol-level overshoot remains invalid. `/gripper_stroke_host` is
the only gripper feedback source.

The raw consumer applies the configured camera crop before recording or policy
input. The minimal Web preview shows both full unoverlaid images. A valid
observation requires fresh frames whose monotonic-time skew remains within the
System limit.

## Episode and dataset commit

Formal collection writes `galaxea_a1_lerobot_dataset_v3_v3` directly under
`data/datasets/EXPERIMENT/`. The standard LeRobot v3 contract is immediately
usable by LeRobot readers:

```text
observation.state = [EEF xyz+xyzw, joint_1_rad..joint_6_rad,
                     gripper_normalized]
action            = [joint_1_rad..joint_6_rad, gripper_normalized]
observation.images.front
observation.images.wrist
observation.images.front_depth       optional, uint16 millimeters
task                              standard LeRobot per-frame task
```

`meta/galaxea_a1.json` adds the tracked config identity, ordered task list,
topic/control path,
camera sources and crop, feature semantics, freshness limits, and gripper
mapping. It supplements rather than forks LeRobot's `info.json`, tasks, episode
metadata, stats, Parquet, and image/video layout.

Each episode records into a hidden sibling snapshot of the complete dataset.
Existing immutable `data/`, `videos/`, and `images/` payloads are hard-linked;
the sibling transaction fails clearly if its filesystem cannot preserve those
links instead of silently copying the complete dataset. Mutable metadata is
copied. LeRobot resume starts new payload files, finalizes the new episode, then
the runtime validates format, robot type, FPS, the exact feature contract,
experiment/task identities, and frame/episode counts before an atomic directory
replacement. Failure or discard preserves the previous complete dataset.
Rejected saves reuse the episode index, and crash leftovers block the next run
for inspection.

One experiment is one LeRobot dataset and may contain episodes from multiple
exact prompts. Standard `task_index`, `meta/tasks.parquet`, per-frame task, and
per-episode task metadata preserve the mapping. Starting collection again with
the same experiment and a new prompt appends that prompt and its episodes to the
same atomic dataset; it does not require a new experiment identity.

Canonical image storage is always video and is therefore not an operator
configuration option. Before ROS or cameras start, dataset preflight validates
the task table, contiguous episode graph, Parquet row counts and schemas, and
every referenced data/video payload. `just dataset-doctor EXPERIMENT` exposes
the same hardware-free validation explicitly. Shared LeRobot graph traversal,
path safety, Parquet accounting, v2.1 episode layout, and video slicing come
from `embodied-ops[lerobot-dataset]`; this Runtime supplies the A1 feature,
task, provenance, derivation, and publication constraints.

After services, bridge, ROS state, and cameras are ready, the collector invokes
the tracked A1 and leader reset lifecycle before the first episode gate. During
each episode, fresh validated samples pass through the shared bounded streaming
trimmer. A1-owned radian and normalized-gripper thresholds decide when sustained
motion begins; only the configured preroll and subsequent frames enter the
LeRobot transaction. This keeps canonical metadata, statistics, Parquet rows,
and videos consistent without a post-export rewrite.

This canonical dataset intentionally stores the richest model-agnostic A1
observation and the command actually sent by Teleop. A training adapter can
select channels without rewriting the recording. An EEF-action or old-version
package is an explicit derivative because it changes action semantics or file
format; it is not part of collection.

The direct-derivation pipeline accepts only a validated canonical dataset as
its source. It can produce Joint v2.1, EEF v3.0, and EEF v2.1. Each output starts
from the canonical source; Joint v2.1 exports directly, while EEF v2.1 uses a
disposable EEF v3 workspace rather than another final derivative. Source
identity comes from
`meta/galaxea_a1.json`, while output paths, repository IDs, overwrite policy,
and kinematics remain in one tracked Dataset config.
The derivative namespaces the immutable recording provenance as
`meta/source_galaxea_a1.json`; it never leaves a stale direct-dataset manifest
claiming to describe the rewritten action representation.

Raw v3 is not a supported source or intermediate. Historical files under
`data/raw/` may remain locally, but runtime and dataset tooling do not consume
them. No final derivative is used as the source of another final derivative.

Published metadata is machine-independent: provenance uses logical dataset IDs,
and external assets use portable names plus content hashes, never host absolute
paths. Model-specific channel selection, normalization rules, and checkpoint
assumptions remain in deployment or training configs, not dataset names or
manifests.

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
terminal line in a CLI and as typed progress in the Operator Panel. Each
deployment has a finite model-call budget sized to cover the
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

Each supervisor exposes only the repository runtime source and the pinned,
dependency-free `embodied-ops` source used by its typed task-catalog contract to
the isolated backend interpreter; it never leaks the main virtual environment
into a model environment. LingBot uses Python 3.12 and
OpenPI uses its pinned Python 3.11 environment, so the shared runtime import
surface remains Python 3.11-parseable even though the main runtime environment
is formally Python 3.12. Repository-owned ROS containers likewise receive only
the runtime source through their fixed container `PYTHONPATH`. The backend
source path is one rendered lifecycle contract shared by the LingBot and OpenPI
supervisors; model environments do not install or own a second version of
`embodied-ops`.

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
unregistered prompt. CLI registration atomically creates and validates the new
JSON record before it becomes selectable; it cannot alter the prompt bound to a
run after startup.

LingBot training summaries normally bind the training code repository and full
revision. For older published artifacts that omit both fields, setup and verify
accept the artifact only when its embedded inference configuration is
byte-identical to the configuration in the pinned backend checkout. A summary
from an explicitly dirty training worktree must instead retain the matching
repository, a full starting revision, the dirty-worktree marker, and the claim
that its exact training files are included; its embedded inference configuration
must still be byte-identical to the pinned backend. These fallback paths prove
inference compatibility, not a clean reproducible training revision, and are
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
| `data/` | canonical LeRobot datasets, derivatives, exports, and quarantined legacy data |
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
- No Raw v3 migration or collection intermediate is provided.
- No deployment is enabled until its checkpoint contract is registered and
  reviewed.
