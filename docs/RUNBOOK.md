# Runbook

## Diffusion2One / Distill-WAM

### A1 teacher

The teacher is registered independently as
`diffusion2one/a1_fruit_blocks_teacher_eef`, step 1000, from
`SeanZheng/Distill-WAM` revision `d1fa5b95b41fa96408be5c37cfae2ce900d1c615`
(`galaxea-a1-teacher/`). It was imported from Torch
`/scratch/yp2841/models/Distill-WAM/galaxea-a1-teacher/` through Globus; its
registered manifest pins the published release contents. Import receipts belong
under `outputs/model_imports/`.

`configs/inference/backends/diffusion2one_teacher.toml` owns the teacher's
20 video steps, 50 action steps, video guidance 5, and action guidance 1.
It shares the pinned Diffusion2One checkout/environment and registered LingBot
foundation components. Its independent backend identity prevents selection by
the single-step student deployment. Teacher inference is considerably slower
than the student and is intended for quality comparison.

```bash
just diffusion2one-teacher verify       # hashes, environment, release semantics
just diffusion2one-teacher smoke        # two synthetic inferences and KV replay
just diffusion2one-teacher              # model server only, 127.0.0.1:1117
just diffusion2one-teacher server-logs
just diffusion2one-teacher server-stop  # stops only the shared policy-server owner
```

These commands do not open cameras, ROS, or robot hardware. The default command
starts only the model server. `smoke` leaves that server available afterward;
use `server-stop` when finished. The teacher and student retain the existing
exclusive LingBot-family server owner and cannot run together.

The teacher deployment is
`configs/deployments/diffusion2one/fruit_blocks_teacher_eef.toml`, with
`execution.execute = true` and `execute_frames = 2` (8 actions per replan).
The `run` and `batch` actions execute on the real arm through the staged tracker
and relay. Select a registered task for a single attempt:

```bash
just diffusion2one-teacher run --task banana_to_red_plate  # MOVES HARDWARE
```

Use `server` or `smoke` for offline work. Single-task/count combinations are
run inputs, not permanent deployment files. Keep only deliberate multi-task
or benchmark protocols under `configs/runs/`; one-off plans and resolved run
records belong under `outputs/`. Create any new tracked batch plan with the
configuration CLI.

### A1 student

The registered A1 student is `SeanZheng/Distill-WAM` at immutable revision
`66d5902432d9715af8d6d4104943b5bb30f42d89`, using the `galaxea-a1/` files only.
Its source is the repository's `galaxea-a1` branch pinned at
`91abcc0db4cd5e282bf155a2ffb71de99b85d37f`. Deployment settings live in
`configs/deployments/diffusion2one/fruit_blocks_eef.toml`. The deployment has
`execution.execute = true`: the runtime command moves hardware. An explicit
user request authorizes the standard launch without another workspace
confirmation; use `execution.execute = false` for
camera-input testing without action execution.

```bash
just diffusion2one-env-setup # install/verify Python and CUDA imports; no model download
just diffusion2one-setup   # isolated environment and verified model components
just diffusion2one-verify # environment, artifact hashes, and release semantics
just diffusion2one-smoke  # synthetic images, GPU inference; no robot/cameras/ROS
```

The backend locks the official PyPI PyTorch 2.9.0 distribution and its CUDA
12.8 dependencies in a separate Python 3.12 environment. Environment-only setup
does not require the student checkpoint and does not start ROS or cameras.

Setup reuses content-verified local foundation files where available. Downloads
are staged under `models/artifacts/` and published only after hash validation.
An interrupted staging directory blocks the normal create-only fetch workflow;
inspect the active download and its logs before retrying. Do not delete model
files or start a competing downloader against the same destination.

After hardware-free smoke passes, review the deployment before running:

```bash
just diffusion2one --task red_block_to_red_plate
# Or: --task banana_to_red_plate
just stop
```

The runtime command attaches the cameras and robot runtime even when action
execution is disabled; follow the usual power/workspace checks. It uses the
same guarded EEF execution, CLI interaction, and run recording as LingBot.
Only the two registered training prompts are accepted. Keep the student at one
video step, one action step, and guidance 1; teacher sampling settings are
rejected. The standard launcher manages the local service at port 1116 and
shares LingBot's exclusive server owner. Remote-host lifecycle and MoT attention
diagnostics are not part of this adapter.

Finite absolute policy XYZ targets are capped to the System workspace:
each coordinate is capped to its configured minimum/maximum before IK. This
applies to all EEF policy bridges using that System
config. `EEF workspace cap` logs show the original and capped targets; LingBot's
temporal cache receives the capped action. Non-finite values, invalid joint
feedback, and relay faults still stop the run.

Diffusion2One sets `[execution] ik_replan_max_attempts = 3`. A typed IK
non-convergence or solution-delta rejection discards the remaining chunk,
stages fresh current joints as a hold on the already healthy ACTIVE relay,
resets model KV state, and captures a fresh episode origin. The next request
reads new camera observations and follows first-chunk indexing. The rejected
chunk's partial cache is not committed. Three consecutive replans are allowed;
only a fully executed chunk with successful cache synchronization replenishes
the allowance. Each inference still counts toward `max_model_calls`. Exhausted
allowances or hold/reset/feedback failures stop the run. The existing LingBot
deployments explicitly use `0`, preserving immediate IK safety stops.

Diffusion2One also enables `[execution.ik_subgoal]` in its deployment. After an
IK rejection it establishes a fresh hold and tries a short intermediate pose
toward the requested position and shortest-arc orientation. The initial target
step is at most 0.01 m / 0.05 rad; up to five attempts halve that step. Each solve
tightens the joint displacement bound to 0.05 rad and retains the System IK
convergence tolerances and absolute joint limits. Both the intermediate target
and its solved FK endpoint must lie in the configured workspace. The gripper
command is unchanged during this recovery move.

After publishing one intermediate target, the bridge waits up to 2 s for new
joint feedback confirming arrival within the original IK tolerances and
measurable progress toward the original pose. It then holds, discards all
remaining chunk actions and partial KV state, and re-infers from new camera
observations with a fresh episode origin. Confirmed progress replenishes the
rejection retry allowance, but every inference consumes `max_model_calls`; no
subgoal is attempted on the last call. If no intermediate target solves, the
ordinary bounded hold/replan path applies. Stale feedback, relay faults, or a
feedback timeout stop the run. This is bounded intermediate-target execution,
not permission to publish a non-converged IK candidate. Existing LingBot
deployments explicitly disable this feature. These endpoint and joint checks
do not implement swept-path collision checking.

This document is the operator procedure for setup, hardware acceptance, Teleop
collection, dataset conversion, recovery, and policy deployment. Commands that
can move the arm are labeled **MOVES HARDWARE**.

## Workspace measurement and review

This procedure measures an application workspace; it does not reset encoder
zeros or change mechanical joint limits. The current box bounds the origin of
`arm_seg6` in `base_link`, in meters. It does not directly bound the fingertips,
table clearance, other arm links, or the swept path. The historical Z limits
0.06 and 0.50 m were inherited defaults; no measurement record establishing
their clearance was found in the audited configuration history. X minimum
0.04 and Y maximum 0.17 came from outward-rounded demonstrated data in commit
`95e04898`. X maximum 1.0 and Y minimum -1.0 were changed after the operator
reported remeasurement on 2026-09-09; that report is not independent verification.
Those earlier bounds were replaced on 2026-09-10 using the per-axis extrema
of all 3092 saved frames in `workspace_boundary_20260910`, episode 0, then
manually adjusted by the operator. The operator confirmed checking combinations
of the extrema. `outputs/calibration/workspace_boundary_20260910/xyz_extrema.json`
retains the original measurements and source frames. Edit `[eef] xyz_min` and
`xyz_max` in `configs/system/a1.toml` for subsequent manual adjustments; the
runtime reads that file rather than the measurement report.

1. End autonomous inference before measuring. An explicit measurement/Teleop
   request authorizes the existing guarded path. Do not push a powered
   arm by hand or deliberately seek mechanical stops. Preserve existing joint
   limits while measuring; an actual encoder-zero error needs a separate,
   hardware-version-specific calibration procedure.
2. Identify the physical `base_link` origin and axes from the installed URDF
   and mounting geometry. Independently measure several accessible reference
   points and compare them with fresh joint-feedback forward kinematics. Record
   the joint vector, `arm_seg6` position/quaternion, tool geometry, mounting, URDF
   revision, and measurement error. Resolve systematic offsets before choosing
   new bounds; recording FK alone does not verify its calibration.
3. Measure at least three non-collinear table points in the same base frame,
   plus table edges and relevant obstacles. Use additional points to check the
   fitted plane and measurement uncertainty. Measure the offset and occupied
   volume of the open/closed gripper and carried object relative to `arm_seg6`.
   A tilted table needs a plane constraint or a conservative box for the whole
   task region; a constant Z threshold is not a general table-plane check.
4. Using guarded Teleop, record the intended pick, place, approach, and retract
   poses, including their orientations. Select a conservative task region
   inside the measured free region, with inward margins accounting for tool
   extent, measurement error, tracking error, and stopping distance. Do not use
   the independent extrema of reachable samples as proof that every point in
   their bounding box is reachable or collision-free. For a horizontal table,
   the required link-origin Z floor depends on table height and the lowest
   rotated tool/object point; a fixed floor must cover every permitted pose.
5. Save the measurements and assumptions under `outputs/calibration/<session>/`.
   Review the six resulting bounds in `configs/system/a1.toml`; keep the source
   record with the review. Run the static configuration doctor and
   `safety-report` below before starting any live validation. Static validation
   verifies configuration consistency, not physical clearance.
6. Check representative boundary poses and transitions offline for IK and
   joint limits. This runtime has no collision planner, so inspect link/tool
   clearances separately. Only after review, validate small motions from well
   inside the region through the guarded path and compare actual feedback with
   targets. Do not start a full autonomous rollout as the calibration test.

```bash
.venv/bin/python -m galaxea_a1_runtime.cli doctor
.venv/bin/python -m galaxea_a1_runtime.cli safety-report
```

TRAC-IK searches within absolute joint limits intersected with the System-owned
joint displacement bound around fresh feedback. Runtime independently checks
joint limits, FK position/orientation error, and displacement before staging the
endpoint. This displacement bound does not implement a time-based velocity or
acceleration limiter.

## 1. Static preflight

On a new checkout:

```bash
git submodule update --init --recursive
just setup
just check
```

`just check` runs the static doctor, shell syntax/style checks, and the Foxglove
extension build/lint checks. Individual app preflights stay with their app command
instead of being repeated here. `just models` is an optional deployment
preflight; missing checkpoints do not block Teleop collection.

Install the tracked serial and RealSense rules once per machine with `just
udev`, then start a new login shell and verify that `/dev/a1` exists and the
current user belongs to `dialout`.

## 2. Hardware and cameras

Power the arm, clear its workspace, and connect the configured leader and
cameras. These checks enumerate or read devices but do not command motion:

```bash
just hardware
just camera-check
```

`just camera-check` writes snapshots and FPS results to the output path printed from
System config. Resolve missing devices, stale frames, wrong image shapes, or USB
bandwidth failures before continuing.

Start or verify the persistent scoped LAN monitor once. This command also
ensures the shared roscore, telemetry adapter, and Foxglove Bridge:

```bash
just cameras
```

The live app entrypoints also ensure this monitor automatically, so the command
is only needed when monitoring without starting an app.

Open the printed URL and leave the page open. It contains only the AgentView and
wrist streams. The marked background Camera Bridge, not a tmux session, remains
the sole camera owner while LingBot, pi0.5, or Teleop collection runs. Those apps
consume the bridge's local uncompressed BGR/depth pairs with the original source
sequence numbers and monotonic timestamps. Web JPEG encoding is a low-rate,
latest-frame-only side branch and is display-only, so a slow browser drops Web
frames instead of queuing or changing inference and collection data.

`just camera-check` is the one explicit exception: the direct hardware diagnostic
temporarily stops the bridge so it can exercise camera construction and USB/FPS
checks, then restarts it. Normal inference and collection do not perform this
handoff.

To explicitly close the cameras, Web monitor, telemetry adapter, and Foxglove
Bridge:

```bash
just cameras stop
```

Use `just cameras status` or `just cameras logs` for the combined monitoring
stack. Normal `just stop` stops motion/inference resources but deliberately
leaves this stack running.

The preview is unauthenticated, unencrypted, and LAN-only. Do not port-forward
it.

### Persistent Foxglove workspace and collection console

The normal A1, joint, Teleop, LingBot, pi0.5, and TFP runtime compositions ensure and
reuse the same persistent Foxglove WebSocket at
`ws://<this-host>:8766`; they never start a competing bridge.

To inspect cameras without starting the A1 driver, tracker, relay, or any
command publisher, start the combined persistent observation stack:

```bash
just cameras start
```

In Foxglove, add a **Foxglove WebSocket** connection to
`ws://127.0.0.1:8766` (or the host's trusted-LAN address), then import
[`foxglove/layouts/a1_observability.json`](../foxglove/layouts/a1_observability.json).
The canonical organization layout is named **Galaxea A1 Operations**. Pushes to
`main` build the pinned, robot-neutral **Embodied Ops Collection Console** from
`external/embodied-ops`, publish it privately to the organization, and then
create or update the A1 layout through the Foxglove API. The layout supplies its
exact A1 topic and service names, so organization members normally select it
from the Layouts menu instead of installing the extension or importing JSON
manually.
The first tab contains both camera streams, A1 URDF/TF, and diagnostics; the
second contains measured/staged/forwarded joint and gripper plots plus ROS logs.
Staged and forwarded joint curves start disabled so the measured traces remain
readable. The first tab keeps the English-only collection console fully visible;
its lower **Diagnostics / 3D** tabs keep the robot model collapsed by default.
The collection console and the second tab's raw status
subscribe to the sanitized versioned workflow state on
`/a1/ops/workflow_status`; child argv and terminal logs are deliberately
excluded. The layout has no Publish panel.

To open the session and select its prompt:

```bash
just collect <experiment> "<exact prompt>"
```

Keep that terminal command running. It owns or attaches to the single Operator
Session and follows the child log. In Foxglove:

- `Ready`: **Start recording**, **Reset position**, and **End session** are
  available.
- `Preparing`: the dataset transaction is open and both cameras must produce a
  new frame; episode controls remain disabled.
- `Recording`: **Stop & save**, **Reset after save**, **Discard episode**, and
  **End session** are available. The status shows sampled/stored frame counts
  and effective FPS. Turn **Reset after save** off to keep the current pose and
  proceed directly to the next `Ready` gate; a discard still runs its configured
  automatic Reset.
- `Saving`, `Discarding`, `Resetting`, and other busy phases disable episode
  buttons until the child announces the next one-shot input gate.
- An unavailable session, stale telemetry, rejected command, or failed workflow
  is shown explicitly. Reset, discard, and end-session actions ask for
  confirmation.

If `just panel` is already running, `just collect` submits the collection to its
existing workflow owner. Otherwise, the command starts only a private
current-user Operator Session for that collection; it does not open another LAN
HTTP server. `Ctrl+C` stops the exact active run through the same supervised
boundary.

Standalone mode is intentionally partial: camera images are available when the
persistent Camera Bridge is running, while joints, relay status, motor status,
and TF remain unavailable until their owning execution runtime exists. This may
show red relay/motor diagnostics with the arm off; it does not open or probe the
arm. No `CameraInfo` or camera-to-robot calibration is currently tracked, so the
layout keeps images in 2D panels instead of inventing a 3D camera transform.

The shared stack remains alive across normal A1 runtime stop/start transitions
and does not require the arm to be powered. It remains a background service
until `just cameras stop`, host shutdown, or process failure; it is not installed
as an operating-system boot service. Use these commands when managing Foxglove
without changing Camera Bridge ownership:

```bash
just foxglove status
just foxglove logs
just foxglove restart
just foxglove stop
```

The endpoint has no authentication or TLS. Restrict port `8766` to the trusted
LAN and never proxy or port-forward it. Foxglove can inspect configured command
topics and call only the eight exact collection `std_srvs/Trigger` services; the
bridge denies client publication, parameter access, client-advertised topics,
and every other service. The service proxy accepts only the active `collect`
run's exact phase/action/input revision. Regenerate the committed layout after a
System topic, service, joint-name, or URDF change with `just foxglove-layout`;
review the generated diff and run `just foxglove restart` after changing tracked
observability configuration.

### Unified operator panel

Start the tracked control panel without opening hardware:

```bash
just panel
```

Open `http://127.0.0.1:8765` on this host, or
`http://<this-host-LAN-IP>:8765` from another device on the trusted LAN. The
tracked listener is `0.0.0.0:8765`; `hostname -I` prints candidate host
addresses. The panel lists every valid tracked Teleop, LingBot deployment,
Batch, model, and A1 reset configuration. It embeds the read-only Camera Web
streams and provides Collect, Dataset Doctor, Export v2.1, Evaluation, Batch,
and Reset views. It does not create configurations or register Prompts;
repository changes are CLI-owned. Use **Start cameras** if the persistent Camera
Bridge is not already running. Each preview reports its encoded preview FPS and
latest frame age from Camera Web's read-only health endpoint. A dark image alone
is not treated as a failure; the status changes only for missing, stale, or
errored frames. Use **Collapse preview** when more vertical room is useful for
workflow controls.

Buttons that start Collect, Evaluation, Batch, or Reset **MOVE HARDWARE**. They
launch the existing repository entrypoints and never publish ROS messages from
the Web server. Only one workflow may run at a time. Input buttons appear only
when the child is at the corresponding prompt; one click locks them until the
next prompt, so decisions cannot queue through a later step. **Stop** sends
`SIGINT` so the owning script can lock the relay and clean up. If cleanup does
not finish, the panel stays available and requires `just stop` before retrying.
The Reset view is A1-only: it validates the selected System and pose first,
starts ROS, the driver, joint tracker, and locked relay, performs the staged
reset, and always stops those owned services afterward.

Reset and LingBot inference progress appears above the session terminal and
updates in place. The terminal keeps durable lifecycle output in its own scroll
area, follows new lines while it is at the bottom, and preserves the current
position when you scroll up to inspect history.

Repository configuration and Prompt maintenance use the unified CLI:

```bash
just configs
just prompts
just prompt-catalog-create \
  configs/tasks/button_press/catalog.json button-press-v1 press_button_5s \
  "press and hold the button for 5 seconds, then release it" train
just prompt-register \
  configs/tasks/fruit_placement/catalog.json green_apple_bowl \
  "put the green apple into the bowl" ood
.venv/bin/galaxea-a1-runtime config template batch \
  configs/runs/lingbot/fruit_placement.toml > /tmp/new_batch.toml
# Edit /tmp/new_batch.toml, including a unique batch.id, then:
.venv/bin/galaxea-a1-runtime config validate batch new_batch /tmp/new_batch.toml
.venv/bin/galaxea-a1-runtime config create batch new_batch /tmp/new_batch.toml
.venv/bin/galaxea-a1-runtime hardware
.venv/bin/galaxea-a1-runtime collect EXPERIMENT --task "TASK"
.venv/bin/galaxea-a1-runtime dataset doctor EXPERIMENT
.venv/bin/galaxea-a1-runtime dataset doctor EXPERIMENT --json
.venv/bin/galaxea-a1-runtime dataset export-v21 EXPERIMENT
.venv/bin/galaxea-a1-runtime dataset export-v21 EXPERIMENT --json
.venv/bin/galaxea-a1-runtime evaluate TASK_ID --scene-note "SCENE"
.venv/bin/galaxea-a1-runtime batch configs/runs/lingbot/mango_placement.toml \
  --scene-note "SCENE" --resume
.venv/bin/galaxea-a1-runtime reset configs/poses/a1_collection_start.toml
```

Prompt registration is hardware-free and create-only: it rejects duplicate ids
or exact text, validates the complete catalog, assigns the next stable display
order, and writes one JSON record. Use `train` only for a prompt known to belong
to the training set; otherwise use `ood`. Review and commit the generated file.
New configurations follow the same agent-owned pattern: start from a same-kind
template, validate the candidate, then create and commit it.

The control panel uses a random per-process request-integrity token, but that
token is delivered to every browser that opens the page and is not user
authentication. Bind it only on a trusted LAN, restrict port `8765` with the
host firewall when the LAN is shared, and never proxy or port-forward it. The
separate Camera Web remains read-only and has no control endpoints.

The reusable Web/process/configuration core is provided by the pinned
`embodied-ops` dependency and documented in its
[`operator_panel/README.md`](../external/embodied-ops/src/embodied_ops/operator_panel/README.md).
Another repository provides its own adapter and child input announcements;
A1-specific loaders and commands are not part of that core.

Optional EEF acceptance **MOVES HARDWARE**:

```bash
just eef-test
```

This step-gated check sends each accepted Cartesian nudge through the same
bounded URDF IK, named joint target, isolated jointTracker, and fail-closed relay
used by both policy bridges. A target outside the tracked workspace stops the
rollout before that target is published and reports the offending axes and bounds.

Run it only with a clear workspace. After any partial startup failure, use
`just stop` before retrying.

Optional read-only ROS bag capture while an A1 runtime is already running:

```bash
just rosbag start SESSION_NAME
just rosbag status
just rosbag stop
```

The recorder does not publish commands. It records the configured state,
target, staged, host, relay, gripper, telemetry-camera, mirror, diagnostics, and
TF topics under `outputs/rosbags/`. Stop it with `just rosbag stop` so the active
bag is finalized cleanly. These bags can be opened directly in Foxglove with the
same committed layout.

## 3. Reset and Teleop acceptance

Reset **MOVES BOTH DEVICES**:

```bash
just reset
```

It loads the tracked reset pose, moves A1 through the staged joint runtime,
moves the SO leader, closes both grippers, disables leader torque, and stops the
runtime. Do not manually move either device during reset.

Optional Teleop acceptance **MOVES THE A1** without recording:

```bash
just teleop-test
```

Exercise all six joint directions and the continuous gripper over a small
range. This starts the modified leader Teleoperator, the Runtime-owned
relative-anchor processor, the narrow A1 Robot protocol client, and the
supervised A1 Runtime service as one owned control chain. The first submitted
command must exactly hold the A1's observed six joints and normalized gripper;
relative mapping, configured bias, and leader gripper input begin on the second
frame. Leader degree feedback is unwrapped across the encoder zero; a
remaining processed joint jump above the tracked per-frame threshold stops the
session before publication. The Robot reaches the Runtime-owned backend through
the lightweight `galaxea-a1-runtime-protocol` package and tracked local Unix
socket; it does not load ROS or Runtime Python code. Do not substitute the
generic `lerobot-teleoperate` command: in
LeRobot 0.6 it uses identity processors and cannot safely pair leader degrees
with A1 radians. Use `just logs` for failures and `just stop` when finished.

## 4. Record episodes

Collection **MOVES THE A1**:

```bash
just collect EXPERIMENT "put the fruit into the bowl"
```

`collect` starts the tracked services and cameras, then automatically resets A1
and the SO leader before exposing the first episode prompt. A separate
`just reset` remains available for acceptance and recovery, but is not part of
the normal collection sequence.

Reuse the same `EXPERIMENT` for related prompts. For example, four socket
positions belong to one `plug_insertion_v1` dataset:

```bash
just collect plug_insertion_v1 "pick up the charger and insert it into the first socket from the left on the power strip"
just collect plug_insertion_v1 "pick up the charger and insert it into the second socket from the left on the power strip"
```

Each invocation appends episodes to the same dataset while standard LeRobot
task metadata keeps the prompts distinct. The panel's Collect task field offers
tracked training prompts and still accepts an exact new prompt.

At the episode prompt:

- `Enter`: start recording; while recording, request save and validation;
- `d` + `Enter`: discard, reset both devices, and retry the same index;
- `q` + `Enter`: quit without reset;
- `Ctrl+C`: stop immediately.

Every frame requires fresh joint, EEF, gripper, action, and paired-camera data.
The collector buffers the stationary prefix and starts storing only after the
tracked per-action thresholds report sustained motion, while retaining a short
preroll. An episode with no detected motion remains empty and is discarded.
Save validates continuity and finalizes a standard LeRobotDataset v3 episode in
a hidden sibling snapshot, then atomically installs the complete dataset under
`data/datasets/EXPERIMENT/`. A rejected save removes only its snapshot, reuses
its index, and resets before retry when configured. A successful save resets
before the next episode when configured. Leader reset keeps the tracked goal
tolerance strict while making a bounded number of smooth corrective passes for
servo lag or backlash; a final failure reports the offending joint errors.

The collector loads ROS1 from the tracked Python 3.12 overlay and A1 SDK without
adding Ubuntu's system Python packages. This keeps LeRobot resume isolated from
ABI-incompatible optional packages such as the system SciPy build.

Do not hand-edit a dataset while collecting. The exact feature contract and
atomic append behavior are documented in
[Architecture](ARCHITECTURE.md).

## 5. Inspect the direct LeRobot dataset

After quitting:

```bash
just stop
just dataset-doctor EXPERIMENT
just dataset-doctor EXPERIMENT --json
find data/datasets/EXPERIMENT -maxdepth 3 -type f | sort | head
.venv/bin/python - <<'PY'
from pathlib import Path
from lerobot.datasets import LeRobotDatasetMetadata

root = Path("data/datasets/EXPERIMENT")
meta = LeRobotDatasetMetadata("OWNER/REPO-ID", root=root)
print(meta)
print(meta.features)
PY
```

Replace `OWNER/REPO-ID` with the ID printed by the collector. Joint-action
training can consume this v3 dataset directly; it already contains canonical
state/action vectors, paired cameras, per-frame task text, stats, and episode
metadata. Hidden sibling staging directories indicate an interrupted append;
inspect them before removal.

### Derive EEF or LeRobot v2.1 from canonical v3

Joint-action v3 training consumes the recorded dataset directly. For EEF action
semantics or an older LeRobot reader, create a strict tracked config such as
`configs/datasets/EXPERIMENT_derivatives.toml` with these owners:

```toml
[system]
config = "configs/system/a1.toml"

[derivation]
overwrite = false

[source]
root = "data/datasets/EXPERIMENT"

[outputs.joint_v21]
target_root = "data/processed/EXPERIMENT_joint_v21"
archive_path = "data/exports/EXPERIMENT_joint_v21.tar.gz"
repo_id = "OWNER/EXPERIMENT-joint-v21"

[outputs.eef_v3]
target_root = "data/processed/EXPERIMENT_eef_v3"
archive_path = "data/exports/EXPERIMENT_eef_v3.tar.gz"
repo_id = "OWNER/EXPERIMENT-eef-v3"

[outputs.eef_v21]
target_root = "data/processed/EXPERIMENT_eef_v21"
archive_path = "data/exports/EXPERIMENT_eef_v21.tar.gz"
repo_id = "OWNER/EXPERIMENT-eef-v21"

[kinematics]
urdf = "third_party/A1_SDK/install/share/mobiman/urdf/A1/urdf/A1_URDF_0607_0028.urdf"
base_link = "base_link"
tip_link = "arm_seg6"
```

Then build all derivatives, or one independently:

```bash
just derive configs/datasets/EXPERIMENT_derivatives.toml
just derive configs/datasets/EXPERIMENT_derivatives.toml eef-v3
just export-v21 EXPERIMENT
just export-v21 EXPERIMENT --json
just derive configs/datasets/EXPERIMENT_derivatives.toml eef-v2.1
```

The source repo ID and task are read from its committed provenance instead of
being duplicated in the derivative config. Every final output derives from the
canonical v3 root. Joint v2.1 exports directly; the temporary EEF v3 workspace
used for EEF v2.1 export is removed.

## 6. Failure recovery

First stop repository-owned resources:

```bash
just stop
```

Then diagnose the narrow layer:

| Symptom | Command or action |
| --- | --- |
| serial/device missing | `just hardware` |
| camera missing, stale, or slow | `just camera-check`; inspect USB topology |
| Teleop process exited | `just logs` |
| A1 Robot service socket already exists | run `just stop`, verify the service is absent, then inspect and remove only the exact socket configured in System config |
| model missing | `just models` |
| configuration or test failure | `just check` |
| derivation rejected a dataset | inspect its metadata and files; do not weaken validation |

Never run two apps that own the same driver, tracker, camera, serial port, or
publisher. A1 status interpretation and direct-debug procedures are maintained
only in [Safety](SAFETY.md).

The A1 Robot service deliberately refuses to unlink a pre-existing socket at startup. This
keeps a second runtime from taking over an endpoint that may still be owned. Treat a
leftover socket as crash evidence and remove it only after the owning process is proven
stopped.

## 7. Policy deployment

### EEF IK

TRAC-IK Distance is the sole numerical IK implementation, including the OpenRAL
gateway. `configs/system/a1.toml` keeps the URDF, base/tip links, solve timeout,
Cartesian acceptance tolerances and joint displacement bound under `[eef_ik]`.
The adapter owns its cache path, numerical epsilon and worker protocol deadlines.
Model chunk size, recovery, reset, tracker and relay remain separate concerns.

Build and check the numerical adapter without hardware:

```bash
just trac-ik-setup
just trac-ik-check
just check
```

The build uses the System runtime image's installed `ros-noetic-trac-ik-lib`.
The cache receipt records the immutable image and source/binary hashes. Rebuild
after changing the native adapter or runtime image. Static config parsing and
`just check` do not require this local build. EEF run/batch launchers check it
before opening cameras or robot services; the solver also verifies it before
starting its private worker. Missing or stale builds stop execution.

The solver uses fresh feedback and independently checks FK, absolute joint limits
and displacement. Bridges recheck feedback after solving. Worker errors stop the
application. Recordings include the fixed `trac_ik_distance` identity and solve
time. The solver returns endpoints, not velocity-limited or collision-checked
trajectories. The retired numerical backends are not fallback options.

### Model setup

Set up either pinned EEF-policy backend and its immutable Hugging Face artifact,
then exercise the complete model-service protocol without ROS, cameras, or arm
I/O:

```bash
just lingbot-setup
just lingbot-smoke
just lingbot-attention
scripts/apps/lingbot/a1_lingbot_runtime.sh server-stop

just pi05-setup
just pi05-smoke
scripts/apps/pi05/a1_pi05_runtime.sh server-stop

just tfp-setup
just tfp-smoke
```

LingBot smoke validates reset, inference, temporal-cache synchronization, and
reinference. `just lingbot-attention` loads the tracked real teacher-forcing
episode without opening hardware, populates the actual-observation KV cache,
decodes the model's exact paired-camera prediction, aligns the four latent
attention frames to RGB anchors 0/4/8/12 of the 13-frame VAE decode, and writes
source images plus 30-layer WAM overlays under
`outputs/inference/lingbot-fruit-placement-eef/attention-audits/`. The audit
separates action-to-predicted-future, predicted-future-to-actual-history,
direct action-to-actual-history, and the composed
action-via-predicted-future-to-history paths. The four cached real
observations produce one new streaming-VAE latent, so actual-history maps align
to raw frame anchors 0/4 rather than inventing maps for frames 1–3. Every
front/wrist 8×8 map is overlaid on the actual or decoded future image that owns
those tokens. These are diagnostic associations: the rollout averages heads,
averages only the configured executed-action queries, and omits text
cross-attention, MLP/gating, cached-action, and other unselected cache paths. It
is not causal attribution.
Pi0.5 smoke runs one synthetic two-camera/state inference and validates the
returned horizon. TFP smoke loads the transferred step-1500 checkpoint with
the pinned TFP-Ultra checkout, runs synthetic RGB inference on CUDA, and sends
no robot command. The LingBot and Pi0.5 smokes leave their managed GPU server available for log
inspection; the matching `server-stop` command releases it. Follow the
[Model registry](../models/README.md) to review the exact input/action contract.
A new weight revision gets a new model descriptor and manifest; do not repoint
a mutable alias or edit an existing revision in place.

For an OpenRAL-owned deployment, keep the Runtime ROS execution bridge stopped.
Start only the persistent cameras, the contract-checked LingBot model server,
and the ROS-free policy gateway:

```bash
just cameras start
scripts/apps/lingbot/a1_lingbot_runtime.sh server
uv run galaxea-a1-openral-policy \
  --config configs/deployments/lingbot/fruit_placement_eef.toml \
  --repo-root .
```

The gateway and Camera Bridge publish private per-user Unix sockets under
`A1_PROCESS_STATE_ROOT` (or the standard runtime directory). OpenRAL discovers
their versioned contracts directly; it does not need the Runtime checkout on
`PYTHONPATH`. Stop the OpenRAL process before stopping either provider.

For a hardware-free replay against the real processed training episodes:

```bash
just offline-eval
# or assign a durable run identity
just offline-eval REVIEW_ID
```

This command starts only one managed GPU model service at a time. It does not
initialize ROS, cameras, serial devices, or robot publishers. It validates all
130 episode tables, checkpoint provenance, trim/RGB alignment and normalization,
then runs LingBot first-frame inference plus teacher-forced cache replay and
Pi0.5 first/middle/late-frame inference. Results and contact sheets are written
under `outputs/offline_evaluation/fruit_placement/RUN_ID/`. This is a
training-set regression check, not evidence of held-out generalization or live
closed-loop task success.

For a focused sequential Teacher Forcing replay of the tracked episode, with a
per-step prediction-to-ground-truth action report for both models:

```bash
just teacher-force
# or assign a durable run identity
just teacher-force REVIEW_ID
```

LingBot receives ground-truth post-action images and ground-truth actions in
each temporal-cache update. Pi0.5 receives the ground-truth image and full state
at every step; its service has no action-history input. The result is written as
`TEACHER_FORCING_REPORT.md` beside the detailed per-step JSON files under the
offline evaluation output root.

Starting either app may **MOVE THE A1** when its tracked execution setting is
enabled:

```bash
just lingbot
# select another registered LingBot checkpoint without editing a deployment
just lingbot --model mango_placement_eef
# plug insertion uses its own model and task catalog
just lingbot --config configs/deployments/lingbot/plug_insertion_eef.toml

just pi05
tmux attach -t pi05-a1

# Absolute-joint TFP-Ultra checkpoint; runs in the invoking terminal.
just tfp
```

`just lingbot` first requires a non-empty scene note, then starts a fresh marked
policy-server process and the A1 services and runs the bridge directly in the
invoking terminal. Its single `[RUN]` line
updates in place with inference, execution, EEF, and paired-camera recording
progress. `Ctrl+C` stops the foreground bridge, locks the relay, and tears down
the policy server and A1 services. LingBot has no tmux attach/detach lifecycle.

`just tfp` verifies the pinned source, isolated package versions, metadata, and
full checkpoint SHA256 before opening hardware, then performs the tracked
smooth A1-only reset to the checkpoint's training start pose. Diffusion sampling
uses the pinned seed after model construction, matching the upstream deployment
smoke. It then reads the same raw
Camera Bridge pair and sends the checkpoint's seven absolute joint/gripper
values through the local A1 Robot service. The service reuses the shared current
hold, exclusive lease, finite/absolute-limit checks, staged tracker, and
fail-closed relay. TFP targets are not clipped or projected at action-chunk
boundaries; the operator has accepted the reviewed checkpoint's discontinuity
at its first replanning boundary. The tracked `max_actions=0` means unlimited
execution at 30 Hz; `Ctrl+C`, camera staleness, an invalid action, or a service timeout
all close the command lease and lock the relay.
At each eight-action policy query, the tracked diagnostics print the LTC belief
norm/update, cosine retention, effective-time-constant and write-gain percentiles,
and aggregate per-layer memory-FiLM contribution. The complete 256-dimensional
before/after belief and all 12 layer ratios are written under
`outputs/inference/tfp-press-button/diagnostics/<run-id>/belief.jsonl`. These are
continuous hidden-state diagnostics, not labeled event probabilities. Collection
reuses the existing forward pass and never submits an additional diagnostic action.

The persistent AgentView/wrist dashboard remains at
`http://0.0.0.0:8088` (replace `0.0.0.0` with this host's LAN address from
another machine) before, during, and after a run. The bridge records the full,
unoverlaid AgentView and wrist streams from the same atomic raw-frame pair used
by inference. Normal completion, an execution error, and `Ctrl+C` all lock the
relay before closing the camera and atomically publish both named MP4s plus
their source timeline under
`outputs/inference/lingbot-fruit-placement-eef/recordings/`.
Every selected run,
including a startup that later fails, gets one timestamped task directory. A
successful recorded run contains
`SCENE_NOTE__INPUT_PROMPT__YYYYMMDD_HHMMSS__front.mp4`, the matching
`__wrist.mp4`, `camera_timeline.jsonl`, `camera_recording.json`, `runtime.log`,
`policy_server.log`, and `metadata.json`. Filename components retain Unicode
letters/digits and replace punctuation or whitespace with `_`. The metadata
binds the original scene note, exact prompt, task id and distribution,
deployment/System/model configuration, model and Git revisions, timestamps,
exit status, and artifact names to that run. The final absolute run directory
and video frame count are printed after finalization.

LingBot and Diffusion2One rollouts also always save a read-only `motion/` journal,
independently of the camera-video switch. It starts when the policy bridge opens
(after the batch reset) and closes after motion is disabled. `events.jsonl`
retains measured joint position/velocity/effort and names, EEF and gripper
feedback, named joint targets, tracker staged commands, relay forwarded commands
and status, model targets, exact IK input joints, solutions/rejections, and
recovery origins. It records delivered messages without periodic sampling. Each event has a
contiguous journal sequence, host monotonic/wall timestamps, and original ROS
header stamps. Divide `monotonic_ns` by 1e9 to align receipt times with the
camera timeline's source monotonic clock; source capture and callback receipt
times are distinct. `robot.urdf`, `system.toml`, and `deployment.toml` snapshot
the reconstruction inputs with SHA-256 receipts. Journal metadata reports
written stream counts and completion; `source_sequence_stats` separately flags
source sequence discontinuities (upstream/ROS receive loss or publisher restarts).
Completion means received events were drained to disk, not guaranteed delivery
of every source publication. Same-process ROS subscribers share the control
subscriber's existing receive policy. Callbacks use a bounded asynchronous
writer, and overflow/write failures stop the rollout at the next recording
health check. An absent completion manifest indicates an interrupted recorder,
not a complete trajectory. This journal does not include the preceding reset.
While running it is in `.<run_id>.motion/`; finalization moves it into the run
directory. It creates no ROS command publishers and changes no IK limits.

For an Enter-gated sequence of multiple prompts and repeated trials:

```bash
just lingbot-batch
# or use another tracked plan
just lingbot-batch configs/runs/lingbot/fruit_placement.toml
# run all six catalog tasks with the registered step-200 mango checkpoint
just lingbot-batch --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
# three Enter-gated attempts of the trained first-socket plug prompt
just lingbot-batch configs/runs/lingbot/plug_insertion.toml
```

Edit `retries_per_prompt` and the ordered `task_ids` in the tracked run plan.
`retries_per_prompt=0` means one attempt per prompt; `2` means one initial
attempt plus two repetitions. Enter one scene note for the batch. Before every
attempt, the command displays its task/repetition index and waits: `Enter`
starts the tracked A1-only reset and then inference, while `q` stops before the
next reset. The SO leader is not opened. Every attempt gets its own paired
videos, camera timeline, metadata, and logs. An IK target that does not converge
or exceeds the tracked
solution-delta bound safely locks the arm without publishing the rejected target, finalizes the
attempt with `status=safety_stopped`, and asks for an evaluation decision.
`Enter` counts it and advances, `d` records it as discarded and returns the same
slot to the Enter/reset gate, and `q` stops with that slot pending. The decision
is stored in `metadata.json`. Reset, model, ROS, camera, serial, and other
infrastructure failures still abort the batch and remain pending.

The plug-insertion checkpoint was trained only on `charger_socket_1`. Its
dedicated model-bound catalog exposes only that prompt; the separate collection
catalog retains the other socket prompts without claiming that this checkpoint
trained on them. The tracked 286-call budget covers the longest 1143-frame
source episode at four executed actions per model call, so the policy replans
twice as often while retaining the original 1144-action budget. Use the
dedicated deployment; selecting `plug_insertion_eef` on the default fruit
deployment would retain the wrong task catalog.

Resume the same scene without repeating durable completed slots:

```bash
just lingbot-batch-resume
just lingbot-batch-resume --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
```

Enter the exact same scene note. Resume validates the current plan's batch id,
task position, attempt number, both videos, camera timeline/sidecar, shared frame
count, and both logs. It skips
`completed` slots and `safety_stopped` slots explicitly counted by the operator;
discarded, undecided, interrupted, and infrastructure-failed slots run again.
Earlier runs made before typed safety-stop metadata are also recognized when
their valid runtime log contains a known IK or workspace target rejection
message and their operator decision is recorded as counted.

Inspect which slots are valid at any time:

```bash
just lingbot-batch-report randomized_A
just lingbot-batch-report randomized_A --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
```

The report lists every plan slot as `VALID`, `PENDING`, or
`DUPLICATE_VALID`, including the selected run id and evaluation decision. A
normal completed rollout is valid automatically; a target safety stop is valid
only after the operator counts it. Discarded and undecided runs are excluded.

After all slots are valid, export exactly one run per slot:

```bash
just lingbot-batch-export randomized_A
just lingbot-batch-export randomized_A --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
```

Export refuses incomplete or duplicate-valid batches. It atomically writes one
`.tar` under `outputs/exports/lingbot/` containing a manifest plus each selected
run's MP4, `metadata.json`, `runtime.log`, and `policy_server.log`. The manifest
records prompt/task provenance, attempt indices, status/decision, file sizes,
and SHA-256 hashes. MP4s are already compressed, so the tar is intentionally not
gzip-compressed.

Run one live app at a time and use `just stop` when switching.

Both commands first display the approved prompts from
`configs/tasks/fruit_placement/prompts/`: five training prompts and the explicitly
marked OOD `lemon_bowl` evaluation prompt. Select by number, tracked task id, or
the exact prompt; `q` cancels before the model server, ROS, cameras, or hardware
are opened. The selected task id, train/OOD provenance, and exact prompt are
printed again by the bridge.

The reviewed fruit-placement deployments are currently live-enabled. After
task selection, they start their first inference automatically when fresh
observations are available. Each deployment reads its operator-selected rollout
cadence directly from its tracked `[execution]` table; edit that owning config
when changing how much model output is consumed before replanning. Both solve
EEF targets with the tracked first-party IK and publish named joint targets
through jointTracker. Neither deployment waits for inference or action
confirmation. Their tracked finite call budgets cover the longest 526-step
training episode. Use `Ctrl+C` in the foreground LingBot terminal when its
rollout should end, or `just stop` from another terminal; normal completion and
manual stop both lock the relay, finalize AgentView recording, and end
successfully. A
genuine feedback or safety failure remains a nonzero error and identifies the
stale feedback source.

### Settled observations for LingBot / Diffusion2One

Each deployment declares `[execution.settle]`. Disabled settling contains only
`enabled = false`; enabled settling requires `min_wait_s`, `stable_window_s`,
`timeout_s`, `joint_range_rad`, and normalized `gripper_range`. Both Student and Teacher deployments currently enable this experiment.
After the last action of each chunk, it retains the staged target, checks fresh
joint/gripper feedback and relay health, waits at least `min_wait_s`, and requires
measured joint and gripper ranges within the configured bounds for a continuous
stable window. This detects stillness, not target arrival or grasp success.
Timeout or invalid feedback stops the rollout through normal cleanup.

The final history image is captured from both cameras strictly after the
stillness gate, preserving the earlier history images and requested-action
cache shape. Only then is `compute_kv_cache` called. The upstream service encodes
ordinary query images only on the initial call, so sleeping after cache update
does not provide a settled image to later predictions. No extra temporal token
or fabricated repeated frame is inserted. The changed final sampling interval
is an experimental deployment cadence and is not equivalent to training timing.
Motion events `settle_start`, `settle_complete`, and
`settled_history_observation` retain duration, measured ranges, endpoint tracking
error and the camera capture lower bound. Full paired video and feedback remain
recorded. To disable it, replace the table with only `enabled = false`. The same
enabled/disabled schema rule applies to `[execution.ik_subgoal]`.
