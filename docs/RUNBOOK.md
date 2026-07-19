# Runbook

This document is the operator procedure for setup, hardware acceptance, Teleop
collection, dataset conversion, recovery, and policy deployment. Commands that
can move the arm are labeled **MOVES HARDWARE**.

## 1. Static preflight

On a new checkout:

```bash
git submodule update --init --recursive
just setup
just check
```

`just check` runs one static doctor, shell syntax and style checks, then the
hardware-free test suite. Individual app preflights stay with their app command
instead of being repeated here. `just models` is an optional deployment
preflight; missing checkpoints do not block Teleop collection.

Install serial rules once per machine with `just udev`, then start a new login
shell. See [Environment setup](SETUP_ENV.md) and [udev setup](SETUP_UDEV.md).

## 2. Hardware and cameras

Power the arm, clear its workspace, and connect the configured leader and
cameras. These checks enumerate or read devices but do not command motion:

```bash
just hardware
just cameras
```

`just cameras` writes snapshots and FPS results to the output path printed from
System config. Resolve missing devices, stale frames, wrong image shapes, or USB
bandwidth failures before continuing.

Start or verify the persistent read-only LAN monitor once:

```bash
just camera-web
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

`just cameras` is the one explicit exception: the direct hardware diagnostic
temporarily stops the bridge so it can exercise camera construction and USB/FPS
checks, then restarts it. Normal inference and collection do not perform this
handoff.

To explicitly close both cameras and the Web monitor:

```bash
just camera-web stop
```

Use `just camera-web status` or `just camera-web logs` for monitor diagnostics.

The preview is unauthenticated, unencrypted, and LAN-only. Do not port-forward
it.

### Unified operator panel

Start the local control panel without opening hardware:

```bash
just panel
```

Open `http://127.0.0.1:8765`. The panel lists every valid tracked Teleop,
LingBot deployment, Batch, model, and A1 reset configuration. It embeds the
read-only Camera Web streams and provides Collect, Evaluation, Batch, and Reset
views. Use **Start cameras** if the persistent Camera Bridge is not already
running.

The **Configurations** view creates Teleop, LingBot deployment, Batch, or A1
reset TOML from an existing same-kind template. Choose a template, load it,
change the filename and content, then validate before creating it. Creation runs
the owning strict loader and atomically exposes a new file; it never edits or
overwrites an existing configuration and is disabled during a live workflow.
For a new Batch, change `batch.id` as well as the filename because Batch IDs are
unique. Review and commit a new configuration before treating it as durable
repository state.

Buttons that start Collect, Evaluation, Batch, or Reset **MOVE HARDWARE**. They
launch the existing repository entrypoints and never publish ROS messages from
the Web server. Only one workflow may run at a time. Input buttons appear only
when the child is at the corresponding prompt; one click locks them until the
next prompt, so decisions cannot queue through a later step. **Stop** sends
`SIGINT` so the owning script can lock the relay and clean up. If cleanup does
not finish, the panel stays available and requires `just stop` before retrying.

The same configuration registry is available from the unified CLI:

```bash
.venv/bin/galaxea-a1-runtime configs
.venv/bin/galaxea-a1-runtime config template batch \
  configs/runs/lingbot/fruit_placement.toml > /tmp/new_batch.toml
# Edit /tmp/new_batch.toml, including a unique batch.id, then:
.venv/bin/galaxea-a1-runtime config validate batch new_batch /tmp/new_batch.toml
.venv/bin/galaxea-a1-runtime config create batch new_batch /tmp/new_batch.toml
.venv/bin/galaxea-a1-runtime collect EXPERIMENT --task "TASK"
.venv/bin/galaxea-a1-runtime evaluate TASK_ID --scene-note "SCENE"
.venv/bin/galaxea-a1-runtime batch configs/runs/lingbot/mango_placement.toml \
  --scene-note "SCENE" --resume
.venv/bin/galaxea-a1-runtime reset configs/poses/a1_collection_start.toml
```

The control panel is intentionally fixed to localhost and uses a random
per-process request token. Do not proxy or port-forward it. The separate Camera
Web remains read-only and has no control endpoints.

The reusable Web/process/configuration core is documented in
[`operator_panel/README.md`](../operator_panel/README.md). Another repository
provides its own adapter and child input announcements; A1-specific loaders and
commands are not part of that core.

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
target, staged, host, relay, and gripper topics under `outputs/rosbags/`. Stop
it with `just rosbag stop` so the active bag is finalized cleanly.

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
range. This starts the modified leader Teleoperator, the tracked relative-anchor
processor, the A1 Robot plugin, and the supervised runtime backend as one owned
control chain. Do not substitute the generic `lerobot-teleoperate` command: in
LeRobot 0.6 it uses identity processors and cannot safely pair leader degrees
with A1 radians. Use `just logs` for failures and `just stop` when finished.

## 4. Record episodes

Collection **MOVES THE A1**:

```bash
just reset
just teleop EXPERIMENT
```

Enter the natural-language task once. At the episode prompt:

- `Enter`: start recording; while recording, request save and validation;
- `d` + `Enter`: discard, reset both devices, and retry the same index;
- `q` + `Enter`: quit without reset;
- `Ctrl+C`: stop immediately.

Every frame requires fresh joint, EEF, gripper, action, and paired-camera data.
Save validates continuity and finalizes a standard LeRobotDataset v3 episode in
a hidden sibling snapshot, then atomically installs the complete dataset under
`data/datasets/EXPERIMENT/`. A rejected save removes only its snapshot, reuses
its index, and resets before retry when configured. A successful save resets
before the next episode when configured.

Do not hand-edit a dataset while collecting. The exact feature contract and
atomic append behavior are documented in
[Architecture](ARCHITECTURE.md).

## 5. Inspect the direct LeRobot dataset

After quitting:

```bash
just stop
just dataset-doctor EXPERIMENT
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
just derive configs/datasets/EXPERIMENT_derivatives.toml joint-v2.1
just derive configs/datasets/EXPERIMENT_derivatives.toml eef-v2.1
```

The source repo ID and task are read from its committed provenance instead of
being duplicated in the derivative config. Every final output derives from the
canonical v3 root; temporary v3 workspaces used for v2.1 export are removed.

### Legacy Raw v3 migration

The commands below exist for the recordings already under `data/raw/`; they are
not a post-processing requirement for new collection.

Create a tracked `configs/datasets/EXPERIMENT.toml`, then run the complete
conversion pipeline:

```bash
just legacy-convert EXPERIMENT
```

The default builds all four outputs. Build one independently when only one
training format is needed:

```bash
just legacy-convert EXPERIMENT joint-v3
just legacy-convert EXPERIMENT joint-v2.1
just legacy-convert EXPERIMENT eef-v3
just legacy-convert EXPERIMENT eef-v2.1
```

The legacy dataset config references the tracked Raw-package config that owns the
logical Raw v3 identity and its non-empty task-root list; it owns processed
packaging paths, overwrite policy, and the explicit boundary-trim policy. A
multi-task output preserves each root's `task.txt` on its episodes. Observation
and action contracts derive from the referenced Teleop and System configs. Use
the reviewed trim values unless a dataset inspection justifies changing them:

```toml
[trim]
enabled = true
anchor_window_s = 0.5
joint_deadband_rad = 0.01
gripper_deadband = 0.01
confirm_frames = 5
pre_roll_s = 0.5
post_roll_s = 0.75
max_trim_fraction = 0.20
min_kept_duration_s = 5.0
```

Legacy migration rejects incomplete or mismatched raw data and preserves an existing
complete output if replacement fails. It removes only stable stationary
episode boundaries, never interior pauses; uncertain or over-large candidates
remain untrimmed. Inspect `meta/trim.json` in any output for the exact source
frame interval and reason for every episode. Every selected output starts from
the same trimmed Raw v3 view; processed output directories are never chained
together. A complete run emits model-agnostic Joint and EEF datasets in both
LeRobotDataset v3.0 and v2.1. Training and deployment adapters select the
appropriate representation without changing the stored contract.

## 6. Failure recovery

First stop repository-owned resources:

```bash
just stop
```

Then diagnose the narrow layer:

| Symptom | Command or action |
| --- | --- |
| serial/device missing | `just hardware` |
| camera missing, stale, or slow | `just cameras`; inspect USB topology |
| Teleop process exited | `just logs` |
| model missing | `just models` |
| configuration or test failure | `just check` |
| conversion rejected an episode | inspect its metadata and files; do not weaken validation |

Never run two apps that own the same driver, tracker, camera, serial port, or
publisher. A1 status interpretation and direct-debug procedures are maintained
only in [Safety](SAFETY.md).

## 7. Policy deployment

Set up either pinned EEF-policy backend and its immutable Hugging Face artifact,
then exercise the complete model-service protocol without ROS, cameras, or arm
I/O:

```bash
just lingbot-setup
just lingbot-smoke
scripts/apps/lingbot/a1_lingbot_runtime.sh server-stop

just pi05-setup
just pi05-smoke
scripts/apps/pi05/a1_pi05_runtime.sh server-stop
```

LingBot smoke validates reset, inference, temporal-cache synchronization, and
reinference. Pi0.5 smoke runs one synthetic two-camera/state inference and
validates the returned horizon. Both leave the managed GPU server available for
log inspection; the matching `server-stop` command releases it. Follow [Model
registry](../models/README.md) to review the exact input/action contract. A new
weight revision gets a new model descriptor and manifest; do not repoint a
mutable alias or edit an existing revision in place.

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

just pi05
tmux attach -t pi05-a1
```

`just lingbot` first requires a non-empty scene note, then starts a fresh marked
policy-server process and the A1 services and runs the bridge directly in the
invoking terminal. Its single `[RUN]` line
updates in place with inference, execution, EEF, and AgentView recording
progress. `Ctrl+C` stops the foreground bridge, locks the relay, and tears down
the policy server and A1 services. LingBot has no tmux attach/detach lifecycle.

The persistent AgentView/wrist dashboard remains at
`http://0.0.0.0:8088` (replace `0.0.0.0` with this host's LAN address from
another machine) before, during, and after a run. The bridge records the full,
unoverlaid AgentView stream from its raw-frame channel. Normal completion, an
execution error, and
`Ctrl+C` all lock the relay before closing the camera and atomically publish the
named MP4 under `outputs/inference/lingbot-fruit-placement-eef/recordings/`.
Every selected run,
including a startup that later fails, gets one timestamped task directory. A
successful recorded run contains one
`SCENE_NOTE__INPUT_PROMPT__YYYYMMDD_HHMMSS.mp4`, `runtime.log`,
`policy_server.log`, and `metadata.json`. Filename components retain Unicode
letters/digits and replace punctuation or whitespace with `_`. The metadata
binds the original scene note, exact prompt, task id and distribution,
deployment/System/model configuration, model and Git revisions, timestamps,
exit status, and artifact names to that run. The final absolute run directory
and video frame count are printed after finalization.

For an Enter-gated sequence of multiple prompts and repeated trials:

```bash
just lingbot-batch
# or use another tracked plan
just lingbot-batch configs/runs/lingbot/fruit_placement.toml
# run all six catalog tasks with the registered step-200 mango checkpoint
just lingbot-batch --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
```

Edit `retries_per_prompt` and the ordered `task_ids` in the tracked run plan.
`retries_per_prompt=0` means one attempt per prompt; `2` means one initial
attempt plus two repetitions. Enter one scene note for the batch. Before every
attempt, the command displays its task/repetition index and waits: `Enter`
starts the tracked A1-only reset and then inference, while `q` stops before the
next reset. The SO leader is not opened. Every attempt gets its own video,
metadata, and logs. An IK target that does not converge or exceeds the tracked
solution-delta bound, or a finite target outside the tracked EEF workspace,
safely locks the arm without publishing the rejected target, finalizes the
attempt with `status=safety_stopped`, and asks for an evaluation decision.
`Enter` counts it and advances, `d` records it as discarded and returns the same
slot to the Enter/reset gate, and `q` stops with that slot pending. The decision
is stored in `metadata.json`. Reset, model, ROS, camera, serial, and other
infrastructure failures still abort the batch and remain pending.

Resume the same scene without repeating durable completed slots:

```bash
just lingbot-batch-resume
just lingbot-batch-resume --model mango_placement_eef configs/runs/lingbot/mango_placement.toml
```

Enter the exact same scene note. Resume validates the current plan's batch id,
task position, attempt number, video, frame count, and both logs. It skips
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

Both commands first display the six approved prompts from
`configs/tasks/fruit_placement.toml`: five training prompts and the explicitly
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
