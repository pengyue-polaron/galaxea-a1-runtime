# Runbook

This document is the operator procedure for setup, hardware acceptance, Teleop
collection, dataset conversion, recovery, and policy deployment. Commands that
can move the arm are labeled **MOVES HARDWARE**.

## 1. Static preflight

On a new checkout:

```bash
just setup
just check
```

`just check` validates tracked configuration, shell syntax, formatting, and
unit tests without opening hardware. `just models` is an optional deployment
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

For read-only LAN preview when no app owns the cameras:

```bash
just camera-web
```

Open the printed URL. The AgentView overlay shows the exact configured policy
crop but is not recorded. Stop preview before another camera-owning app:

```bash
just camera-web stop
```

The preview is unauthenticated, unencrypted, and LAN-only. Do not port-forward
it.

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
range. Use `just logs` for failures and `just stop` when finished.

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
Save validates continuity and exact files, then atomically installs the episode
under `data/raw/EXPERIMENT/`. A rejected save is deleted, reuses its index, and
resets before retry when configured. A successful save resets before the next
episode when configured.

Do not mix manually edited or older-schema episodes into a current experiment.
The exact raw contract and commit behavior are documented in
[Architecture](ARCHITECTURE.md).

## 5. Inspect and convert

After quitting:

```bash
just stop
find data/raw/EXPERIMENT -maxdepth 2 -type f | sort | head
```

Each episode contains metadata, frame records, and configured camera folders.
Hidden sibling staging directories indicate an interrupted commit; inspect them
before removal.

Create a tracked `configs/datasets/EXPERIMENT.toml`, then run the complete
conversion pipeline:

```bash
just convert EXPERIMENT
```

The default builds all four outputs. Build one independently when only one
training format is needed:

```bash
just convert EXPERIMENT joint-v3
just convert EXPERIMENT joint-v2.1
just convert EXPERIMENT eef-v3
just convert EXPERIMENT eef-v2.1
```

The dataset config references the tracked Raw-package config that owns the
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

Conversion rejects incomplete or mismatched raw data and preserves an existing
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

just pi05
tmux attach -t pi05-a1
```

`just lingbot` starts its marked policy-server process and the A1 services, then
runs the bridge directly in the invoking terminal. Its single `[RUN]` line
updates in place with inference, execution, EEF, and AgentView recording
progress. `Ctrl+C` stops the foreground bridge, locks the relay, and tears down
the policy server and A1 services. The policy-server log is written to
`outputs/inference/lingbot-fruit-placement-eef/policy_server.log`; LingBot has no
tmux attach/detach lifecycle.

While LingBot is running, open the AgentView/wrist dashboard at
`http://0.0.0.0:8088` (replace `0.0.0.0` with this host's LAN address from
another machine). The bridge records the full, unoverlaid AgentView stream from
the already-owned camera reader. Normal completion, an execution error, and
`Ctrl+C` all lock the relay before closing the camera and atomically publishing
`agent_view.mp4` under
`outputs/inference/lingbot-fruit-placement-eef/recordings/`. The final absolute
video path and frame count are printed after the MP4 is finalized.

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
