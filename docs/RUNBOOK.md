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

Run it only with a clear workspace. After any partial startup failure, use
`just stop` before retrying.

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

The dataset config owns packaging paths and policy only; observation and action
contracts derive from its referenced Teleop and System configs. Conversion
rejects incomplete or mismatched raw data and preserves an existing complete
output if replacement fails.

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

Bring reviewed weights onto this machine and register them without copying:

```bash
just model-link act-a1-agentview-square /path/to/act_checkpoint
just model-link lingbot-a1-agentview-square /path/to/lingbot_checkpoint
just models
```

Follow [Model registry](../models/README.md) to verify the input and action
contract. Update the owning deployment config and review it before marking the
checkpoint ready or enabling execution.

Starting either app may **MOVE THE A1** when its tracked execution setting is
enabled:

```bash
just act
tmux attach -t act-a1

just lingbot
tmux attach -t lingbot-a1
```

Run one live app at a time and use `just stop` when switching.
