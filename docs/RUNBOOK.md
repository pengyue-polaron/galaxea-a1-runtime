# Runbook

This is the operator procedure for checking the rig, teleoperating, recording
new data, and converting it. Commands that can move the arm are called out
explicitly.

## 1. Static Preflight

These commands do not publish robot commands:

```bash
just stop
just check
```

`just check` validates all tracked configs, shell syntax, formatting, and unit
tests. `just models` is an optional inference preflight and is expected to fail
until new deployment weights are registered; missing weights do not block
Teleop collection.

Inspect current static safety values when needed:

```bash
.venv/bin/python -m galaxea_a1_runtime.cli safety-report
.venv/bin/python -m galaxea_a1_runtime.cli safety-report --json
```

If this is a new machine, run `just setup` and `just udev`, then open a new
login shell so serial group membership applies.

## 2. Hardware and Camera Acceptance

Power the arm, clear its workspace, connect the SO leader and both RealSense
cameras, then run non-motion enumeration:

```bash
just hardware
just cameras
```

Expected configured devices:

- A1 controller at `/dev/a1`;
- SO leader at the by-id path in `configs/teleop/a1_so100.toml`;
- AgentView D455 serial `341522300456`;
- wrist D405 serial `218622276998`.

`just cameras` writes diagnostics under the system-configured output directory,
checks sustained frame production, and reports the USB link. The AgentView
policy image must be `480x480`; the wrist image must be `640x480` with the
current config.

For a read-only LAN preview when no app owns the cameras:

```bash
just camera-web
```

Open `http://<robot-lan-ip>:8088`. The full AgentView contains a red rectangle
showing the exact square region saved and sent to policies. Stop standalone
preview before Teleop:

```bash
just camera-web stop
```

The preview has no login or encryption. Keep it on the LAN and do not
port-forward it.

Optional EEF hardware acceptance moves the arm:

```bash
just eef-test
```

Run it only with a powered arm and clear workspace. A failed partial startup
must be followed by `just stop` before retrying.

## 3. Reset the Collection Rig

This command moves both devices:

```bash
just reset
```

It uses `configs/poses/a1_so100_collection_start.toml`, moves the A1 through the
staged jointTracker/relay path, moves the SO leader to its tracked position,
closes both grippers, disables leader torque, and stops the runtime afterward.

Do not manually move either arm during reset. If reset fails, run `just stop`,
inspect the error, restore a safe physical position, and retry only after the
cause is understood.

## 4. Optional Teleop Acceptance

This command moves the A1 from leader input but does not start collection:

```bash
just teleop-test
```

Check all six joint directions and the continuous gripper over a small range.
The mapping is relative to both startup poses. Unknown leader action keys and
non-finite samples fail instead of guessing an ordering.

Watch logs if needed:

```bash
just logs
```

Stop before proceeding:

```bash
just stop
```

## 5. Record New Episodes

Use a clean experiment identity whose dataset config will be tracked:

```bash
just reset
just teleop EXPERIMENT
```

At the task prompt, enter the natural-language task once. At the episode prompt:

- press `Enter` to start;
- while recording, press `Enter` to request save;
- use `d` + `Enter` to discard;
- use `q` + `Enter` to quit;
- use `Ctrl+C` for immediate stop.

The current default records at 30 FPS:

- cropped `480x480` AgentView RGB;
- full `640x480` wrist RGB;
- EEF pose, six named A1 joints, and normalized gripper state;
- six absolute joint targets and normalized gripper action;
- camera sequence numbers and monotonic sample times;
- reproducibility metadata for configs, topics, cameras, control path, and
  freshness limits.

Gripper values are continuous `0..1` and map exactly once to the tracked
physical `0..104 mm` A1 stroke. The SO leader's usable `0..53.16` input range
maps to the same normalized interval, so leader full-open, collected action
`1.0`, and an A1 target of `104 mm` have one meaning. Measured full-open feedback
is about `103.8 mm`. `/gripper_stroke_host` is required feedback.

During every frame, collection requires fresh joint, EEF, gripper-feedback,
joint-action, and paired-camera samples. A stale stream or excessive camera
skew aborts and removes the partial episode.

Enter-to-save is a durability boundary. The recorder first validates the staged
files and rejects a joint action discontinuity beyond
`collection.max_joint_action_step_rad`. Only a complete episode is atomically
renamed into:

```text
data/raw/<experiment>/episode_NNN_timestamp/
```

Rejected saves print the exact failure and reuse the index. With
`collection.auto_reset_after_save = true`, successful saves reset both devices
before the next episode. With `collection.auto_reset_after_discard = true`, a
user discard or quality-check rejection removes the staged data, resets both
devices, and only then permits retrying the same episode index. Quit does not
reset because collection does not continue.

Formal collection writes only `galaxea_a1_teleop_raw_v3`. Do not mix manually
edited episodes or previous raw schemas into the experiment directory.

## 6. Inspect a Collection

After quitting and running `just stop`, inspect without modifying the data:

```bash
find data/raw/EXPERIMENT -maxdepth 2 -type f | sort | head
```

Each episode must have `metadata.json`, `frames.csv`, `cam0/`, and `cam1/`.
`cam0_depth/` exists only when depth was intentionally enabled in the System
config. Hidden sibling staging directories indicate an interrupted commit;
inspect them before removal rather than allowing a new run to ignore them.

## 7. Convert to Training Data

Create or update `configs/datasets/<experiment>.toml` with the raw source,
base LeRobot output, three derived output locations, archives, repo IDs, and
URDF links. Collection state/action names and hardware observation shapes are
not copied into this file; they derive from its referenced Teleop config and
that config's System reference.

Run the complete pipeline:

```bash
just convert EXPERIMENT
```

The command performs:

```text
current raw v3
  -> base LeRobot v3
       -> LingBot EEF continuous v3
            -> LingBot EEF continuous v2.1
       -> ACT/joint continuous v3
```

It rejects old or incomplete raw schemas, task mismatches, non-finite values,
incorrect camera shapes, mismatched episode contracts, and missing files.
Outputs and archives use sibling staging paths; a failed conversion preserves
the previous complete target. Overwrite policy is tracked in the dataset TOML,
not passed on the CLI.

## 8. Failure Recovery

First stop all repository-owned resources:

```bash
just stop
```

Then diagnose the narrow layer:

- serial or device missing: `just hardware`;
- camera missing/stale/slow: `just cameras` and check USB topology;
- teleop process exited: `just logs`;
- model missing: `just models`;
- configuration error: `just check`;
- raw conversion error: inspect the named episode and its metadata/files; do
  not weaken validation to accept damaged data.

Status `64` alone is accepted as the observed idle ECU-to-ACU timeout. Codes
with additional bits, such as `68`, remain faults. Never start two trackers,
drivers, camera owners, or command publishers for the same hardware.

## 9. Policy Deployment

New policy weights must be registered under the ignored local model registry:

```bash
just model-link act-a1-agentview-square /path/to/act_checkpoint
just model-link lingbot-a1-agentview-square /path/to/lingbot_checkpoint
just models
```

The checkpoints must use the same AgentView square crop and continuous gripper
contract as the new data. Update the owning deployment TOML and review its
model-specific normalization before setting `deployment_ready = true`.

ACT and LingBot remain dry-run until `execution.execute = true` is explicitly
committed in their separate configs:

```bash
just act
tmux attach -t act-a1

just lingbot
tmux attach -t lingbot-a1
```

Run only one live app at a time, and use `just stop` when switching.
