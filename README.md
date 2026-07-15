# Galaxea A1 Runtime

Fail-closed runtime, SO leader teleoperation, data collection, dataset
conversion, and policy deployment for a real Galaxea A1 arm. The repository
uses LeRobot v0.6 and LeRobotDataset v3 as its current baseline.

The arm may be powered and reachable while this repository is open. Treat every
ROS publish path as live hardware. Read [Safety](docs/SAFETY.md) before running
motion commands.

## Operator Flow

Static checks do not move the robot:

```bash
just setup       # first-time Python environment
just check       # config, lint, shell, and unit tests
just hardware    # enumerate configured hardware; no motion
just cameras     # camera snapshots and FPS probe; no motion
```

Normal teleoperation and collection:

```bash
just stop
just reset
just teleop banana_in_the_plate
```

The recorder prompts once for a task, then uses this episode loop:

- `Enter`: start recording; while recording, save and validate the episode.
- `d` + `Enter`: discard the current episode, reset both devices, then retry
  the same episode index.
- `q` + `Enter`: quit.
- `Ctrl+C`: stop immediately.

Successful saves are atomic. A save is rejected if samples are stale, files are
incomplete, or a joint action jump exceeds the tracked collection threshold.
The rejected episode is deleted and its index is reused. With the default
config, a successful save resets both arms before the next episode.

Convert the newly collected raw experiment with its tracked dataset config:

```bash
just convert banana_in_the_plate
```

That one command atomically builds, in order:

1. raw teleop v3 -> base LeRobot v3;
2. LingBot EEF continuous v3;
3. LingBot-compatible EEF continuous v2.1 export;
4. ACT/joint continuous v3.

Only the current `galaxea_a1_teleop_raw_v3` input contract is supported. Old
raw schemas are intentionally not migrated. Each experiment needs a tracked
`configs/datasets/<experiment>.toml` before conversion.

Useful lifecycle commands:

```bash
just teleop-test
just camera-web
just eef-test
just act
just lingbot
just logs
just stop
```

`just eef-test`, `just reset`, `just teleop-test`, `just teleop`, `just act`,
and `just lingbot` may command live hardware. Power and position the arm safely
first. ACT and LingBot are currently fail-closed until new checkpoints are
registered and their deployment configs are explicitly marked ready.

See [Runbook](docs/RUNBOOK.md) for the full preflight, recording, conversion,
and recovery procedure.

## Safe Control Paths

EEF applications use:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

Teleop and ACT joint policies use:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

Normal gripper commands use:

```text
/a1_gripper_target
  -> safe_arm_command_relay.py
  -> /gripper_position_control_host
```

The relay starts `LOCKED`, requires fresh validated inputs, and is the only
normal publisher to host command topics. Direct host publishing is reserved for
explicit debug after `just stop`.

## Configuration Ownership

| Path | Owns |
| --- | --- |
| `configs/system/a1.toml` | physical devices, ROS topics, cameras, joint/EEF safety, relay, 0-104 mm physical gripper stroke |
| `configs/teleop/a1_so100.toml` | SO leader mapping, collection state/FPS/quality behavior, reset reference |
| `configs/poses/` | tracked reset target values and reset motion behavior |
| `configs/datasets/` | raw source and generated dataset/package locations; reference to its Teleop collection contract |
| `configs/deployments/` | model registry paths and inference/execution semantics |

App configs reference the system config rather than restating physical values.
Normal live CLIs accept lifecycle commands or an experiment identity, not
per-run overrides for hardware or safety settings.

## Observation and Action Contract

- AgentView D455: capture `640x480`, then use the configured
  `x=103, y=0, width=480, height=480` square crop for collection and inference.
- Wrist D405: full `640x480` RGB frame.
- Web preview: full AgentView with the actual recorded/policy ROI outlined in
  red; it does not save the overlay.
- Default state: EEF pose + six arm joints + continuous gripper.
- Action: six absolute joint targets + continuous gripper.
- Gripper: normalized `0..1` everywhere above hardware, mapped exactly once to
  the system-owned `0..104 mm` physical stroke. The SO leader's tracked
  `0..53.16` usable input range maps to that same `0..1` contract; there is no
  binary threshold or second scale rewrite.
- Feedback: `/gripper_stroke_host`; the seventh joint-state value is never
  reinterpreted as millimeters.
- Runtime action-step protection is intentionally disabled in the system
  config. Enter-to-save collection continuity validation remains enabled in
  the teleop config.

The read-only camera preview listens on `0.0.0.0:8088` with no login:

```text
http://<robot-lan-ip>:8088
```

It is LAN-only, unencrypted, and contains no robot-control endpoint. Do not
port-forward it. Only one process may own each RealSense; standalone preview
is for use when Teleop, ACT, and LingBot are not already using the cameras.

## Repository Layout

```text
galaxea_a1_runtime/
  configuration/   typed strict config loading
  hardware/        cameras, EEF helpers, and preview
  runtime/         relay, ROS feedback, doctors
  apps/            Teleop, ACT, and LingBot implementations
  collection/      raw episode schema and validation
  teleop/          SO leader adapter and joint mapping
  lerobot/         raw conversion and deterministic packaging
  policies/        shared policy action contracts
scripts/
  runtime/         app-agnostic ROS/driver/tracker/relay lifecycle
  apps/            thin operator entrypoints by application
configs/           tracked system, app, pose, dataset, and deployment contracts
models/            ignored local deployment registry
data/              ignored raw, processed, and exported datasets
assets/cad/         versioned mechanical assets
third_party/        pinned vendor snapshots
```

Inference configs reference weights only through the ignored local
[`models/`](models/README.md) registry. Never commit model weights or patch the
vendored LeRobot tree for A1-specific behavior.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Runbook](docs/RUNBOOK.md)
- [Safety](docs/SAFETY.md)
- [Environment setup](docs/SETUP_ENV.md)
- [udev and serial setup](docs/SETUP_UDEV.md)
- [Model registry](models/README.md)
- [Third-party policy](third_party/README.md)
- [SO-100 leader CAD](assets/cad/so100_leader/README.md)
