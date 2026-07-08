# Safety

The arm may be powered and reachable while this repo is open. Treat ROS publish
paths as live hardware.

## Required Path

Normal apps must publish EEF targets only:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay_v2.py
  -> /arm_joint_command_host
```

Do not publish directly to `/arm_joint_command_host` from app or inference code.

Teleop collection is the one normal joint-space app. It still does not publish
host commands directly:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay_v2.py
  -> /arm_joint_command_host
```

## Runtime Gates

- Relay starts `LOCKED`.
- An app must explicitly enable `/a1_arm_motion_enable`.
- Relay requires fresh joint feedback, staged tracker command, and motor status.
- First staged command must align with current joint feedback within `0.05rad`.
- After validation, relay forwards staged tracker commands unchanged.
- Relay does not apply hidden joint tracking-error or joint-speed clamps.
- Motor status code `64` alone is accepted as observed idle timeout; additional
  error bits still fault.
- Teleop starts from the current A1 joint pose and maps relative SO leader
  motion onto A1 joint targets before arming the relay.
- Teleop target joint limits are explicit bridge arguments and are checked by
  `just check`.

## Action Behavior

- Generic `GalaxeaA1Robot` forwards LeRobot EEF deltas unchanged by default.
  Optional `RuntimeConfig.safety` delta limits must be set explicitly.
- Generic ROS1 adapter needs live `/end_effector_pose` before arm motion.
- Generic ROS1 adapter rejects `joint_absolute`.
- Generic gripper input must be normalized `0..1`; direct adapter misuse raises
  instead of silently clipping.
- LingBot workspace bounds apply to outgoing targets, not feedback state.
- LingBot orientation defaults to `hold-current`.
- LingBot execution settings live in `configs/inference/lingbot_va_a1.toml`;
  avoid per-run hidden flags.
- LingBot gripper mapping is linear: normalized `0..1 -> 0..60mm`.
- LingBot waits for relay `ACTIVE` before gripper publish because the gripper
  topic is independent of the arm relay.

## Direct Debug

Direct debug is explicit and isolated. Stop safe runtime first:

```bash
just stop
```

Then use only documented direct-debug commands. Do not leave safe and direct
trackers running at the same time.

## Static Disclosure

Print current safety settings without touching hardware:

```bash
python -m galaxea_a1_runtime.cli safety-report
python -m galaxea_a1_runtime.cli safety-report --json
```
