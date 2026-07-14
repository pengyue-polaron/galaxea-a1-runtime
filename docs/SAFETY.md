# Safety

The arm may be powered and reachable while this repo is open. Treat ROS publish
paths as live hardware.

The camera web preview is deliberately outside every ROS control path. It
exposes only read-only HTML, JPEG/MJPEG, and health endpoints on the LAN. It has
no login and uses unencrypted HTTP, so do not port-forward
it to the public Internet; never add motion-enable or command endpoints to the
camera service.

## Required Path

Normal EEF apps must publish EEF targets only:

```text
/a1_ee_target
  -> isolated eeTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

Do not publish directly to `/arm_joint_command_host` from app or inference code.

Teleop collection and ACT joint-state inference are the normal joint-space
apps. They still do not publish host commands directly:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

## Runtime Gates

- Relay starts `LOCKED`.
- An app must explicitly enable `/a1_arm_motion_enable`.
- Relay requires fresh joint feedback, staged tracker command, and motor status.
- First staged command must align with current joint feedback within `0.05rad`.
- After validation, relay forwards staged tracker commands unchanged.
- Normal gripper commands follow `/a1_gripper_target -> relay ->
  /gripper_position_control_host`. The relay rejects non-finite or out-of-range
  targets, requires gripper status `0` or idle `64`, and forwards only fresh
  targets while `ACTIVE`.
- Relay does not apply hidden joint tracking-error or joint-speed clamps.
- Motor status code `64` alone is accepted as observed idle timeout; additional
  error bits still fault.
- Teleop starts from the current A1 joint pose and maps relative SO leader
  motion onto A1 joint targets before arming the relay.
- Teleop target joint limits are explicit bridge arguments and are checked by
  `just check`.
- ACT starts dry-run by default. When execution is enabled, it first commands
  the current joint feedback target through jointTracker and waits for staged
  alignment before arming the relay.

## Action Behavior

- `GalaxeaA1Robot` has no implicit ROS implementation. A caller must inject an
  explicit `A1HardwareIO`; all supported live control is owned by the managed
  Teleop, ACT, LingBot, reset, and EEF app paths below.
- Collected gripper state and action are continuous normalized values. Teleop,
  dataset conversion, ACT, and LingBot use the same linear mapping from `0..1`
  into the physical range in `configs/system/a1.toml`.
- Collection and ACT require fresh `/gripper_stroke_host` feedback. They do not
  fall back to the unit-ambiguous seventh `/joint_states_host` value.
- LingBot workspace bounds apply to outgoing targets, not feedback state.
- LingBot orientation defaults to `hold-current`.
- LingBot execution settings live in `configs/deployments/lingbot_va.toml`;
  avoid per-run hidden flags.
- The LingBot KV cache records tracker commands because its training
  action is a commanded episode-relative EEF target. Measured EEF feedback is
  still used for freshness checks, workspace-relative diagnostics, and camera
  context.
- LingBot bridge exit is guarded: normal completion, errors, and `Ctrl-C` stop
  the A1 runtime and policy server.
- ACT execution settings live in `configs/deployments/act_joint.toml`; the
  tracked default is `execution.execute = false`.
- ACT action-step jump rejection is explicitly disabled by
  `configs/system/a1.toml [joint_safety.action_step_guard_enabled]`. Finite
  values and absolute joint limits are still enforced before execution.
- Teleop gripper state/action is continuous normalized `0..1`, mapped to the
  unique `0..100 mm` range in `configs/system/a1.toml`.
- Teleop, ACT, LingBot, reset, and EEF tools publish only the staged gripper
  target. The relay owns the hardware command topic.

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
