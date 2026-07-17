# Safety

This document is authoritative for live control paths, relay gates, A1 status
handling, and direct hardware debug. The arm may be powered and reachable while
the repository is open; treat every ROS publisher as live hardware.

## Managed control paths

Normal EEF-policy applications solve their reviewed Cartesian target into a
named joint target through the tracked first-party URDF IK contract, then
publish only:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

Before enabling motion, the bridge publishes the current named joints as a hold
and waits for a fresh staged command aligned with feedback. IK rejects
non-convergence, joint-limit violations, non-finite results, and solutions whose
maximum joint delta exceeds the System-owned limit.

Teleop publishes joint targets only:

```text
/arm_joint_target_position
  -> isolated jointTracker
  -> /arm_joint_command_a1_staged
  -> safe_arm_command_relay.py
  -> /arm_joint_command_host
```

Normal gripper applications publish only:

```text
/a1_gripper_target
  -> safe_arm_command_relay.py
  -> /gripper_position_control_host
```

App and policy code must never publish host commands directly. The relay is the
only normal owner of both host command topics.

## Relay gates

- The relay starts `LOCKED`; an app must explicitly enable
  `/a1_arm_motion_enable`.
- Joint feedback, staged tracker commands, gripper targets, and motor status
  must be fresh.
- Every named driver vector is reordered against configured joint names and
  must have the expected DOF and finite values. Gains must be non-negative and
  control mode must be allowed by System config.
- The first staged command must align with current joint feedback within the
  configured startup tolerance.
- Absolute joint, workspace, and physical gripper limits come from System
  config. No hidden tracking-error, speed, or action-step clamp is applied.
- The complete episode-relative model pose is always composed into the absolute
  IK target; its quaternion is never replaced with current feedback.
- Verbose action logging reports IK residuals and maximum joint deltas when
  enabled by the deployment; the tracked Cartesian tolerance is 2 mm and the
  maximum single-joint IK solution delta from fresh feedback is 1.50 rad.
- Gripper forwarding occurs only while `ACTIVE` and healthy. State/action above
  hardware is continuous `0..1`, mapped exactly once to physical stroke;
  `/gripper_stroke_host` is the only feedback source.
- Normal completion, errors, and `Ctrl+C` must disable motion and stop the
  owning runtime.

## Observed A1 status semantics

Status `64` alone is the observed idle ECU-to-ACU timeout and is accepted as
non-blocking. Any additional fault bits remain blocking; for example, `68` is
not equivalent to idle `64`.

The only tracked compatibility exception is gripper bit 3, Position Jump
(`8`). The relay may ignore it only through
`relay.gripper_ignored_error_mask`. Idle bit 6 remains acceptable; every other
additional gripper bit latches `FAULT`.

## Resource ownership

- Parse and validate all tracked configuration before opening hardware or
  creating processes.
- One process owns each driver, tracker, camera, serial bus, and command
  publisher.
- Embedded preview and AgentView recording consume the policy app's existing
  latest-frame reader; they never open competing camera handles. Bridge cleanup
  finalizes recording before closing the owned cameras.
- After partial startup failure, run `just stop` before retrying.
- The configuration-independent shutdown fallback may stop only marked
  repository-owned containers and tmux sessions.
- The camera preview is read-only LAN HTTP/MJPEG. It has no authentication or
  encryption and must not be port-forwarded or gain control endpoints.

## Direct debug

Direct host publishing is reserved for explicit hardware diagnosis. Stop every
managed runtime first:

```bash
just stop
```

The normal acceptance test uses the same IK, jointTracker, and relay route as
the model bridges:

```bash
just eef-test
```

For explicit diagnosis of the vendor eeTracker itself, the headless debug
launch remaps that isolated tracker back to the official host topic:

```bash
roslaunch /workspace/scripts/runtime/ee_tracker_staged.launch \
  staged_command_topic:=/arm_joint_command_host \
  joint_states_topic:=/joint_states_host \
  target_topic:=/a1_ee_target \
  ee_pose_topic:=/end_effector_pose \
  tracker_node:=/eeTracker_demo_node
```

Mount the repository read-write in the debug container because `mobiman` may
write generated CppAD files. Do not start the managed relay or another tracker
at the same time. The tracker is MPC/IK-style and may couple axes or under-track
small Cartesian targets; never assume a published EEF target was reached.

`/end_effector_pose` is feedback in `base_link`; `/a1_ee_target` is a
`geometry_msgs/PoseStamped` command accepted in `world`. The managed launch
currently supplies an identity `world -> base_link` transform.

Explicit direct gripper checks, also only after `just stop`, use the physical
stroke from System config. With the current rig that range is `0..104 mm`:

```bash
rostopic pub /gripper_position_control_host signal_arm/gripper_position_control \
  "{header: {stamp: now}, gripper_stroke: 104.0}"
rostopic pub /gripper_position_control_host signal_arm/gripper_position_control \
  "{header: {stamp: now}, gripper_stroke: 0.0}"
```

Useful read-only ROS inspection inside the runtime environment:

```bash
rostopic echo -n1 /end_effector_pose
rostopic echo -n1 /joint_states_host
rostopic echo -n1 /arm_status_host
rostopic info /a1_ee_target
rostopic info /arm_joint_command_host
```

## Static disclosure

Print current safety settings without opening hardware:

```bash
.venv/bin/python -m galaxea_a1_runtime.cli safety-report
.venv/bin/python -m galaxea_a1_runtime.cli safety-report --json
```
