# ROS 2 migration

## Implemented boundary

The production camera path uses ROS 2 Jazzy and the official RealSense driver,
`sensor_msgs/Image`, `cv_bridge`, `message_filters`, image transport plugins and
rosbag2/MCAP. The previous production Python SDK acquisition loop is removed.
SDK access remains for device discovery and exclusive hardware diagnostics.

```text
D455 / D405 -> official RealSense ROS 2 nodes
                   |-> raw images + calibration + metadata -> rosbag2 MCAP
                   |-> ApproximateTimeSynchronizer
                           -> atomic source-stamped camera pair
                               -> model/collection IPC boundary
                               -> paired video recorder
                               -> Web preview / native ROS 2 Foxglove telemetry
```

One System configuration owns the serials, stream settings, crop, freshness,
pair tolerance and ROS domain. Topic names, driver QoS policy and container
identity are implementation constants. Device-derived D405/D455 module names
are passed internally by the supervisor, not operator backend choices.

Both source timestamps are retained in protocol v3 and paired video timelines.
Monotonic timestamps now estimate source capture time in the host clock domain,
rather than the time a Python acquisition call completed. This estimate depends
on upstream RealSense global-time mapping; small numerical skew alone does not
prove synchronized exposure. Current runtime does not support simulated time.

The Unix boundary keeps ROS packages out of model Python environments. It does
not reopen devices, perform a second match or manufacture a common image stamp.
Both members must carry the same increasing pair sequence. The HTTP preview
and ROS 2 compressed-image previews are side branches, never inference inputs.

## Validation on 2026-09-18

Hardware: front D455 (firmware 5.15.1.55), wrist D405 (firmware 5.16.0.1), both
enumerated as USB 2.1. Driver/librealsense versions tested: 4.58.4 / 2.58.4.
Both streams successfully use 640x480 at 30 FPS with global-time mapping enabled.

- Configuration/architecture validation, Ruff, shell parsing and Foxglove
  extension build/lint are required through `just check`.
- Hardware-free timestamp validation exercised reversed arrival, duplicate,
  out-of-order, zero, future and stale timestamps, excessive skew, recovery,
  clock discontinuity and atomic IPC timestamp retention.
- A 30-second live probe with 20 ms tolerance yielded 847 pairs (28.23 Hz),
  source-stamp skew P50/P95/P99/max 10.44/18.41/19.79/19.99 ms, and mapped
  source age P50/P95/P99/max 43.56/52.16/54.16/62.78 ms. These are one-run
  observations, not exposure-accuracy or throughput guarantees.
- A subsequent 30-second run with reliable driver publication yielded 821
  pairs (27.36 Hz), skew P95 16.04 ms (max 18.52 ms), mapped age P95 49.34 ms
  (max 143.23 ms). Both transport loss and source phase can affect accepted
  pair rate; a 30 FPS source does not guarantee 30 accepted pairs per second.
- Policy observations retained the 480x480 front crop and 480x640 wrist image.
- The collection frame assembler accepted real synchronized camera images with
  synthetic robot state. This exercised the capture interface without starting
  a robot driver; it was not a live collection episode or a motion test.
- A paired H.264 run wrote 92 frames per stream and retained source clocks in
  the timeline. Pausing the camera container caused live observations to reject
  stale images; unpausing recovered fresh pairs.
- Initial best-effort MCAP recording reported transport loss. Reliable driver
  publication resolved loss in the subsequent short capture. That archive
  contained 142 images per camera; isolated native rosbag2 replay delivered all
  284 images. Offline standard synchronization matched 141 pairs with maximum
  source-stamp skew 12.13 ms. Replay waited for subscriber discovery before
  resuming and waited for reliable acknowledgements before exit.
- ROS 1 Python import readiness and the existing TRAC-IK build receipt passed.

Local raw captures, video and validation receipts live under `outputs/ros2/` and
are not committed. No robot driver was started and no motion command was sent.

## Native observation and operator plane

The persistent Foxglove endpoint is now upstream `ros-jazzy-foxglove-bridge`.
The URL and checked-in layout are unchanged. A native `rclpy` node owns camera
previews, combined diagnostics, sanitized workflow status and the eight exact
`std_srvs/srv/Trigger` collection services. Its callbacks retain the existing
phase, declared-action, run-id and input-revision gates through Operator Session.
No ROS 2 service publishes motion commands directly.

```text
Vendor ROS 1 state / command messages
  -> read-only validating legacy adapter -> standard display messages
  -> upstream ros1_bridge (ROS 1 -> ROS 2 only)
  -> Jazzy diagnostics / Foxglove
Atomic camera IPC + private Operator Session
  -> native Jazzy previews / workflow heartbeat / gated services -> Foxglove
```

The bridge runs in Ubuntu 24.04/Jazzy with a separately source-built ROS 1 client stack.
Noble has no supported ROS 1 binary distribution: the exact Noetic releases
are listed in `docker/ros1-bridge/ros1.rosinstall` and built only inside this
compatibility image. The ROS 2 side matches Jazzy throughout, avoiding the
Humble/Jazzy graph-message ABI mismatch found during validation. Its upstream revision is
pinned in `docker/ros1-bridge/Dockerfile`. A small C++ entrypoint composes only
`create_bridge_from_1_to_2` factories using the exact System-derived topic list.
No dynamic wildcard bridge, bidirectional parameter bridge, custom A1 message
mapping, ROS 1 publisher, or service bridge is installed into the runtime path.
It forwards measured joints, EEF pose, TF, relay status, validated command/gripper
mirrors and legacy diagnostics. `/tf_static` uses transient-local ROS 2 durability.
Native ROS 2 logs are visible; old vendor ROS 1 `/rosout` is not bridged.

Original robot mirror header stamps are preserved, including zero if the vendor
source is unstamped. Forwarding does not manufacture a sampling timestamp.
Loss of the legacy diagnostics stream is displayed as an error; absent arm data
is never synthesized. Preview images retain each member's source ROS timestamp.

`ros2.domain_id` now owns the whole ROS 2 graph; the old camera-specific key is
rejected. All managed ROS 2 processes use local-host discovery. The vendor mesh
package is registered in an ephemeral ament index inside the Foxglove container;
only the configured URDF and its exact mesh URIs are retrievable. No vendor code
or libraries are sourced into Jazzy. Client publication and parameter access
remain disabled and only the configured collection services are exposed.

`just ros2-setup` builds both the Jazzy and bridge images. `just cameras start`
ensures the whole observation stack. `just foxglove restart` restarts only
observation services; `just stop` preserves them. `just cameras stop` closes the
camera and all observation containers, retaining a shared ROS master if another
managed runtime still uses it. Startup checks actual diagnostic delivery on both
sides of the bridge before declaring readiness.

## Observation validation on 2026-09-18

- Built the Jazzy camera/Foxglove image, source-built Noetic client/Jazzy bridge,
  and the vendor image with the obsolete ROS 1 Foxglove build removed.
- A 20-second live subscription received 511/513 raw front/wrist images and
  136 previews per camera. The compared preview stamps matched raw source
  stamps exactly; maximum observed preview source age was 130.8 ms.
- The actual Foxglove SDK WebSocket delivered both JPEG streams (640x480),
  combined diagnostics and exactly eight collection services. Fetching the
  configured URDF and all nine meshes succeeded (7,887,643 bytes total);
  fetching an unlisted local file was rejected.
  Native bridge 3.5.0 requires a client offering `foxglove.sdk.v1`.
- An actual CDR Trigger call with no Operator Session returned `success=false`.
  Hardware-free callback validation covered accepted run/revision forwarding,
  consumed gates, inactive sessions, wrong workflow/phase and missing run IDs.
- In a separate network namespace and ROS master, synthetic JointState values,
  ordering and nanosecond source stamps survived the upstream bridge. A late
  subscriber received retained static TF. The ROS 1 master reported zero
  publishers owned by the one-way bridge.
- Pausing the bridge produced a stale legacy-diagnostics error and unpausing
  restored delivery. Pausing cameras produced stale-camera diagnostics and
  unpausing restored fresh images. No robot driver or live command publisher
  was started; synthetic messages were confined to the isolated graph.

- A final six-second bounded MCAP run committed 165 front and 166 wrist raw
  images. Normal stop preserved observation services; explicit camera stop
  removed them and a clean restart passed both diagnostic delivery probes.
- Final `just check`, ROS 1 Python readiness and `git diff --check` passed.

These are observation/transport checks, not live collection, policy inference,
physical motion or hardware exposure-synchronization acceptance.

## Powered hardware smoke check on 2026-09-18

With operator authorization and power on, the runtime doctor received all seven
named joints, EEF feedback and motor state. Real robot joint feedback also
arrived on ROS 2 `/joint_states_host`. The relay was initially locked.

The first motion attempt stopped before enabling motion: the TRAC-IK build
receipt referenced a deleted Docker image. After stopping the execution stack,
`just trac-ik-setup` rebuilt the worker. Build verification now checks that the
receipt's exact image is accessible, in addition to source and binary hashes.
An ad hoc check confirmed actionable rejection of Docker inspection failure,
timeout and missing executable; the real rebuilt image passed verification.

The existing guarded EEF nudge workflow then completed all six 3 cm targets
through TRAC-IK, the staged tracker and the fail-closed relay. Measured movement
along the requested axis after the script's one-second wait was:

| Direction | Measured displacement (cm) |
| --- | ---: |
| x+ | 2.55 |
| x- | -2.77 |
| y+ | 2.42 |
| y- | -2.64 |
| z+ | 3.00 |
| z- | -2.96 |

These feedback measurements establish a working motion path, not external
metrology or settled tracking accuracy. The workflow exited successfully and
`just stop` removed the driver, tracker and command relay while preserving
cameras and observation services. Final `just check` and `git diff --check`
passed. This check did not exercise gripper motion, an end-to-end collection
episode, policy inference or camera/robot historical timestamp alignment.

## Remaining migration

This is not complete removal of ROS 1. The checked-in vendor `signal_arm`
package uses catkin/roscpp/rospy and vendor binaries. The SDK, trackers, safe
relay, feedback adapters and robot service retain the reviewed ROS 1 control
path. Native A1 driver support and ros2_control hardware interfaces require
separate validation before replacing this boundary. The observation bridge is
not a motion migration and must never gain a reverse command route.

The collector still reads current robot state/action after selecting a camera
pair. Historical state interpolation and action-time semantics remain work;
MCAP camera timestamps do not automatically fix that alignment. Existing
dataset provenance prevents silently resuming a dataset with a changed capture
contract. Use a new experiment when the collector reports a provenance mismatch.

## Upstream references

- [RealSense ROS driver](https://github.com/realsenseai/realsense-ros)
- [ROS 2 ApproximateTimeSynchronizer](https://docs.ros.org/en/ros2_packages/jazzy/api/message_filters/doc/Tutorials/Approximate-Synchronizer-Cpp.html)
- [rosbag2](https://github.com/ros2/rosbag2)
- [ROS 1 bridge platform constraints](https://index.ros.org/p/ros1_bridge/)
- [ros1_bridge C++ API](https://github.com/ros2/ros1_bridge/blob/master/include/ros1_bridge/bridge.hpp)
- [Foxglove ROS 2 bridge](https://github.com/foxglove/foxglove-sdk/tree/main/ros/src/foxglove_bridge)
