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
                               -> Web preview / ROS 1 Foxglove telemetry
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
and ROS 1 compressed-image mirrors are side branches, never inference inputs.

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

## Remaining migration

This is not a complete removal of ROS 1. The checked-in vendor `signal_arm`
package uses catkin/roscpp/rospy and vendor binaries. The SDK, trackers, safe
relay, feedback adapters, operator services and existing Foxglove endpoint remain
on the reviewed ROS 1 control path. Do not replace them with unchecked direct
ROS 2 command publishers. Native A1 driver support and ros2_control hardware
interfaces require separate validation before replacing this boundary.

The collector still reads current robot state/action after selecting a camera
pair. Historical state interpolation and action-time semantics remain work;
MCAP camera timestamps do not automatically fix that alignment. Existing
dataset provenance prevents silently resuming a dataset with a changed capture
contract. Use a new experiment when the collector reports a provenance mismatch.

Native ROS 2 Foxglove telemetry/operator services are also pending. A generic
ros1_bridge is not assumed to work in the Jazzy container: official Ubuntu
24.04 does not supply ROS 1 and A1 custom messages need build-time mappings.
The existing read-only camera boundary provides a working transition without
introducing a second physical camera owner.

## Upstream references

- [RealSense ROS driver](https://github.com/realsenseai/realsense-ros)
- [ROS 2 ApproximateTimeSynchronizer](https://docs.ros.org/en/ros2_packages/jazzy/api/message_filters/doc/Tutorials/Approximate-Synchronizer-Cpp.html)
- [rosbag2](https://github.com/ros2/rosbag2)
- [ROS 1 bridge platform constraints](https://index.ros.org/p/ros1_bridge/)
