# Bag-first collection

Approved direction: preserve original ROS 2 messages in MCAP, then reproducibly
align and export canonical LeRobot v3. Raw recording is authoritative and is
never deleted on discard, interruption, or failed conversion. Existing staged
motion, operator input and reset gates remain unchanged.

Use rosbag2's Jazzy recorder and offline reader in the existing ROS 2 image.
Record raw camera images plus standard robot observations and joint/gripper
command mirrors. No reverse ROS bridge and no offline ROS publisher. Add the
existing named joint target topic to the one-way bridge.

An episode manifest records identity, exact task, config snapshots/digests,
operator start/stop wall-clock boundaries and disposition. Start is announced
only after recorder subscriptions are ready. Saved episodes automatically
export after finalizing the bag; an offline CLI can retry conversion later.

Align on an integer-nanosecond grid at configured collection FPS. Retain camera
source timestamps, bag receive timestamps, robot interpolation endpoints,
command sample times, and image reuse in per-episode Parquet timing records.
Use source time where valid, explicitly identify receive-time fallback for
unstamped robot messages, and reject unstamped cameras or clock reversals.
State interpolates only inside valid bounded intervals; action is last-known
command, never future interpolation. Camera pairs respect the configured skew
and freshness limits. Leading stillness trims the aligned grid without
compressing internal gaps. Retain LeRobot's nominal timestamp for compatibility.

Timing records and bag provenance commit atomically with every derived episode.
Old datasets remain readable, but cannot be silently appended with a different
capture contract. Dataset doctor validates the new timing graph. No vendor or
external submodule modifications and no tracked test files.
