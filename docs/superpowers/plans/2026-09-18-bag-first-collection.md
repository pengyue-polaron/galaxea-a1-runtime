# Bag-first collection implementation plan

> Execute inline on the existing user-requested feat/ros2-runtime branch, following executing-plans. User has approved implementation; no additional design gate is needed.

**Goal:** Authoritative MCAP collection and reproducible timestamped LeRobot export.
**Architecture:** Existing Jazzy recorder captures exact topics; a network-isolated rosbag2 reader feeds a pure alignment layer and existing atomic LeRobot writer. The collector keeps its guarded interaction and resets.
**Tech Stack:** ROS 2 Jazzy, rosbag2/MCAP, msgpack, NumPy, Parquet, LeRobot v3.
**Spec:** docs/superpowers/specs/2026-09-18-bag-first-collection.md

## Constraints

No direct motor publishing; no reverse bridge; preserve raw bags on every outcome.
Use existing config owners and safety limits. No new tracked tests or submodule edits.

## Tasks

- [x] Add bag topic contract, capture process lifecycle and episode manifests in apps/teleop/bag_capture.py; bridge the named target topic in observability.py. Validate native recorder service readiness and graceful finalization.
- [x] Add network-isolated rosbag2 reader worker and host bridge in apps/teleop/bag_reader.py. Validate actual MCAP decoding, source/receive timestamp distinction, type contracts and image channel/stride handling.
- [x] Add pure time alignment in collection/bag_alignment.py: integer grid, bounded nearest camera pairs, state interpolation and causal action hold. Validate drift, missing/zero stamps, clock reversals, gaps and image reuse using throwaway probes.
- [x] Add offline converter apps/teleop/bag_export.py and Justfile entry. Add atomic timing/provenance attachments and inspection to lerobot/direct_recording.py. Validate numeric/video roundtrip, append, duplicate source rejection and interrupted conversion preservation.
- [x] Integrate bag capture/export into guarded collector_episode.py and update provenance preflight. Keep readiness, operator decisions and reset policy. Remove obsolete direct capture loop only where no longer used.
- [x] Update RUNBOOK, ARCHITECTURE, ROS2 and SAFETY with commands, semantics and acceptance limitations. Run just check, git diff --check, isolated synthetic MCAP and live-camera export probes. Commit and push reviewed results.
