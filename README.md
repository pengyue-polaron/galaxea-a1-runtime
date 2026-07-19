<h1 align="center">Galaxea A1 Runtime</h1>

<p align="center">
  End-to-end teleoperation, LeRobot data collection, and policy deployment for
  the Galaxea A1 robot arm.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&amp;logoColor=white">
  <img alt="ROS 1 Noetic" src="https://img.shields.io/badge/ROS_1-Noetic-22314E?logo=ros&amp;logoColor=white">
  <img alt="Containerized ROS runtime" src="https://img.shields.io/badge/ROS_Runtime-Dockerized-2496ED?logo=docker&amp;logoColor=white">
  <img alt="LeRobot 0.6" src="https://img.shields.io/badge/LeRobot-0.6-FFD21E">
  <img alt="LeRobotDataset v2.1 and v3.0" src="https://img.shields.io/badge/LeRobotDataset-v2.1_%7C_v3.0-0A7BBC">
  <a href="https://arxiv.org/abs/2607.08283"><img alt="arXiv 2607.08283" src="https://img.shields.io/badge/arXiv-2607.08283-B31B1B?logo=arxiv&amp;logoColor=white"></a>
</p>

![Galaxea A1 follower and modified SO-101 leader](assets/images/a1-teleoperation-setup.png)

## What it does

- **Teleoperate** a Galaxea A1 follower with a modified six-axis SO-101 leader
  and continuous gripper control.
- **Collect** synchronized joint, EEF, action, gripper, and paired-camera data
  into atomically validated raw episodes.
- **Convert** current raw-v3 experiments into model-agnostic Joint and EEF
  LeRobotDataset v2.1 and v3.0 outputs.
- **Deploy** LingBot EEF and OpenPI pi0.5 EEF policies through
  isolated trackers and a locked, validating command relay.
- **Operate** collection, live evaluation, tracked batch plans, resets, and the
  two camera views from one localhost-only black-and-white control panel whose
  reusable core is isolated from A1-specific adapters.
- **Run on modern Ubuntu hosts** with ROS Noetic and the A1 SDK isolated inside
  a Focal-based Docker runtime—no native Ubuntu 20.04 or ROS installation
  required.

The current baseline is Python 3.12, ROS 1 Noetic, and LeRobot 0.6, with
first-party LeRobotDataset v2.1 and v3.0 conversion. Hardware, safety,
collection, and deployment behavior is owned by strict tracked configuration
rather than per-run overrides.

The current host is Ubuntu 22.04; Ubuntu 24.04 is also suitable as a Docker
host. Cameras, serial devices, and optional GPU acceleration remain host
resources passed into the containerized ROS execution layer.

## Quick start

Create the Python environment, build the ROS runtime image, and run the
hardware-free validation suite:

```bash
git submodule update --init --recursive
just setup
docker compose -f docker-compose.a1-noetic.yml build a1-noetic
just check
```

Continue with the [Runbook](docs/RUNBOOK.md) for hardware acceptance, reset,
Teleop collection, conversion, recovery, and deployment. Every command that can
move the robot is labeled there.

## Hardware setup

The reference setup pairs a modified SO-101 leader with a Galaxea A1 follower.
Its wrist view comes from an Intel RealSense D405 on a custom mount; collection
also uses the configured external AgentView camera.

<p align="center">
  <img src="assets/images/a1-d405-wrist-camera.png" width="520" alt="Intel RealSense D405 wrist camera mounted on the Galaxea A1">
</p>

Mechanical files are kept with the hardware they describe:

- [RealSense D405 wrist-camera holder](assets/cad/d405_wrist_camera_holder/README.md)
  — STEP source for the mount shown above.
- [Modified SO-101 leader parts](assets/cad/so100_leader/README.md) — printable
  STL files used by the leader arm.

## Research

The Galaxea A1 platform behind this repository was used for the real-robot
experiments reported in:

> **[TFP: Temporally Conditioned Memory-Fusion Policies for Visuomotor Learning](https://arxiv.org/abs/2607.08283)**<br>
> Yushen Liang, Yue Peng, Baosheng Jin, et al. · SemRob 2026 @ RSS 2026

## Repository map

| Path | Purpose |
| --- | --- |
| `galaxea_a1_runtime/` | first-party runtime, hardware, collection, policy, and conversion logic |
| `scripts/` | thin lifecycle and operator entrypoints |
| `configs/` | tracked system, data, backend, model, and deployment contracts |
| `docker/` | Ubuntu 20.04 / ROS Noetic execution environment |
| `assets/` | setup images and versioned mechanical files |
| `data/`, `outputs/`, `models/` | ignored local datasets, durable run results, and deployment weights |
| `external/` | pinned `embodied-ops`, A1 Robot, and A1 SO-Leader plugin submodules plus ignored local model checkouts |
| `third_party/` | pinned vendor snapshots; no A1-specific behavior |

## Embodied SDK and LeRobot plugins

The framework-neutral contracts and both hardware adapters are independently
versioned public repositories:

- [`embodied-ops`](https://github.com/pengyue-polaron/embodied-ops) defines
  capability, manifest, health, lifecycle, and backend-discovery protocols.
- [`lerobot-robot-galaxea-a1`](https://github.com/pengyue-polaron/lerobot-robot-galaxea-a1)
  provides the auto-discovered `galaxea_a1` LeRobot Robot and its pair-specific
  relative-anchor processor.
- [`lerobot-teleoperator-galaxea-a1-so-leader`](https://github.com/pengyue-polaron/lerobot-teleoperator-galaxea-a1-so-leader)
  provides the auto-discovered `galaxea_a1_so_leader` Teleoperator.

This repository supplies the `galaxea_a1_runtime` embodied-ops backend. It owns
ROS and attaches to an already supervised runtime; construction and `connect()`
never move the arm. The first command stages the current named-joint hold and
opens the locked relay only after fresh alignment. LeRobot plugins never publish
host motor topics directly.

The tracked Teleop application is the composition root for the modified
six-axis leader/A1 pair. It constructs both LeRobot plugins, derives the
relative-anchor processor from the strict Teleop and System configs, and runs
the standard observation → teleoperator action → processor → Robot ordering.
Generic LeRobot 0.6 CLI commands still install identity processors, so this
pair must be started through the tracked A1 Teleop workflow.

## Documentation

| Document | Covers |
| --- | --- |
| [Runbook](docs/RUNBOOK.md) | operator commands, expected results, and recovery |
| [Safety](docs/SAFETY.md) | live control paths, relay invariants, and direct debug |
| [Architecture](docs/ARCHITECTURE.md) | layers, configuration ownership, data contracts, and artifact layout |
| [Environment setup](docs/SETUP_ENV.md) | Python environment and dependency baseline |
| [udev setup](docs/SETUP_UDEV.md) | persistent A1 serial permissions and device alias |
| [Model registry](models/README.md) | immutable model artifacts and inference backends |
| [Agent guide](AGENTS.md) | constraints for code-writing agents |
