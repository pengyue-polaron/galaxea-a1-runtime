# DragDataCoach Environment Setup

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.11–3.12 | |
| CUDA driver ≥ 12.6 | Required for PyTorch + RTX 5090 |
| ROS Noetic | `/opt/ros/noetic/setup.bash` |
| A1 SDK | Built at `third_party/A1_SDK/install/` |

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify: `uv --version`

## 2. Clone and initialise submodules

```bash
git clone <repo-url>
cd DataCoach
git submodule update --init --recursive
```

## 3. Sync the Python environment

```bash
uv sync
```

This creates `.venv/` and installs all dependencies including `torch==2.7.1+cu126`,
`openpi` (editable from `third_party/openpi`), `lerobot`, `hydra-core`, etc.

Verify:

```bash
just doctor
just which-python
```

## 4. External dependencies

Two paths in `Justfile` point to other machines. Update them if your setup differs:

```just
uv     := "/home/pengyue/.local/bin/uv"   # path to uv binary
openpi := "/home/eric/openpi"              # openpi source + checkpoints on Eric's machine
```

Policy checkpoints (e.g. `/home/eric/4999`) also live on Eric's machine and are
passed at runtime:

```bash
just policy /home/eric/4999
```

## 5. Verify CUDA + torch

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# Expected: 2.7.1+cu126  NVIDIA GeForce RTX 5090
```

## Runtime: source ROS before hardware commands

The `just launch` commands source ROS automatically. If you run scripts directly,
source first:

```bash
source /opt/ros/noetic/setup.bash
source third_party/A1_SDK/install/setup.bash
```

## Notes on PyTorch for RTX 5090

The Blackwell architecture (SM_100) is supported from PyTorch 2.6+. The standard
PyPI wheel `torch==2.7.1` ships with CUDA 12.6 support (`+cu126`) and works with
any driver ≥ 12.6. **Do not use the CUDA 11.x toolkit version to select the wheel —
check the driver version instead (`nvidia-smi`).**
