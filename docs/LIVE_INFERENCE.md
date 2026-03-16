# A1 Live Inference

This document describes the full workflow for running A1 policy inference on the real robot.

## Prerequisites

- Serial port permissions (confirm the correct port):
  ```bash
  sudo chmod 777 /dev/ttyACM0 /dev/ttyACM1
  ```
  Or configure persistent udev rules (recommended): see `docs/SETUP_UDEV.md`.

- Python environment installed: see `docs/SETUP_ENV.md`.

## Launch Order

Five terminals are required. Start them in order:

### Terminal 1 — ROS master
```bash
just launch roscore
```

### Terminal 2 — A1 arm driver
```bash
just launch driver
```

### Terminal 3 — Camera server
```bash
just launch camera-server   # publishes camera frames over ZMQ (port 5558)
```

### Terminal 4 — A1 ZMQ bridge
```bash
just launch a1-server       # publishes joint state (port 5557), receives policy actions (port 5559)
```

### Terminal 5 — Policy server (WebSocket on port 8000)
```bash
just policy                        # uses default checkpoint: /home/eric/4999
just policy /path/to/checkpoint    # specify a different checkpoint
```

### Terminal 6 — ZMQ ↔ WebSocket bridge (starts the inference loop)
```bash
just zmq-bridge                                        # default prompt
just zmq-bridge --prompt "pick up the cup"             # custom prompt
just zmq-bridge --prompt "..." --action-chunk-size 3   # tune action chunk size
just zmq-bridge --step-mode --prompt "..."             # press Enter for each infer step
```

## Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--prompt` | `"swap the position of the marker and the yellow block"` | Language instruction |
| `--action-chunk-size` | `2` | Number of consecutive actions to execute before re-querying the policy |
| `--step-mode` | `false` | Manual stepping mode: press Enter to run one infer→publish step |
| `--host` / `--port` | `127.0.0.1` / `8000` | Policy server address |

## Debug Tools

```bash
just debug camera                            # dump frames the model actually receives to data/debug/
just teacher-forcing                         # offline teacher forcing on training data → trajectory.html
just openloop-rollout --policy-dir /home/eric/4999   # open-loop eval on processed data
```

## Common Issues

- **`Config 'xxx' not found`**: check the config registration at the top of `serve_policy_a1.py` and confirm `pi05_a1_single_arm` is injected.
- **`ROS master is not online`**: confirm Terminal 1 (`roscore`) is running.
- **No actions published**: check for ZMQ port conflicts (5557/5558/5559) with `ss -lntp | grep 555`.
- **Serial errors**: confirm the `ttyACM` number and permissions, or reload udev rules.
