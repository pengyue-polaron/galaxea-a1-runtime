# A1 Live Inference (Reuse jolia uv/ckpt)

本文档用于在**当前仓库代码**下运行 A1 真机推理，并继续复用：
- `uv`: `/home/jolia/.local/bin/uv`
- checkpoint: `/home/pengyue/29000`

## 0) 前提

- 当前代码仓库：`/home/pengyue/Codespace/DataCoach`
- 不改动源仓库：`/home/jolia/DataCoach`
- 串口权限（按实际端口调整）：

```bash
sudo chmod 777 /dev/ttyACM0
sudo chmod 777 /dev/ttyACM1
```

## 1) 终端启动顺序（最小集）

### Terminal A: policy server（jolia uv + 你仓库脚本）

二选一配置启动（可按需替换）：

```bash
/home/jolia/.local/bin/uv run --project /home/jolia/DataCoach \
  python /home/pengyue/Codespace/DataCoach/scripts/inference/my_serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_ltc_pick_twice \
  --policy.dir /home/pengyue/29000
```

```bash
/home/jolia/.local/bin/uv run --project /home/jolia/DataCoach \
  python /home/pengyue/Codespace/DataCoach/scripts/inference/my_serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_localdata_a1_lora \
  --policy.dir /home/pengyue/29000
```

### Terminal B: ROS master

```bash
roscore
```

### Terminal C: A1 arm node

```bash
roslaunch signal_arm single_arm_node.launch single_arm_serial_port_path:=/dev/ttyACM1
```

### Terminal D: ee tracker

```bash
roslaunch mobiman eeTrackerdemo.launch
```

### Terminal E: unified data services（live 模式）

```bash
cd /home/pengyue/Codespace/DataCoach
python scripts/collect_data/run_data_services.py service_mode=live
```

说明：`service_mode=live` 会启动：
- `camera_server`
- `a1_server`（leader udp + ros bridge + policy action subscriber）

## 2) 回退方式

如果 `uv run --project` 在机器上行为不一致，可改用 jolia 的 venv python：

```bash
/home/jolia/DataCoach/.venv/bin/python \
  /home/pengyue/Codespace/DataCoach/scripts/inference/my_serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_localdata_a1_lora \
  --policy.dir /home/pengyue/29000
```

## 3) 常见问题

- `Config 'pi05_ltc_pick_twice' not found`：
  - 当前实现在 `my_serve_policy.py` 内做了别名兼容，会映射到 `pi05_localdata_a1_lora`。
- `ROS master is not online`：
  - 先确认 `roscore` 已启动。
- 无动作输出：
  - 检查 `run_data_services.py service_mode=live` 是否在运行。
  - 检查 `ZMQ` 端口是否冲突（`5557/5558/5559`）。
- 串口报错：
  - 重新确认 `ttyACM` 号与 `chmod` 权限。
