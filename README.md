# DragDataCoach

DragDataCoach 的目标流程：
1. 手拖机械臂录 bag。
2. 用 bag 回放轨迹。
3. 回放期间采集双相机视频 + 机械臂状态。

最终每条 demo 会包含两路视频和轨迹状态数据。

## 1. 安装 `just`

```bash
sudo snap install just --classic
just --version
```

## 2. 一次性准备

环境检查：

```bash
just doctor
just which-python
```

## 3. 新的 `just` 风格（空格子命令）

```bash
just drag start
just drag stop

just launch driver /dev/ttyACM0
just launch ee-record /dev/ttyACM0
just launch tracker

just gripper start
just gripper open
just gripper close
just gripper stop

just record start drag_demo
just record stop

just replay
just replay /path/to/demo.bag 1.0 position

just collect
just drag-collect --serial /dev/ttyACM0 --tag drag_demo

just camera test
just camera raw --config configs/drag_replay.yaml

just bag latest
just bag info /path/to/demo.bag
```

查看全部命令：

```bash
just --list
```

## 4. 相机链路测试

```bash
just camera test
```

现在只做设备枚举和串口/设备节点检查，不会尝试打开相机或保存探测图。

`just replay` 不带 bag 参数时，会自动使用 `third_party/A1_SDK/data/records/` 里的最新 bag。

`just replay` 会在开始回放前检查 `cam_0` 和 `cam_1`；任意一个没连上就直接退出。`just drag-collect` 只会在 replay 阶段检查一次，而且检查发生在启动 `collect` 之前。

## 5. 标准手动流程

录制（拖拽）阶段：

```bash
just launch ee-record /dev/ttyACM0
just drag start
just gripper start               # 可选
just record start drag_demo
# 完成拖拽后
just record stop
just drag stop
just gripper stop
```

回放 + 采集阶段（建议 3 终端）：

```bash
just launch driver /dev/ttyACM0
just launch tracker
just collect
just replay /path/to/demo.bag 1.0 position
```

## 6. 本地麦克风阈值触发远端夹爪

你的麦克风如果在本地电脑，而控制代码跑在 SSH 连上的远端主机，建议在本地跑音量阈值监听，再通过 `ssh` 调远端 `just gripper open/close`。

远端已经支持一次性夹爪命令：

```bash
just gripper open
just gripper close
```

本地监听脚本在：

```bash
scripts/collect_data/gripper_audio_threshold.py
```

先在本地电脑安装音频依赖：

```bash
python3 -m pip install numpy sounddevice
```

先列出本地麦克风设备：

```bash
python3 scripts/collect_data/gripper_audio_threshold.py --ssh-host <your-ssh-host> --list-devices
```

最小用法，超过阈值就交替执行开/关：

```bash
python3 scripts/collect_data/gripper_audio_threshold.py \
  --ssh-host <your-ssh-host> \
  --threshold-db -24 \
  --trigger-mode toggle
```

如果你只想“声音大于阈值就打开”：

```bash
python3 scripts/collect_data/gripper_audio_threshold.py \
  --ssh-host <your-ssh-host> \
  --threshold-db -24 \
  --trigger-mode open
```

常用参数：

- `--threshold-db`：触发阈值，单位 dBFS，越接近 `0` 越难触发。
- `--reset-db`：重新 armed 的阈值，默认比 `threshold` 低 8 dB。
- `--cooldown-seconds`：触发后的冷却时间，避免连续抖动。
- `--initial-state close`：`toggle` 模式下第一次触发会执行 `open`。
- `--device <name-or-index>`：指定本地麦克风。

## 7. All-in-One 单终端脚本

你可以直接用 `just` 启动 all-in-one：

```bash
just drag-collect --serial /dev/ttyACM0 --tag drag_demo
```

它等价调用脚本：

```bash
scripts/collect_data/dragdatacoach_all_in_one.sh
```

它会在后台创建 `tmux` 会话并自动拉起多窗口，完成录制 + 回放 + 采集的完整流程。
在录制阶段，脚本会在当前终端打开 `just gripper start`，你可以直接键盘控制夹爪。
如果检测到同名 tmux 会话，脚本默认会直接 `restart`；你也可以用 `--on-existing` 改成 `ask / attach / new / abort`。

最常用：

```bash
just drag-collect --serial /dev/ttyACM0 --tag drag_demo
```

只做回放采集（跳过录制）：

```bash
just drag-collect \
  --skip-record \
  --bag /path/to/demo.bag \
  --serial /dev/ttyACM0
```

可选参数：

```bash
--rate <float>              # replay 速度
--gripper-mode <mode>       # replay 夹爪模式
--session <name>            # tmux session 名称
--no-gripper-keyboard       # 录制时不启动键盘夹爪控制
--no-auto-stop              # 流程结束后不自动停掉 replay 的 launch 窗口
--on-existing <policy>      # ask|restart|attach|new|abort
```

## 8. 输出路径

原始数据：

```bash
data/raw_data/<task_name>/demo_<index>_<YYYYMMDD_HHMMSS>/
```

典型文件：
- `cam_0_rgb_video.mp4`
- `cam_1_rgb_video.mp4`
- `states.pkl`
- `commanded_states.pkl`
- `trajectory.csv`
