# DragDataCoach

DragDataCoach 的核心流程是：
1. 人手拖拽机械臂，录制 bag。
2. 用 bag 回放轨迹。
3. 回放期间采集双相机视频 + 机械臂状态，生成数据集。

本仓库已经把流程封装成 `just` 命令，避免手敲长指令。

---

## 1. 安装 `just`

本机推荐安装方式（已验证可用）：

```bash
sudo snap install just --classic
just --version
```

---

## 2. 一次性准备

### 2.1 同步 A1 SDK 到 third-party

```bash
scripts/collect_data/sync_a1_sdk.sh /home/eric/A1_SDK
```

会同步到：

```bash
third_party/A1_SDK
```

### 2.2 环境检查

```bash
just doctor
just which-python
```

环境文档：
- [Environment Setup](docs/SETUP_ENV.md)
- [A1 Serial Setup (udev)](docs/SETUP_UDEV.md)

---

## 3. `just` 命令总览

```bash
just --list
```

常用命令：
- `just launch-driver [serial]`
- `just launch-ee-record [serial]`
- `just drag-start`
- `just drag-stop`
- `just gripper-keyboard`
- `just gripper-stop`
- `just record-start [tag]`
- `just record-stop`
- `just launch-tracker`
- `just replay <bag> [rate] [gripper_mode]`
- `just replay-arm-only <bag> [rate]`
- `just collect`
- `just latest-bag`
- `just bag-info <bag>`
- `just camera-test`

---

## 4. 相机连接测试

先测相机再开采集：

```bash
just camera-test
```

等价直跑脚本：

```bash
PY=$(scripts/collect_data/dragdatacoach.sh which-python)
$PY scripts/collect_data/test_camera_connections.py --config configs/drag_replay.yaml
```

默认会保存探测图到 `outputs/camera_probe/`。

如果你只想快速测一次并且不落盘：

```bash
just camera-test-raw --config configs/drag_replay.yaml --timeout-s 1.0 --no-save
```

如果提示 `pyrealsense2 is not installed`，说明当前 Python 环境缺少 RealSense 依赖；此时可以先安装依赖，或者先把配置里 RealSense 相机 `enabled` 设为 `false` 只测手眼相机。

---

## 5. 录制（拖拽）流程

建议开 3~4 个终端。

### 终端 A：启动录制链路（driver + end_effector_pose 发布）

```bash
just launch-ee-record /dev/ttyACM0
```

### 终端 B：开启拖拽模式

```bash
just drag-start
```

### 终端 C（可选）：键盘夹爪

```bash
just gripper-keyboard
```

### 终端 D：开始/结束录包

```bash
just record-start drag_demo
# 结束后
just record-stop
```

结束拖拽：

```bash
just drag-stop
just gripper-stop
```

查看最新 bag：

```bash
just latest-bag
```

---

## 6. 回放 + 数据采集流程

建议开 3 个终端。

### 终端 A：机械臂驱动

```bash
just launch-driver /dev/ttyACM0
```

### 终端 B：eeTracker 控制器

```bash
just launch-tracker
```

### 终端 C：DataCoach 采集器

```bash
just collect
```

按提示 `Enter` 开始采集，然后在另一个终端执行回放：

```bash
just replay /path/to/your.bag 1.0 position
```

如果你只想先验证机械臂轨迹，不回放夹爪：

```bash
just replay-arm-only /path/to/your.bag 1.0
```

---

## 7. 关键防呆（已内置）

- `record-start` 前会检查 `/end_effector_pose`。
  - 若没有该 topic，会阻止录制（否则会出现“夹爪能回放，机械臂不动”）。
- `replay` 前会检查是否有 `a1_gripper_keyboard.py` 在后台跑。
  - 若在跑，会阻止回放（避免夹爪命令冲突导致异常抖动）。

---

## 8. 输出路径

原始采集输出：

```bash
data/raw_data/<task_name>/demo_<index>/
```

对齐与转换后输出：

```bash
data/processed_data/<task_name>/demo_<index>/
```

典型文件：
- `cam_0_rgb_video.mp4`
- `cam_1_rgb_video.mp4`
- `states.pkl`
- `commanded_states.pkl`
- `trajectory.csv`
