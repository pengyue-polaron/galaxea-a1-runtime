The A1_SDK includes both the CONTROL_SDK and DRIVER_SDK. 
For detailed usage instructions, please refer to the documentation below. 
The URDF files can be found at install/share/mobiman/urdf. 
Details of the URDF file updates are available in the repository [URDF repository](https://github.com/userguide-galaxea/URDF).

[CONTROL_SDK](README_CONTROL.md)

[DRIVER_SDK](README_DRIVER.md)

## Drag Collection Utilities

This repository now includes a record-only launch and helper scripts for hand-guided EEF trajectory collection.

1. Start record-only pipeline (driver + EEF pose publisher, no tracker controller):

```bash
source install/setup.bash
roslaunch mobiman ee_record_only.launch
```

2. Start/stop rosbag recording:

```bash
./tools/a1_record.sh start drag_demo
./tools/a1_record.sh stop
```

3. Optional but recommended for hand-guided data collection: enable low-stiffness drag mode.

```bash
./tools/a1_drag_mode.sh start
# If still too stiff, use lower gains:
# ./tools/a1_drag_mode.sh start "1.0,1.0,0.8,0.6,0.4,0.3" "0.06,0.06,0.05,0.04,0.03,0.02"
# drag mode also publishes gripper position hold to /gripper_position_control_host by default.
# disable it if needed:
# A1_DRAG_HOLD_GRIPPER_POSITION=0 ./tools/a1_drag_mode.sh start
./tools/a1_drag_mode.sh status
# After collection:
# ./tools/a1_drag_mode.sh stop
```

4. Optional: keyboard gripper control during recording.

```bash
# If you use drag mode, disable drag-mode gripper hold first to avoid command conflict:
# A1_DRAG_HOLD_GRIPPER_POSITION=0 ./tools/a1_drag_mode.sh start

./tools/a1_gripper_keyboard.py
# custom key mapping example:
# ./tools/a1_gripper_keyboard.py --open-key u --close-key i --quit-key q
# keys:
#   o: fully open
#   c: fully close
#   q: quit
```

5. Convert bag to EEF CSV:

```bash
./tools/bag_to_eef_csv.py --bag data/records/a1_eef_drag_demo_YYYYMMDD_HHMMSS.bag
# default CSV columns include:
# t,x,y,z,qx,qy,qz,qw,gripper_cmd,gripper_feedback
# if you only want old EEF columns:
# ./tools/bag_to_eef_csv.py --bag <bag_path> --no-gripper
```

6. One-click replay (arm + gripper commands from bag):

```bash
# Keep driver + eeTracker running in other terminals, then:
./tools/a1_replay.sh --bag /home/eric/A1_SDK/data/records/a1_eef_drag_demo_YYYYMMDD_HHMMSS.bag --rate 1.0
# default gripper replay mode is auto (position > command > force), to avoid topic conflicts.
# if needed, force position mode explicitly:
# ./tools/a1_replay.sh --bag <bag_path> --gripper-mode position
```
