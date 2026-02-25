import argparse
import time
import socket
import numpy as np
from scipy.spatial.transform import Rotation as R
from lerobot.teleoperators.so101_leader import SO101Leader
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.model.kinematics import RobotKinematics
import hydra
from omegaconf import DictConfig
from pathlib import Path

def run_leader_monitor(port, id, urdf_path, target_frame_name, calibrate, interval, send_addr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    config = SO101LeaderConfig(port=port, id=id)
    teleop = SO101Leader(config)
    teleop.connect(calibrate=calibrate)
    print("Leader arm connected — reading joint angles in a loop.")

    names = list(teleop.bus.motors.keys())
    kin = RobotKinematics(urdf_path, target_frame_name=target_frame_name, joint_names=names)

    n_lines = 4
    def print_status(lines):
        print("\033[F" * n_lines, end='')
        for l in lines:
            print(l)
    for _ in range(n_lines):
        print()
    while True:
        raw = teleop.bus.sync_read("Present_Position", names)
        joint_deg = np.array([raw[n] for n in names])
        ee_pose = kin.forward_kinematics(joint_deg)
        pos = ee_pose[:3, 3]
        rot = ee_pose[:3, :3]
        euler = R.from_matrix(rot).as_euler('zyx', degrees=True)
        quat = R.from_matrix(rot).as_quat()
        gripper_angle = raw.get('gripper', None)

        line1 = ', '.join(f'{k}: {v:.2f}' for k, v in raw.items())
        line2 = f"EE pos: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}"
        line3 = f"rot(euler ZYX): {euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}"
        line4 = f"quat: x={quat[0]:.3f}, y={quat[1]:.3f}, z={quat[2]:.3f}, w={quat[3]:.3f}"
        print_status([line1, line2, line3, line4])

        if gripper_angle is not None:
            msg = f"{pos[0]},{pos[1]},{pos[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]},{gripper_angle}"
        else:
            msg = f"{pos[0]},{pos[1]},{pos[2]},{quat[0]},{quat[1]},{quat[2]},{quat[3]}"
        sock.sendto(msg.encode(), send_addr)
        time.sleep(interval)


def main(cfg: DictConfig):
    """
    Hydra entrypoint for SO101 Leader monitor.
    """
    print("🤖 Starting LeRobot Leader Monitor with config:")
    print(cfg)

    run_leader_monitor(
        port=cfg.port,
        id=cfg.robot_id,
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.target_frame_name,
        calibrate=cfg.calibrate,
        interval=cfg.interval,
        send_addr=(cfg.send_ip, cfg.send_port),
    )


if __name__ == "__main__":
    main()