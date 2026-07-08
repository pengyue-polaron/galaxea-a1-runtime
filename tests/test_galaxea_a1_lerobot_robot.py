import pytest

from galaxea_a1_runtime.hardware.io import A1Observation, NullA1HardwareIO
from galaxea_a1_runtime.lerobot.robot import GalaxeaA1Robot, GalaxeaA1RobotConfig
from galaxea_a1_runtime.schema import ActionMode, CameraSpec


def test_galaxea_a1_robot_uses_null_io_without_hardware():
    io = NullA1HardwareIO(
        A1Observation(
            state=(0.0,) * 14,
            images={"front": "front-image"},
            timestamp=1.25,
        )
    )
    robot = GalaxeaA1Robot(
        GalaxeaA1RobotConfig(
            action_mode=ActionMode.EEF_TRANSLATION,
            camera_specs=(CameraSpec("front", height=480, width=640),),
        ),
        io=io,
    )

    robot.connect()
    observation = robot.get_observation()
    sent = robot.send_action({"action": [0.2, -0.2, 0.01, 2.0]})

    assert observation["observation.state"] == pytest.approx((0.0,) * 14)
    assert observation["observation.images.front"] == "front-image"
    assert sent == {
        "delta_x": 0.03,
        "delta_y": -0.03,
        "delta_z": 0.01,
        "gripper": 1.0,
    }
    assert io.last_action is not None
    robot.disconnect()
    assert robot.is_connected is False


def test_galaxea_a1_robot_features_match_contract():
    robot = GalaxeaA1Robot(GalaxeaA1RobotConfig())

    assert robot.observation_features["observation.state"] == (14,)
    assert robot.action_features["action"] == (7,)
