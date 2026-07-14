import pytest

from galaxea_a1_runtime.apps.eef_bridge import (
    EefCommandPublisher,
    condition_state_from_action8,
    decode_relay_status,
    format_xyz_direction,
    pose_msg_to_xyz_quat,
    relay_state_summary,
    relay_status_is_fresh,
)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class FakeTime:
    @staticmethod
    def now():
        return "now"


class FakeRospy:
    Time = FakeTime


class FakeHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class FakePosition:
    x = 0.0
    y = 0.0
    z = 0.0


class FakeOrientation:
    x = 0.0
    y = 0.0
    z = 0.0
    w = 1.0


class FakePose:
    def __init__(self):
        self.position = FakePosition()
        self.orientation = FakeOrientation()


class FakePoseStamped:
    def __init__(self):
        self.header = FakeHeader()
        self.pose = FakePose()


class FakeBool:
    def __init__(self, data=False):
        self.data = bool(data)


class FakeGripper:
    def __init__(self):
        self.header = FakeHeader()
        self.gripper_stroke = 0.0


def test_relay_status_helpers_are_small_and_predictable():
    status = decode_relay_status('{"state": "ACTIVE", "reason": "ok"}')

    assert status.state == "ACTIVE"
    assert relay_status_is_fresh(10.0, max_age_s=1.0, now=10.5)
    assert "fresh" in relay_state_summary(status, 10.0, max_age_s=1.0, now=10.5)
    assert decode_relay_status("not-json").state == "FAULT"


def test_pose_msg_and_condition_helpers():
    msg = FakePoseStamped()
    msg.pose.position.x = 0.1
    msg.pose.position.y = -0.2
    msg.pose.position.z = 0.3

    xyz_quat = pose_msg_to_xyz_quat(msg)
    state = condition_state_from_action8([0.1, -0.2, 0.3, 0, 0, 0, 1, 0.5], frame_chunk_size=4, action_per_frame=20)

    assert xyz_quat is not None
    xyz, quat = xyz_quat
    assert xyz == pytest.approx((0.1, -0.2, 0.3))
    assert quat == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert state.shape == (8, 4, 20)
    assert format_xyz_direction([0.0, -0.02, 0.03], deadband_m=0.001) == "y-,z+"


def test_eef_command_publisher_publishes_pose_enable_and_gripper():
    pose_pub = FakePublisher()
    gripper_pub = FakePublisher()
    enable_pub = FakePublisher()
    publisher = EefCommandPublisher(
        rospy=FakeRospy,
        pose_pub=pose_pub,
        gripper_pub=gripper_pub,
        motion_enable_pub=enable_pub,
        pose_msg_type=FakePoseStamped,
        bool_msg_type=FakeBool,
        gripper_msg_type=FakeGripper,
        command_frame="world",
        gripper_to_stroke=lambda value: value * 100.0,
    )

    publisher.publish_motion_enable(True)
    publisher.publish_action([0.1, 0.2, 0.3, 0, 0, 0, 1, 0.5], publish_gripper=True)

    assert enable_pub.published[-1].data is True
    assert pose_pub.published[-1].header.frame_id == "world"
    assert pose_pub.published[-1].pose.position.x == pytest.approx(0.1)
    assert gripper_pub.published[-1].gripper_stroke == pytest.approx(50.0)


def test_eef_command_publisher_dry_run_keeps_active_target_without_publishing():
    pose_pub = FakePublisher()
    publisher = EefCommandPublisher(
        rospy=FakeRospy,
        pose_pub=pose_pub,
        gripper_pub=FakePublisher(),
        motion_enable_pub=FakePublisher(),
        pose_msg_type=FakePoseStamped,
        bool_msg_type=FakeBool,
        gripper_msg_type=FakeGripper,
        command_frame="world",
        gripper_to_stroke=lambda value: value * 100.0,
        execute=False,
    )

    publisher.publish_action([0.1, 0.2, 0.3, 0, 0, 0, 1, 0.5], publish_gripper=True)

    assert publisher.active_pose_target is not None
    assert pose_pub.published == []
