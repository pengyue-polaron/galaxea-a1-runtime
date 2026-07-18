import pytest

from galaxea_a1_runtime.apps.eef_bridge import (
    EefIkCommandPublisher,
    condition_state_from_action8,
    format_xyz_direction,
    pose_msg_to_xyz_quat,
)
from galaxea_a1_runtime.hardware.eef_ik import IkSolution
from galaxea_a1_runtime.runtime.relay import (
    decode_relay_status,
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


class FakeJointState:
    def __init__(self):
        self.header = FakeHeader()
        self.name = []
        self.position = []


class FakeIkSolver:
    def __init__(self):
        self.calls = []

    def solve(self, current, xyz, quat):
        self.calls.append((tuple(current), tuple(xyz), tuple(quat)))
        return IkSolution(
            joint_positions=(0.1, 0.2),
            iterations=3,
            position_error_m=0.0001,
            orientation_error_rad=0.001,
            max_joint_delta_rad=0.2,
        )


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
    state = condition_state_from_action8(
        [0.1, -0.2, 0.3, 0, 0, 0, 1, 0.5], frame_chunk_size=4, action_per_frame=20
    )

    assert xyz_quat is not None
    xyz, quat = xyz_quat
    assert xyz == pytest.approx((0.1, -0.2, 0.3))
    assert quat == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert state.shape == (8, 4, 20)
    assert format_xyz_direction([0.0, -0.02, 0.03], deadband_m=0.001) == "y-,z+"


def test_eef_joint_publisher_solves_named_target_before_gripper_publication():
    target_pub = FakePublisher()
    gripper_pub = FakePublisher()
    enable_pub = FakePublisher()
    solver = FakeIkSolver()
    publisher = EefIkCommandPublisher(
        rospy=FakeRospy,
        target_pub=target_pub,
        gripper_pub=gripper_pub,
        motion_enable_pub=enable_pub,
        joint_state_msg_type=FakeJointState,
        bool_msg_type=FakeBool,
        gripper_msg_type=FakeGripper,
        joint_names=("arm_joint1", "arm_joint2"),
        current_joint_positions=lambda: (0.0, 0.0),
        solver=solver,
        gripper_to_stroke=lambda value: value * 100.0,
        execute=True,
    )

    hold = publisher.hold_current_target()
    action = [0.1, 0.2, 0.3, 0, 0, 0, 1, 0.5]
    publisher.publish_action(action, publish_gripper=False)
    publisher.publish_action(action, publish_gripper=True)

    assert hold == pytest.approx((0.0, 0.0))
    assert solver.calls[0] == (
        (0.0, 0.0),
        (0.1, 0.2, 0.3),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert target_pub.published[-1].name == ["arm_joint1", "arm_joint2"]
    assert target_pub.published[-1].position == pytest.approx((0.1, 0.2))
    assert len(gripper_pub.published) == 1
    assert gripper_pub.published[0].gripper_stroke == pytest.approx(50.0)


def test_eef_feedback_decoder_rejects_non_finite_pose():
    msg = FakePoseStamped()
    msg.pose.position.x = float("nan")

    assert pose_msg_to_xyz_quat(msg) is None


def test_condition_state_rejects_non_finite_values_and_invalid_shape():
    with pytest.raises(ValueError):
        condition_state_from_action8(
            [0.1, 0.2, float("nan"), 0, 0, 0, 1, 0.5],
            frame_chunk_size=4,
            action_per_frame=20,
        )
    with pytest.raises(ValueError):
        condition_state_from_action8(
            [0.1, 0.2, 0.3, 0, 0, 0, 1, 0.5],
            frame_chunk_size=0,
            action_per_frame=20,
        )
