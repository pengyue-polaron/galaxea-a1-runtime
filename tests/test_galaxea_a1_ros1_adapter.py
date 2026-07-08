import pytest

from galaxea_a1_runtime.config import RuntimeConfig, RuntimeProfile
from galaxea_a1_runtime.hardware.eef import EefPose
from galaxea_a1_runtime.hardware.ros1 import Ros1A1HardwareIO
from galaxea_a1_runtime.policies.actions import RuntimeAction
from galaxea_a1_runtime.schema import ActionMode


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class FakeStamp:
    def __init__(self, value=12.5):
        self.value = value

    def to_sec(self):
        return self.value


class FakeTime:
    @staticmethod
    def now():
        return FakeStamp()


class FakeRospy:
    Time = FakeTime


class FakeHeader:
    def __init__(self):
        self.stamp = FakeStamp()
        self.frame_id = ""


class FakePosition:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class FakeOrientation:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


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


class FakeGripperCommand:
    def __init__(self):
        self.header = FakeHeader()
        self.gripper_stroke = 0.0


class FakeSubscriber:
    def __init__(self):
        self.unregistered = False

    def unregister(self):
        self.unregistered = True


def configured_adapter() -> Ros1A1HardwareIO:
    adapter = Ros1A1HardwareIO(RuntimeConfig(profile=RuntimeProfile.SAFE))
    adapter._connected = True
    adapter._rospy = FakeRospy
    adapter._ee_pub = FakePublisher()
    adapter._motion_enable_pub = FakePublisher()
    adapter._gripper_pub = FakePublisher()
    adapter._pose_stamped_type = FakePoseStamped
    adapter._bool_type = FakeBool
    adapter._gripper_type = FakeGripperCommand
    return adapter


def test_ros1_adapter_publishes_eef_target_through_relay_enable():
    adapter = configured_adapter()
    adapter._latest_eef_pose = EefPose((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), "base_link")
    action = RuntimeAction(
        mode=ActionMode.EEF_TRANSLATION,
        values=(0.01, -0.02, 0.03, 0.5),
        names=("delta_x", "delta_y", "delta_z", "gripper"),
        source="test",
    )

    sent = adapter.send_runtime_action(action)

    assert sent is action
    assert [msg.data for msg in adapter._motion_enable_pub.published] == [True]
    assert len(adapter._ee_pub.published) == 2
    target_msg = adapter._ee_pub.published[-1]
    assert target_msg.header.frame_id == "base_link"
    assert target_msg.pose.position.x == pytest.approx(0.11)
    assert target_msg.pose.position.y == pytest.approx(0.18)
    assert target_msg.pose.position.z == pytest.approx(0.33)
    assert target_msg.pose.orientation.w == pytest.approx(1.0)
    assert adapter._gripper_pub.published[-1].gripper_stroke == pytest.approx(30.0)


def test_ros1_adapter_requires_eef_feedback_before_arm_motion():
    adapter = configured_adapter()
    action = RuntimeAction(
        mode=ActionMode.EEF_TRANSLATION,
        values=(0.01, 0.0, 0.0, 0.5),
        names=("delta_x", "delta_y", "delta_z", "gripper"),
        source="test",
    )

    with pytest.raises(RuntimeError, match="/end_effector_pose"):
        adapter.send_runtime_action(action)

    assert adapter._ee_pub.published == []
    assert adapter._motion_enable_pub.published == []
    assert adapter._gripper_pub.published == []


def test_ros1_adapter_allows_gripper_only_without_eef_feedback():
    adapter = configured_adapter()
    action = RuntimeAction(
        mode=ActionMode.EEF_TRANSLATION,
        values=(0.0, 0.0, 0.0, 0.25),
        names=("delta_x", "delta_y", "delta_z", "gripper"),
        source="test",
    )

    adapter.send_runtime_action(action)

    assert adapter._ee_pub.published == []
    assert adapter._motion_enable_pub.published == []
    assert adapter._gripper_pub.published[-1].gripper_stroke == pytest.approx(15.0)


def test_ros1_adapter_eef_feedback_updates_observation_state():
    adapter = configured_adapter()
    msg = FakePoseStamped()
    msg.header.frame_id = "base_link"
    msg.pose.position.x = 0.12
    msg.pose.position.y = -0.05
    msg.pose.position.z = 0.31

    adapter._eef_pose_cb(msg)
    observation = adapter.get_observation()

    assert observation.state[:7] == pytest.approx(
        (0.12, -0.05, 0.31, 0.0, 0.0, 0.0, 1.0),
        abs=1e-6,
    )
    assert observation.timestamp == pytest.approx(12.5)
    assert adapter._latest_eef_pose is not None
    assert adapter._latest_eef_pose.frame_id == "base_link"


def test_ros1_adapter_rejects_joint_actions():
    adapter = configured_adapter()
    action = RuntimeAction(
        mode=ActionMode.JOINT_ABSOLUTE,
        values=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
        names=("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"),
        source="test",
    )

    with pytest.raises(ValueError, match="only supports EEF"):
        adapter.send_runtime_action(action)


def test_ros1_adapter_disconnect_disables_motion_and_unregisters_feedback():
    adapter = configured_adapter()
    subscriber = FakeSubscriber()
    adapter._eef_sub = subscriber

    adapter.disconnect()

    assert adapter.is_connected is False
    assert adapter._motion_enable_pub.published[-1].data is False
    assert subscriber.unregistered is True
    assert adapter._eef_sub is None
