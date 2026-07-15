from types import SimpleNamespace

from galaxea_a1_runtime.runtime.health_checks import (
    RosDoctorContext,
    arm_control_result,
    motor_status_level,
    relay_status_result,
)


def _motor_status(*codes: int):
    errors = [SimpleNamespace(error_code=code, error_description=[]) for code in codes]
    return SimpleNamespace(
        data=SimpleNamespace(
            name=[f"motor{index}" for index in range(len(codes))],
            motor_errors=errors,
        )
    )


def test_idle_timeout_is_non_blocking_but_other_motor_bits_fail():
    assert motor_status_level(_motor_status(0, 0, 0, 0, 0, 0, 64))[0] == "WARN"
    assert motor_status_level(_motor_status(0, 0, 0, 0, 0, 0, 68))[0] == "FAIL"


def test_execution_doctor_accepts_only_locked_or_active_relay():
    locked = SimpleNamespace(data='{"state":"LOCKED","reason":"operator gate"}')
    fault = SimpleNamespace(data='{"state":"FAULT","reason":"stale input"}')

    assert relay_status_result(locked, require_execution=True)[0]
    assert not relay_status_result(fault, require_execution=True)[0]
    assert not relay_status_result(fault, require_execution=False)[0]


def test_staged_command_doctor_uses_complete_relay_contract():
    message = SimpleNamespace(
        p_des=[0.0] * 6,
        v_des=[0.0] * 6,
        kp=[20.0] * 6,
        kd=[1.0] * 6,
        t_ff=[0.0] * 6,
        mode=0,
    )

    assert arm_control_result(message, arm_joints=6, allowed_modes=(0,))[0]
    message.kd[0] = float("nan")
    try:
        arm_control_result(message, arm_joints=6, allowed_modes=(0,))
    except ValueError:
        pass
    else:
        raise AssertionError("doctor must reject a relay-invalid staged command")


def test_ros_doctor_context_composes_shared_topic_and_node_checks():
    message = SimpleNamespace(value=3)

    class FakeRospy:
        @staticmethod
        def get_published_topics():
            return [("/ready", "example/Message")]

        @staticmethod
        def wait_for_message(topic, message_type, timeout):
            assert topic == "/ready"
            assert message_type is object
            assert timeout == 0.25
            return message

    class FakeRosnode:
        @staticmethod
        def get_node_names():
            return ["/worker"]

        @staticmethod
        def rosnode_ping(name, max_count, verbose):
            return name == "/worker" and max_count == 1 and verbose is False

    checks = []
    context = RosDoctorContext(
        rospy=FakeRospy,
        rosnode=FakeRosnode,
        checks=checks,
        timeout_s=0.25,
        required=True,
    )

    observed = context.message(
        "ready", "/ready", object, lambda item: (item.value == 3, "valid")
    )
    alive = context.node("worker", "/worker")

    assert observed is message
    assert alive is True
    assert [check.level for check in checks] == ["PASS", "PASS"]
