"""A1-specific SO leader adapter built on LeRobot motor primitives."""

from __future__ import annotations

import logging
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from galaxea_a1_runtime.console import Tone, info, style, success

logger = logging.getLogger(__name__)

# Feetech writes expect a status packet from each motor. A single missed packet
# should not abort a reset, but the operation must still fail closed if the
# motor remains unreachable.
MOTOR_WRITE_NUM_RETRY = 5


class A1SOLeader(Teleoperator):
    """SO leader wiring used by the A1 teleop rig.

    Official LeRobot SO leaders expose shoulder/wrist names. This hardware uses
    six generic arm axes, motor IDs 0..5, plus an independent gripper at ID 6.
    """

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = (
            MotorNormMode.DEGREES
            if config.use_degrees
            else MotorNormMode.RANGE_M100_100
        )
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint0": Motor(0, "sts3215", norm_mode_body),
                "joint1": Motor(1, "sts3215", norm_mode_body),
                "joint2": Motor(2, "sts3215", norm_mode_body),
                "joint3": Motor(3, "sts3215", norm_mode_body),
                "joint4": Motor(4, "sts3215", norm_mode_body),
                "joint5": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()
        self.configure()
        logger.info("%s connected.", self)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            user_input = input(
                style(
                    f"Use calibration for {self.id}: Enter=accept, c=recalibrate > ",
                    Tone.STEP,
                )
            )
            if user_input.strip().lower() != "c":
                logger.info(
                    "Writing calibration file associated with the id %s to the motors",
                    self.id,
                )
                self.bus.write_calibration(self.calibration)
                return

        logger.info("\nRunning calibration of %s", self)
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(
            style(
                f"Move {self} to the middle of its range, then press Enter > ",
                Tone.STEP,
            )
        )
        homing_offsets = self.bus.set_half_turn_homings()
        info(
            "Move all joints sequentially through their entire ranges of motion.\n"
            "Recording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(
            list(self.bus.motors)
        )

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        success(f"Calibration saved: {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque(num_retry=MOTOR_WRITE_NUM_RETRY)
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write(
                "Operating_Mode",
                motor,
                OperatingMode.POSITION.value,
                num_retry=MOTOR_WRITE_NUM_RETRY,
            )

    def enable_torque(self) -> None:
        self.bus.enable_torque(num_retry=MOTOR_WRITE_NUM_RETRY)

    def disable_torque(self) -> None:
        self.bus.disable_torque(num_retry=MOTOR_WRITE_NUM_RETRY)

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(style(f"Connect only motor {motor}, then press Enter > ", Tone.STEP))
            self.bus.setup_motor(motor)
            success(f"Motor {motor} ID set to {self.bus.motors[motor].id}.")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug("%s read action: %.1fms", self, dt_ms)
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {
            key.removesuffix(".pos"): value
            for key, value in feedback.items()
            if key.endswith(".pos")
        }
        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info("%s disconnected.", self)
