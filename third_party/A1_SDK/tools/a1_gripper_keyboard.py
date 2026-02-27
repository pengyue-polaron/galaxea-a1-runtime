#!/usr/bin/env python3
import argparse
import os
import select
import sys
import termios
import time
import tty
from typing import Optional

# Let script run even if user forgets to source ROS env.
for candidate in (
    "/opt/ros/noetic/lib/python3/dist-packages",
    "/usr/lib/python3/dist-packages",
):
    if os.path.isdir(candidate) and candidate not in sys.path:
        sys.path.append(candidate)

try:
    import rospy
    from signal_arm.msg import gripper_position_control
except Exception as exc:
    print(
        f"ROS import failed: {exc}\n"
        "Tip: source install/setup.bash (or /opt/ros/noetic/setup.bash) first.",
        file=sys.stderr,
    )
    sys.exit(1)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def normalize_key(parser_obj, raw_value: str, arg_name: str) -> str:
    text = raw_value.lower()
    if len(text) != 1:
        parser_obj.error(f"{arg_name} must be a single character")
    return text


class RawTerminal:
    def __init__(self):
        self.fd = None
        self.old_settings = None

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)


class GripperKeyboardNode:
    def __init__(self, args):
        self.args = args
        self.target_stroke = None
        if args.initial_state == "open":
            self.target_stroke = self.args.open_stroke
        elif args.initial_state == "close":
            self.target_stroke = self.args.close_stroke

        self.pub = rospy.Publisher(
            self.args.topic, gripper_position_control, queue_size=10
        )
        self.rate = rospy.Rate(self.args.rate)

    def read_key(self, timeout_s: float) -> Optional[str]:
        readable, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not readable:
            return None
        ch = sys.stdin.read(1)
        return ch

    def publish_stroke(self, stroke: float):
        cmd = gripper_position_control()
        cmd.header.stamp = rospy.Time.now()
        cmd.gripper_stroke = float(stroke)
        self.pub.publish(cmd)

    def publish_target(self):
        if self.target_stroke is None:
            return
        self.publish_stroke(self.target_stroke)

    def burst_publish(self, stroke: float):
        for _ in range(self.args.burst_count):
            if rospy.is_shutdown():
                return
            self.publish_stroke(stroke)
            if self.args.burst_interval > 0:
                time.sleep(self.args.burst_interval)

    def apply_key(self, key: str) -> bool:
        key = key.lower()

        if key == " ":
            if self.target_stroke is None:
                self.target_stroke = self.args.close_stroke
            elif abs(self.target_stroke - self.args.close_stroke) <= abs(
                self.target_stroke - self.args.open_stroke
            ):
                self.target_stroke = self.args.open_stroke
            else:
                self.target_stroke = self.args.close_stroke
            rospy.loginfo("Toggle gripper: %.2f mm", self.target_stroke)
            self.burst_publish(self.target_stroke)
            return True

        if self.args.quit_on_enter and key in ("\r", "\n"):
            rospy.loginfo("Enter key received.")
            return False

        if key in (self.args.quit_key, "\x03"):
            rospy.loginfo("Quit key received.")
            return False

        if key == self.args.open_key:
            self.target_stroke = self.args.open_stroke
            rospy.loginfo("Set full OPEN: %.2f mm", self.target_stroke)
            self.burst_publish(self.target_stroke)
            return True

        if key == self.args.close_key:
            self.target_stroke = self.args.close_stroke
            rospy.loginfo("Set full CLOSE: %.2f mm", self.target_stroke)
            self.burst_publish(self.target_stroke)
            return True

        return True

    def print_help(self):
        print("")
        print("Gripper Keyboard Teleop (Binary: Open/Close)")
        print(f"  publish topic : {self.args.topic}")
        print("  keys:")
        print("    Space : toggle open/close")
        print(f"    {self.args.open_key} : fully open")
        print(f"    {self.args.close_key} : fully close")
        print(f"    {self.args.quit_key} : quit")
        if self.args.quit_on_enter:
            print("    Enter : quit")
        print("")
        print(
            f"  stroke range : [{self.args.min_stroke:.2f}, {self.args.max_stroke:.2f}] mm, "
            f"open={self.args.open_stroke:.2f}, close={self.args.close_stroke:.2f}"
        )
        print(
            f"  publish rate : {self.args.rate:.1f} Hz, burst={self.args.burst_count} "
            f"(interval={self.args.burst_interval:.4f}s)"
        )
        print(f"  initial state: {self.args.initial_state}")
        print("")

    def run(self) -> int:
        self.print_help()
        if not sys.stdin.isatty():
            rospy.logerr("stdin is not a TTY. Run this script in an interactive terminal.")
            return 2

        rospy.loginfo("Keyboard control started.")
        with RawTerminal():
            while not rospy.is_shutdown():
                key = self.read_key(1.0 / max(self.args.rate, 1.0))
                if key is not None:
                    keep_running = self.apply_key(key)
                    if not keep_running:
                        break
                self.publish_target()
                self.rate.sleep()
        return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fast binary keyboard teleop for gripper position control."
    )
    parser.add_argument("--topic", default="/gripper_position_control_host")
    parser.add_argument("--rate", type=float, default=120.0)
    parser.add_argument("--min-stroke", type=float, default=0.0)
    parser.add_argument("--max-stroke", type=float, default=60.0)
    parser.add_argument("--open-stroke", type=float, default=60.0)
    parser.add_argument("--close-stroke", type=float, default=0.0)
    parser.add_argument(
        "--burst-count",
        type=int,
        default=8,
        help="How many immediate messages to send after a key press.",
    )
    parser.add_argument(
        "--burst-interval",
        type=float,
        default=0.002,
        help="Seconds between burst messages.",
    )
    parser.add_argument(
        "--initial-state",
        choices=("none", "open", "close"),
        default="none",
        help="Optional initial command state at startup.",
    )
    parser.add_argument("--open-key", default="o", help="Key for full open.")
    parser.add_argument("--close-key", default="c", help="Key for full close.")
    parser.add_argument("--quit-key", default="q", help="Key for quit.")
    parser.add_argument(
        "--quit-on-enter",
        action="store_true",
        help="Allow Enter to quit the teleop loop.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args(rospy.myargv()[1:])

    if args.min_stroke > args.max_stroke:
        parser.error("--min-stroke must be <= --max-stroke")
    args.open_stroke = clamp(args.open_stroke, args.min_stroke, args.max_stroke)
    args.close_stroke = clamp(args.close_stroke, args.min_stroke, args.max_stroke)
    if args.rate <= 0:
        parser.error("--rate must be > 0")
    if args.burst_count <= 0:
        parser.error("--burst-count must be > 0")
    if args.burst_interval < 0:
        parser.error("--burst-interval must be >= 0")

    args.open_key = normalize_key(parser, args.open_key, "--open-key")
    args.close_key = normalize_key(parser, args.close_key, "--close-key")
    args.quit_key = normalize_key(parser, args.quit_key, "--quit-key")
    key_map = {
        "--open-key": args.open_key,
        "--close-key": args.close_key,
        "--quit-key": args.quit_key,
    }
    if len(set(key_map.values())) != len(key_map):
        parser.error("Key bindings must be unique.")

    rospy.init_node("a1_gripper_keyboard", anonymous=False)

    try:
        node = GripperKeyboardNode(args)
        code = node.run()
    except rospy.ROSInterruptException:
        code = 0
    sys.exit(code)


if __name__ == "__main__":
    main()
