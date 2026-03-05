#!/usr/bin/env python3
"""Real-time monitor for end effector position."""

import rospy
from geometry_msgs.msg import PoseStamped
import sys

# Training reference position
REF_POS = {"x": 0.078, "y": -0.028, "z": 0.238}
TRAINING_RANGE = {
    "x": (0.064, 0.453),
    "y": (-0.295, 0.105),
    "z": (0.073, 0.555),
}

def check_in_range(val, key):
    """Check if value is within training range."""
    min_val, max_val = TRAINING_RANGE[key]
    if val < min_val:
        return f"⚠️  BELOW (min={min_val:.3f})"
    elif val > max_val:
        return f"⚠️  ABOVE (max={max_val:.3f})"
    else:
        return "✓"

def pose_callback(msg):
    """Print position with color coding."""
    pos = msg.pose.position

    # Calculate deltas from reference
    dx = pos.x - REF_POS["x"]
    dy = pos.y - REF_POS["y"]
    dz = pos.z - REF_POS["z"]

    # Check ranges
    x_status = check_in_range(pos.x, "x")
    y_status = check_in_range(pos.y, "y")
    z_status = check_in_range(pos.z, "z")

    # Clear line and print
    sys.stdout.write("\r" + " " * 120 + "\r")
    sys.stdout.write(
        f"x: {pos.x:7.4f} ({dx:+.3f}) {x_status:20s} | "
        f"y: {pos.y:7.4f} ({dy:+.3f}) {y_status:20s} | "
        f"z: {pos.z:7.4f} ({dz:+.3f}) {z_status:20s}"
    )
    sys.stdout.flush()

def main():
    rospy.init_node("eef_position_monitor", anonymous=True)

    print("=" * 120)
    print(f"Training reference: x={REF_POS['x']:.3f}, y={REF_POS['y']:.3f}, z={REF_POS['z']:.3f}")
    print(f"Training range: x=[{TRAINING_RANGE['x'][0]:.3f}, {TRAINING_RANGE['x'][1]:.3f}], "
          f"y=[{TRAINING_RANGE['y'][0]:.3f}, {TRAINING_RANGE['y'][1]:.3f}], "
          f"z=[{TRAINING_RANGE['z'][0]:.3f}, {TRAINING_RANGE['z'][1]:.3f}]")
    print("=" * 120)
    print("Monitoring /end_effector_pose... (Ctrl+C to stop)")
    print()

    rospy.Subscriber("/end_effector_pose", PoseStamped, pose_callback)
    rospy.spin()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        print("\nMonitoring stopped.")
