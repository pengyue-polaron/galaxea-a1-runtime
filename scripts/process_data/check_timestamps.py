import pickle
import numpy as np
from pathlib import Path

demo_dir = Path("/home/jolia/DataCoach/data/processed_data/test/demo_1")

# ---------- Load ----------
with open(demo_dir / "states.pkl", "rb") as f:
    states = pickle.load(f)

with open(demo_dir / "commanded_states.pkl", "rb") as f:
    cmds = pickle.load(f)

with open(demo_dir / "cam_0_rgb_video.metadata", "rb") as f:
    cam_meta = pickle.load(f)


cam_ts = np.array(cam_meta["timestamps"])
state_ts = np.array([s["timestamp"] for s in states])
cmd_ts = np.array([c["timestamp"] for c in cmds])

print("=== lengths ===")
print("cam frames:", len(cam_ts))
print("states    :", len(state_ts))
print("commands  :", len(cmd_ts))

print("\n=== first 5 timestamps ===")
for i in range(5):
    print(
        f"[{i}] cam={cam_ts[i]:.6f}, "
        f"state={state_ts[i]:.6f}, "
        f"cmd={cmd_ts[i]:.6f}"
    )

print("\n=== last 5 timestamps ===")
for i in range(1, 6):
    print(
        f"[-{i}] cam={cam_ts[-i]:.6f}, "
        f"state={state_ts[-i]:.6f}, "
        f"cmd={cmd_ts[-i]:.6f}"
    )
