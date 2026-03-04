#!/usr/bin/env python3
"""Monitor ZMQ state and action streams to diagnose inference issues."""
import json
import time
import zmq

STATE_PORT = 5557
ACTION_PORT = 5559

ctx = zmq.Context()

# Subscribe to state
state_sub = ctx.socket(zmq.SUB)
state_sub.setsockopt(zmq.CONFLATE, 1)
state_sub.connect(f"tcp://127.0.0.1:{STATE_PORT}")
state_sub.setsockopt_string(zmq.SUBSCRIBE, "")

# Subscribe to action
action_sub = ctx.socket(zmq.SUB)
action_sub.connect(f"tcp://127.0.0.1:{ACTION_PORT}")
action_sub.setsockopt_string(zmq.SUBSCRIBE, "")

poller = zmq.Poller()
poller.register(state_sub, zmq.POLLIN)
poller.register(action_sub, zmq.POLLIN)

state_count = 0
action_count = 0
last_state = None
last_action = None
start = time.time()

print("Monitoring state:5557 and action:5559 ...")
print("=" * 100)

try:
    while True:
        events = dict(poller.poll(timeout=100))

        if state_sub in events:
            data = state_sub.recv_json()
            state_count += 1
            last_state = data
            if state_count % 50 == 1:
                pos = data.get("pos", "?")
                ori = data.get("ori", "?")
                grip = data.get("gripper", "?")
                ts = data.get("timestamp", 0)
                age = time.time() - ts if ts else -1
                print(f"[STATE #{state_count:>5}] pos={pos} ori={ori} grip={grip:.1f} age={age:.3f}s")

        if action_sub in events:
            data = action_sub.recv_json()
            action_count += 1
            last_action = data
            pos = data.get("pos", "?")
            ori = data.get("ori", "?")
            grip = data.get("gripper", "?")
            ts = data.get("timestamp", 0)
            age = time.time() - ts if ts else -1

            # Compare action to last state
            delta_info = ""
            if last_state:
                sp = last_state.get("pos", (0, 0, 0))
                ap = pos
                dx = ap[0] - sp[0]
                dy = ap[1] - sp[1]
                dz = ap[2] - sp[2]
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                delta_info = f" delta_pos={dist:.4f}m ({dx:.4f},{dy:.4f},{dz:.4f})"

            print(f"[ACTION #{action_count:>4}] pos={pos} grip={grip:.1f} age={age:.3f}s{delta_info}")

        elapsed = time.time() - start
        if elapsed > 0 and int(elapsed) % 10 == 0 and int(elapsed) > 0:
            if int(elapsed * 10) % 100 == 0:
                state_hz = state_count / elapsed
                action_hz = action_count / elapsed
                print(f"--- {elapsed:.0f}s: state={state_hz:.1f}Hz action={action_hz:.2f}Hz ---")

except KeyboardInterrupt:
    elapsed = time.time() - start
    print(f"\n{'=' * 100}")
    print(f"Total: {elapsed:.1f}s, states={state_count} ({state_count/max(elapsed,1):.1f}Hz), actions={action_count} ({action_count/max(elapsed,1):.2f}Hz)")
