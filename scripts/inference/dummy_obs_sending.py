# dummy_state_publisher.py

import zmq
import time
import random


ZMQ_STATE_PORT = 5557  # ⚠️ 改成你的 ZMQ_STATE_PORT


def main():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://127.0.0.1:{ZMQ_STATE_PORT}")

    print(f"[DummyPublisher] PUB bound to tcp://127.0.0.1:{ZMQ_STATE_PORT}")

    # IMPORTANT: allow subscriber to connect
    time.sleep(0.5)

    t = 0
    while True:
        print(f"[DummyPublisher] PUB bound to tcp://127.0.0.1:{ZMQ_STATE_PORT}")
        dummy_state = {
            "timestamp": time.time(),
            "pos": (
                0.1 * random.uniform(-1, 1),
                0.1 * random.uniform(-1, 1),
                0.3 + 0.05 * random.uniform(-1, 1),
            ),
            "ori": (
                0.0,
                0.0,
                0.0,
                1.0,  # identity quaternion
            ),
            "gripper": random.uniform(0.0, 1.0),
        }

        socket.send_json(dummy_state)

        print(f"[DummyPublisher] Sent state {t}")
        print(dummy_state)
        print("------")

        t += 1
        time.sleep(0.1)  # 10 Hz


if __name__ == "__main__":
    main()
