#!/usr/bin/env python3
import argparse
import math
import shlex
import subprocess
import threading
import time


MIN_DBFS = -120.0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Listen on the local microphone and trigger remote gripper open/close "
            "commands when audio crosses a dBFS threshold."
        )
    )
    parser.add_argument("--ssh-host", required=True, help="SSH host used to reach the robot machine.")
    parser.add_argument(
        "--remote-cwd",
        default="/home/pengyue/Codespace/DataCoach",
        help="Remote repository path used before running just.",
    )
    parser.add_argument(
        "--remote-open-cmd",
        default="just gripper open",
        help="Remote command executed for OPEN.",
    )
    parser.add_argument(
        "--remote-close-cmd",
        default="just gripper close",
        help="Remote command executed for CLOSE.",
    )
    parser.add_argument(
        "--trigger-mode",
        choices=("toggle", "open", "close"),
        default="toggle",
        help="Action to run on each threshold crossing.",
    )
    parser.add_argument(
        "--initial-state",
        choices=("open", "close"),
        default="close",
        help="Used only by toggle mode to decide the first action.",
    )
    parser.add_argument(
        "--threshold-db",
        type=float,
        default=-24.0,
        help="Trigger threshold in dBFS.",
    )
    parser.add_argument(
        "--reset-db",
        type=float,
        default=None,
        help="Re-arm threshold. Defaults to threshold-db - 8.",
    )
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=0.8,
        help="Minimum time between remote triggers.",
    )
    parser.add_argument(
        "--samplerate",
        type=int,
        default=16000,
        help="Microphone sampling rate.",
    )
    parser.add_argument(
        "--block-duration",
        type=float,
        default=0.10,
        help="Seconds per audio block.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional sounddevice input device name/index.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List local audio devices and exit.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="How often to print the current dBFS meter.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of executing ssh.",
    )
    return parser


def import_audio_modules():
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as exc:
        raise SystemExit(
            "Missing local audio dependencies. Install them on your local machine with:\n"
            "  python3 -m pip install numpy sounddevice\n"
            f"Original error: {exc}"
        ) from exc
    return np, sd


def compute_dbfs(np, indata) -> float:
    if indata.size == 0:
        return MIN_DBFS
    samples = np.asarray(indata, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms <= 1e-9:
        return MIN_DBFS
    return max(MIN_DBFS, 20.0 * math.log10(rms))


def shell_quote(command: str) -> str:
    return shlex.quote(command)


class AudioLevelMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest_db = MIN_DBFS

    def set_latest_db(self, value: float):
        with self._lock:
            self._latest_db = value

    def latest_db(self) -> float:
        with self._lock:
            return self._latest_db


def build_ssh_command(args, action: str) -> list[str]:
    remote_inner = args.remote_open_cmd if action == "open" else args.remote_close_cmd
    remote_command = f"cd {shell_quote(args.remote_cwd)} && {remote_inner}"
    return ["ssh", args.ssh_host, remote_command]


def run_remote_command(args, action: str):
    cmd = build_ssh_command(args, action)
    pretty = " ".join(shell_quote(part) for part in cmd)
    print(f"[TRIGGER] {action.upper()} -> {pretty}", flush=True)
    if args.dry_run:
        return
    subprocess.run(cmd, check=True)


def choose_action(trigger_mode: str, current_state: str) -> tuple[str, str]:
    if trigger_mode == "open":
        return "open", "open"
    if trigger_mode == "close":
        return "close", "close"
    next_action = "open" if current_state == "close" else "close"
    return next_action, next_action


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cooldown_seconds < 0:
        parser.error("--cooldown-seconds must be >= 0")
    if args.block_duration <= 0:
        parser.error("--block-duration must be > 0")
    if args.print_interval <= 0:
        parser.error("--print-interval must be > 0")
    if args.reset_db is None:
        args.reset_db = args.threshold_db - 8.0
    if args.reset_db > args.threshold_db:
        parser.error("--reset-db must be <= --threshold-db")

    np, sd = import_audio_modules()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    monitor = AudioLevelMonitor()
    status_messages = []

    def callback(indata, _frames, _time_info, status):
        if status:
            status_messages.append(str(status))
        monitor.set_latest_db(compute_dbfs(np, indata))

    blocksize = max(1, int(args.samplerate * args.block_duration))
    state = args.initial_state
    armed = True
    last_trigger_time = 0.0
    last_print_time = 0.0

    print(
        f"Listening on local mic. threshold={args.threshold_db:.1f} dBFS, "
        f"reset={args.reset_db:.1f} dBFS, mode={args.trigger_mode}, "
        f"initial_state={args.initial_state}",
        flush=True,
    )
    print("Press Ctrl+C to stop.", flush=True)

    try:
        with sd.InputStream(
            samplerate=args.samplerate,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            device=args.device,
            callback=callback,
        ):
            while True:
                now = time.monotonic()
                db = monitor.latest_db()

                if status_messages:
                    print(f"[AUDIO] {status_messages.pop(0)}", flush=True)

                if now - last_print_time >= args.print_interval:
                    print(f"[AUDIO] level={db:.1f} dBFS armed={armed} state={state}", flush=True)
                    last_print_time = now

                if db <= args.reset_db:
                    armed = True

                if armed and db >= args.threshold_db and (now - last_trigger_time) >= args.cooldown_seconds:
                    action, state = choose_action(args.trigger_mode, state)
                    run_remote_command(args, action)
                    armed = False
                    last_trigger_time = now

                time.sleep(0.02)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
