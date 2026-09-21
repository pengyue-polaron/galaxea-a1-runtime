"""ROS-free, integer-clock alignment of recorded A1 observations and commands."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from galaxea_a1_runtime.gripper import normalize_stroke
from galaxea_a1_runtime.runtime.ros_feedback import ordered_joint_positions

NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class AlignedSample:
    state: tuple[float, ...]
    action: tuple[float, ...]
    images: dict[str, int]
    timing: dict[str, int | bool]


class Timeline:
    def __init__(self, records, *, role, max_age_s, start_ns, end_ns, values=None):
        if not records:
            raise ValueError(f"bag has no {role} samples")
        self.records = records
        self.role = role
        self.max_age_ns = round(max_age_s * NS_PER_SECOND)
        self.times = []
        self.values = []
        clock = None
        previous_received = -1
        for item in records:
            source, received = item["source_ns"], item["receive_ns"]
            if source < 0 or received <= 0 or received < previous_received:
                raise ValueError(f"{role} has invalid/reversed receive clock")
            previous_received = received
            fallback = source == 0
            if fallback and role in ("front", "wrist", "depth"):
                raise ValueError(f"{role} image has no source timestamp")
            if clock is not None and clock != fallback:
                raise ValueError(f"{role} changed timestamp basis within episode")
            clock = fallback
            effective = received if fallback else source
            # Preparation/teardown messages outside this window cannot supply a
            # bounded endpoint for any output frame. Keep full-stream clock and
            # value validation, but do not reject an episode for their latency.
            can_supply_frame = (
                start_ns - self.max_age_ns <= effective < end_ns + self.max_age_ns
            )
            if can_supply_frame and abs(received - effective) > self.max_age_ns:
                raise ValueError(
                    f"{role} source and receive times exceed freshness bound: "
                    f"age={abs(received - effective) / NS_PER_SECOND:.3f}s, "
                    f"max={max_age_s:.3f}s, index={item['index']}"
                )
            if self.times and (
                effective < self.times[-1]
                or (effective == self.times[-1] and role in ("front", "wrist", "depth"))
            ):
                raise ValueError(f"{role} source clock repeated or reversed")
            self.times.append(effective)
            if values is not None:
                value = np.asarray(values(item), dtype=np.float64)
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"{role} contains non-finite data")
                self.values.append(value)

    def nearest(self, stamp):
        pos = bisect_left(self.times, stamp)
        candidates = [i for i in (pos - 1, pos) if 0 <= i < len(self.times)]
        index = min(
            candidates, key=lambda i: (abs(self.times[i] - stamp), self.times[i])
        )
        if abs(self.times[index] - stamp) > self.max_age_ns:
            raise ValueError(f"{self.role} gap exceeds freshness at {stamp}")
        return index

    def timing(self, index, prefix):
        item = self.records[index]
        return {
            f"{prefix}_source_ns": item["source_ns"],
            f"{prefix}_receive_ns": item["receive_ns"],
            f"{prefix}_receive_fallback": item["source_ns"] == 0,
            f"{prefix}_message_index": item["index"],
        }

    def at(self, stamp, *, hold=False):
        left = bisect_right(self.times, stamp) - 1
        if left < 0 or stamp - self.times[left] > self.max_age_ns:
            raise ValueError(f"{self.role} has no fresh causal sample at {stamp}")
        right = left if hold or self.times[left] == stamp else left + 1
        if right >= len(self.times) or self.times[right] - stamp > self.max_age_ns:
            raise ValueError(
                f"{self.role} has no bounded interpolation endpoint at {stamp}"
            )
        if right != left and self.times[right] - self.times[left] > self.max_age_ns:
            raise ValueError(f"{self.role} interpolation interval exceeds freshness")
        fraction = (
            0.0
            if right == left
            else (stamp - self.times[left]) / (self.times[right] - self.times[left])
        )
        value = self.values[left] * (1.0 - fraction) + self.values[right] * fraction
        if self.role == "eef" and right != left:
            value[3:7] = slerp(
                self.values[left][3:7], self.values[right][3:7], fraction
            )
        return tuple(float(x) for x in value), {
            **self.timing(left, f"{self.role}_left"),
            **self.timing(right, f"{self.role}_right"),
        }


def slerp(left, right, fraction):
    a, b = np.array(left), np.array(right)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        value = a + fraction * (b - a)
        return value / np.linalg.norm(value)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(
        angle
    )


def align_records(records, *, config, start_ns, end_ns):
    """Sample a physical-time grid; gaps fail, camera reuse is explicit."""
    # Preserve operator boundaries: never silently shorten missing coverage.
    if (
        not isinstance(start_ns, int)
        or not isinstance(end_ns, int)
        or end_ns <= start_ns
    ):
        raise ValueError("invalid episode boundaries")
    system = config.system

    def joint(item):
        return ordered_joint_positions(
            SimpleNamespace(name=item["names"], position=item["values"]),
            system.joint_safety.names,
            label=item["role"],
            allow_unnamed=False,
        )

    def gripper(item):
        value = ordered_joint_positions(
            SimpleNamespace(name=item["names"], position=item["values"]),
            ("gripper_stroke_mm",),
            label=item["role"],
            allow_unnamed=False,
        )[0]
        return [
            normalize_stroke(
                value,
                stroke_min_mm=system.gripper.stroke_min_mm,
                stroke_max_mm=system.gripper.stroke_max_mm,
            )
        ]

    def pose(item):
        value = np.array(item["values"], dtype=float)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise ValueError("invalid EEF pose")
        norm = np.linalg.norm(value[3:7])
        if norm < system.eef.min_quat_norm:
            raise ValueError("invalid EEF quaternion")
        value[3:7] /= norm
        return value

    timelines = {}
    for role, samples in records.items():
        image = role in ("front", "wrist", "depth")
        age = (
            system.cameras.max_age_s
            if image
            else system.eef.max_feedback_age_s
            if role == "eef"
            else system.joint_safety.max_feedback_age_s
        )
        decoder = (
            None
            if image
            else pose
            if role == "eef"
            else gripper
            if role.startswith("gripper")
            else joint
        )
        timelines[role] = Timeline(
            samples,
            role=role,
            max_age_s=age,
            start_ns=start_ns,
            end_ns=end_ns,
            values=decoder,
        )
    front, wrist = timelines["front"], timelines["wrist"]
    skew_limit = round(system.cameras.max_pair_skew_s * NS_PER_SECOND)
    pairs = []
    for index, stamp in enumerate(front.times):
        try:
            wi = wrist.nearest(stamp)
        except ValueError:
            continue
        if abs(wrist.times[wi] - stamp) <= skew_limit:
            pairs.append((stamp, index, wi))
    if not pairs:
        raise ValueError("bag has no synchronized camera pairs")
    pair_times = [p[0] for p in pairs]
    fps = int(config.collection.fps)
    previous = {}
    frame = 0
    while (stamp := start_ns + frame * NS_PER_SECOND // fps) < end_ns:
        pos = bisect_left(pair_times, stamp)
        choices = [i for i in (pos - 1, pos) if 0 <= i < len(pairs)]
        chosen = min(choices, key=lambda i: (abs(pair_times[i] - stamp), pair_times[i]))
        _, fi, wi = pairs[chosen]
        image_indices = {"front": fi, "wrist": wi}
        if "depth" in timelines:
            di = timelines["depth"].nearest(front.times[fi])
            if abs(timelines["depth"].times[di] - front.times[fi]) > skew_limit:
                raise ValueError("depth/color source timestamps exceed pair tolerance")
            image_indices["depth"] = di
        timing = {"sample_time_ns": stamp, "grid_index": frame}
        images = {}
        for role, index in image_indices.items():
            line = timelines[role]
            if abs(line.times[index] - stamp) > line.max_age_ns:
                raise ValueError(f"{role} has an unfillable gap at {stamp}")
            images[role] = line.records[index]["index"]
            timing.update(line.timing(index, role))
            timing[f"{role}_reused"] = previous.get(role) == images[role]
            timing[f"{role}_offset_ns"] = line.times[index] - stamp
        previous = images
        timing["camera_pair_skew_ns"] = abs(front.times[fi] - wrist.times[wi])
        values = {}
        for role in ("eef", "joints", "gripper", "action", "gripper_action"):
            values[role], stamps = timelines[role].at(
                stamp, hold=role in ("action", "gripper_action")
            )
            timing.update(stamps)
        yield AlignedSample(
            (*values["eef"], *values["joints"], *values["gripper"]),
            (*values["action"], *values["gripper_action"]),
            images,
            timing,
        )
        frame += 1
