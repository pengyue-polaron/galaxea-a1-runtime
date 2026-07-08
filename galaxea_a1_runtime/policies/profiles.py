"""Policy profile manifest for supported LeRobot v0.6 policy families."""

from __future__ import annotations

from dataclasses import dataclass

from galaxea_a1_runtime.schema import ActionMode


@dataclass(frozen=True)
class PolicyProfile:
    name: str
    lerobot_policy_type: str
    action_mode: ActionMode
    install_extra: str
    default_checkpoint: str
    notes: str


POLICY_PROFILES: dict[str, PolicyProfile] = {
    "lingbot-va": PolicyProfile(
        name="lingbot-va",
        lerobot_policy_type="lingbot_va",
        action_mode=ActionMode.EEF_DELTA,
        install_extra="lingbot_va",
        default_checkpoint="lerobot/lingbot_va_base",
        notes="Autoregressive video-action policy; output is normalized through the A1 action gate.",
    ),
    "fastwam": PolicyProfile(
        name="fastwam",
        lerobot_policy_type="fastwam",
        action_mode=ActionMode.EEF_DELTA,
        install_extra="fastwam",
        default_checkpoint="lerobot/fastwam_base",
        notes="World-action model; inference emits action chunks and still passes through A1 safety.",
    ),
    "groot-n17": PolicyProfile(
        name="groot-n17",
        lerobot_policy_type="groot",
        action_mode=ActionMode.EEF_DELTA,
        install_extra="groot",
        default_checkpoint="nvidia/GR00T-N1.7",
        notes="Use relative EEF action convention where possible for cross-embodiment compatibility.",
    ),
}


def get_policy_profile(name: str) -> PolicyProfile:
    key = name.lower()
    if key not in POLICY_PROFILES:
        raise KeyError(f"unknown policy profile: {name}")
    return POLICY_PROFILES[key]
