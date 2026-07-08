from galaxea_a1_runtime.policies.profiles import POLICY_PROFILES, get_policy_profile
from galaxea_a1_runtime.schema import ActionMode


def test_policy_profiles_cover_target_models():
    assert {"lingbot-va", "fastwam", "groot-n17"} <= set(POLICY_PROFILES)


def test_groot_profile_uses_eef_delta_contract():
    profile = get_policy_profile("groot-n17")

    assert profile.lerobot_policy_type == "groot"
    assert profile.action_mode == ActionMode.EEF_DELTA
