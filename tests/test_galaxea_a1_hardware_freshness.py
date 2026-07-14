import pytest

from galaxea_a1_runtime.hardware.freshness import LatestMessageCache


def test_latest_message_cache_rejects_stale_messages():
    now = [10.0]
    cache = LatestMessageCache[str](clock=lambda: now[0])
    cache.callback("fresh")

    assert cache.get(max_age_s=0.5) == "fresh"
    now[0] = 10.51
    assert cache.get(max_age_s=0.5) is None


def test_latest_message_cache_requires_positive_deadline():
    cache = LatestMessageCache[str]()

    with pytest.raises(ValueError, match="positive"):
        cache.get(max_age_s=0.0)
