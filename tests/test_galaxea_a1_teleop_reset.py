import pytest

from galaxea_a1_runtime.apps.teleop.reset import run_reset_jobs


class FakeProgress:
    def __init__(self):
        self.results = []

    def finish(self, *, success, detail=None):
        self.results.append((success, detail))


def test_expected_reset_failure_returns_clean_status_with_details():
    progress = FakeProgress()

    def fail():
        raise RuntimeError("joint0.pos error 2.110 > 2.000")

    assert run_reset_jobs({"leader": fail}, progress) == 1
    assert progress.results == [(False, "leader: joint0.pos error 2.110 > 2.000")]


def test_unexpected_reset_error_keeps_traceback_path():
    progress = FakeProgress()

    def fail():
        raise KeyError("programming bug")

    with pytest.raises(KeyError, match="programming bug"):
        run_reset_jobs({"leader": fail}, progress)
    assert progress.results == []
