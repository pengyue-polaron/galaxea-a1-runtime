"""Reset only A1 to one tracked pose through the staged safe runtime."""

from __future__ import annotations

from pathlib import Path

from galaxea_a1_runtime.apps.reset.config import load_a1_home_pose
from galaxea_a1_runtime.apps.reset.progress import ResetProgress
from galaxea_a1_runtime.configuration.paths import SYSTEM_CONFIG
from galaxea_a1_runtime.configuration.system import load_system_config
from galaxea_a1_runtime.console import ArgumentParser, success


ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--system-config", type=Path, default=ROOT / SYSTEM_CONFIG)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    system = load_system_config(args.system_config, repo_root=ROOT)
    pose = load_a1_home_pose(args.pose, system=system, repo_root=ROOT)

    if args.validate_only:
        success(f"Valid A1 reset pose: {pose.path}")
        return 0

    from galaxea_a1_runtime.apps.reset.runner import A1HomeRunner

    progress = ResetProgress(("A1",))
    try:
        A1HomeRunner(pose, progress).run()
    except BaseException:
        progress.finish(success=False)
        raise
    progress.finish(success=True)
    return 0
