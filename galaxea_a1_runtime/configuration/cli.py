"""One small CLI contract for typed TOML-to-shell renderers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from galaxea_a1_runtime.console import ArgumentParser

ConfigT = TypeVar("ConfigT")


def run_config_renderer(
    argv: list[str] | None,
    *,
    description: str,
    default_config: Path,
    load_config: Callable[..., ConfigT],
    render_shell: Callable[[ConfigT], str],
) -> int:
    parser = ArgumentParser(description=description)
    parser.add_argument("config", nargs="?", type=Path, default=default_config)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root used to resolve tracked relative paths",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="emit validated shell assignments for a runtime supervisor",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, repo_root=args.repo_root)
    print(render_shell(config) if args.shell else config.path)
    return 0
