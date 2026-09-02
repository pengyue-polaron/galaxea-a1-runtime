#!/usr/bin/env python3.12
# ruff: noqa: E402
"""Render or verify the tracked Foxglove layout and extension configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.configuration.system import SYSTEM_CONFIG, load_system_config
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.foxglove_layout import (
    render_foxglove_extension_config,
    render_foxglove_layout,
)


DEFAULT_OUTPUT = ROOT / "foxglove/layouts/a1_observability.json"
DEFAULT_EXTENSION_OUTPUT = (
    ROOT / "foxglove/extensions/galaxea-a1-collection-console/src/a1Config.ts"
)


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / SYSTEM_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extension-output", type=Path, default=DEFAULT_EXTENSION_OUTPUT
    )
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system = load_system_config(args.config, repo_root=ROOT)
    rendered = render_foxglove_layout(system)
    extension_rendered = render_foxglove_extension_config(system)
    output = args.output.resolve()
    extension_output = args.extension_output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"Foxglove layout is stale: {output}", file=sys.stderr)
            return 1
        if (
            not extension_output.is_file()
            or extension_output.read_text(encoding="utf-8") != extension_rendered
        ):
            print(
                f"Foxglove extension config is stale: {extension_output}",
                file=sys.stderr,
            )
            return 1
        return 0
    _atomic_write(output, rendered)
    _atomic_write(extension_output, extension_rendered)
    print(output)
    print(extension_output)
    return 0


def _atomic_write(output: Path, rendered: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    raise SystemExit(main())
