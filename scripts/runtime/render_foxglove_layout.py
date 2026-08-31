#!/usr/bin/env python3.12
# ruff: noqa: E402
"""Render or verify the tracked read-only Foxglove layout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from galaxea_a1_runtime.configuration.system import SYSTEM_CONFIG, load_system_config
from galaxea_a1_runtime.console import ArgumentParser
from galaxea_a1_runtime.foxglove_layout import render_foxglove_layout


DEFAULT_OUTPUT = ROOT / "foxglove/layouts/a1_observability.json"


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / SYSTEM_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = render_foxglove_layout(load_system_config(args.config, repo_root=ROOT))
    output = args.output.resolve()
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != rendered:
            print(f"Foxglove layout is stale: {output}", file=sys.stderr)
            return 1
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
