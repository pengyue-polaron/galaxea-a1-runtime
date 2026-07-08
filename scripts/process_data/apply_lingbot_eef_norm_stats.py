#!/usr/bin/env python3
"""Apply computed Galaxea A1 EE norm stats to LingBot-VA's local config."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


def default_config_path() -> str:
    env_path = os.environ.get("LINGBOT_VA_A1_CONFIG")
    if env_path:
        return env_path
    candidates = [
        Path.home() / "lingbot-va" / "wan_va" / "configs" / "va_galaxea_a1_cfg.py",
        Path("/home/pengyue/lingbot-va/wan_va/configs/va_galaxea_a1_cfg.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def format_float_list(values: list[float], indent: str = "        ") -> str:
    chunks = []
    for i in range(0, len(values), 7):
        line = indent + ", ".join(repr(float(v)) for v in values[i : i + 7])
        if i + 7 < len(values):
            line += ","
        chunks.append(line)
    return "[\n" + "\n".join(chunks) + "\n    ]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch va_galaxea_a1_cfg.py with EE q01/q99 stats.")
    parser.add_argument("stats_json", help="JSON produced by compute_eef_norm_stats_from_bags.py")
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help="LingBot Galaxea A1 config to patch. Defaults to LINGBOT_VA_A1_CONFIG or ~/lingbot-va/wan_va/configs/va_galaxea_a1_cfg.py.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print replacement block without writing.")
    args = parser.parse_args()

    stats_path = Path(args.stats_json)
    cfg_path = Path(args.config)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    norm = stats.get("lingbot_norm_stat", stats.get("norm_stat"))
    if not norm or "q01" not in norm or "q99" not in norm:
        raise SystemExit(f"No lingbot_norm_stat.q01/q99 found in {stats_path}")

    q01 = [float(v) for v in norm["q01"]]
    q99 = [float(v) for v in norm["q99"]]
    if len(q01) != 30 or len(q99) != 30:
        raise SystemExit(f"Expected 30D q01/q99, got {len(q01)} and {len(q99)}")

    block = (
        "va_galaxea_a1_cfg.norm_stat = {\n"
        f"    \"q01\": {format_float_list(q01)},\n"
        f"    \"q99\": {format_float_list(q99)},\n"
        "}"
    )

    text = cfg_path.read_text(encoding="utf-8")
    new_text, count = re.subn(
        r"va_galaxea_a1_cfg\.norm_stat\s*=\s*\{.*?\n\}",
        block,
        text,
        count=1,
        flags=re.S,
    )
    if count != 1:
        raise SystemExit(f"Could not find a single norm_stat block in {cfg_path}")

    print(block)
    if args.dry_run:
        print(f"[dry-run] Would patch {cfg_path} from {stats_path}")
        return
    cfg_path.write_text(new_text, encoding="utf-8")
    print(f"[ok] Patched {cfg_path} from {stats_path}")


if __name__ == "__main__":
    main()
