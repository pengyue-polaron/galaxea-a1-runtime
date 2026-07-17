#!/usr/bin/env python3
"""Verify all registered LingBot inference inputs without opening hardware."""

from galaxea_a1_runtime.apps.lingbot.verify import main


if __name__ == "__main__":
    raise SystemExit(main())
