#!/usr/bin/env python3
"""Validate the live LingBot server contract without running inference."""

from galaxea_a1_runtime.apps.lingbot.probe import main


if __name__ == "__main__":
    raise SystemExit(main())
