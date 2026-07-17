#!/usr/bin/env python3
"""Operator entrypoint for the OpenPI pi0.5 server handshake probe."""

from galaxea_a1_runtime.apps.pi05.probe import main


if __name__ == "__main__":
    raise SystemExit(main())
