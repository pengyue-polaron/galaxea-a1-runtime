#!/usr/bin/env python3.12
"""Read-only vendor telemetry entrypoint."""

from galaxea_a1_runtime.apps.observability.legacy import main

if __name__ == "__main__":
    raise SystemExit(main())
