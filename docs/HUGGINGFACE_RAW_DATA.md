# Legacy Hugging Face Raw Data Packaging

This workflow packages existing Raw v3 recordings only. New formal collection
writes LeRobotDataset v3 directly and does not use this format.

The tracked raw-data package config is
`configs/datasets/fruit_placement_raw.toml`. It identifies immutable raw v3
source task directories and a separate ignored output under `data/exports/`.

Build the local review package with:

```bash
.venv/bin/python -m galaxea_a1_runtime.datasets.raw_package \
  --config configs/datasets/fruit_placement_raw.toml
```

The builder validates and hashes the source trees, copies them into an isolated
hidden sibling staging directory, and creates archives only from that copy. It
extracts and byte-validates every archive, re-hashes the original source trees,
and atomically installs the package only after all checks pass. A stale staging
directory from a crash blocks a retry until an operator inspects it.

The output includes a draft Hugging Face dataset card at `README.md`. Review and
edit that card before creating the remote dataset repository or uploading any
archive. Packaging does not perform either remote operation.
