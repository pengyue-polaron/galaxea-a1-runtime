# A1 SDK Docker Setup

This setup keeps the host on Ubuntu 22.04 while running the A1 ROS 1 driver
stack inside an Ubuntu 20.04 + ROS Noetic arm64 container.

## What gets prepared

`scripts/collect_data/prepare_a1_sdk_runtime.sh` generates:

- `third_party/A1_SDK_runtime/`

It is built from:

- official arm64 SDK clone at `third_party/A1_SDK_official_arm/`
- DragDataCoach overlay from `third_party/A1_SDK/`

The overlay adds:

- `tools/`
- `mobiman/auto_generated/`
- customized launch files, including `ee_record_only.launch`

## Commands

Prepare the host serial alias first:

```bash
scripts/collect_data/install_a1_udev.sh
```

Then build the runtime SDK and Docker image:

```bash
scripts/collect_data/a1_noetic_docker.sh build
```

Open a shell:

```bash
scripts/collect_data/a1_noetic_docker.sh shell
```

Verify ROS + package resolution:

```bash
scripts/collect_data/a1_noetic_docker.sh doctor
```

Run the driver:

```bash
scripts/collect_data/a1_noetic_docker.sh driver /dev/a1
```

Run the record-only pipeline:

```bash
scripts/collect_data/a1_noetic_docker.sh ee-record /dev/a1
```

## Notes

- The compose service maps `/dev/a1` into the container. Make sure the host has
  the stable symlink first.
- If you want RViz on the host display, allow local X access before entering the
  container:

```bash
xhost +local:
```
- The runtime SDK is generated and should not be edited directly.
