# A1 serial setup

This document owns persistent device permissions and the stable `/dev/a1`
alias. The tracked rules—not this prose—are the source of truth:
[`99-galaxea-a1.rules`](../configs/udev/99-galaxea-a1.rules) and
[`99-realsense-libusb.rules`](../configs/udev/99-realsense-libusb.rules).

## Inspect

Plug in the controller and leader, then inspect their identities:

```bash
ls -l /dev/ttyACM* /dev/serial/by-id
udevadm info -q property -n /dev/ttyACM0 | \
  rg 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_MODEL=|ID_VENDOR='
```

## Install

```bash
just udev
```

The installer applies the tracked serial and RealSense udev rules plus
`configs/modules-load/cdc_acm.conf`, loads `cdc_acm`, and adds the current user
to `dialout` when needed. Open a new login shell afterward so group membership
applies.

To reload an edited tracked rule manually:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

## Verify

```bash
ls -l /dev/a1 /dev/ttyACM*
id
```

`/dev/a1` should point to the current arm-controller `ttyACM*` device, whose
group is `dialout`; the current user must also be in that group. The symlink
remains stable when the kernel assigns a different port number after replug or
reboot.
