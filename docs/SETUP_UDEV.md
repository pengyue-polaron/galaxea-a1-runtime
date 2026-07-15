# A1 Serial Setup (udev)

This setup keeps serial permissions persistent across reboot/replug, and creates stable `/dev/a1` symlink so the device name won't change.

## 1) Inspect device identity

Plug in devices, then:

```bash
ls -l /dev/ttyACM* /dev/serial/by-id
udevadm info -q property -n /dev/ttyACM0 | rg "ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_MODEL=|ID_VENDOR="
```

In this project machine:

- arm controller: `0483:5740` (`STM32 Virtual ComPort`)
- secondary serial: `1a86:55d3` (`USB Single Serial`)

## 2) Install udev rules

The rules file is tracked in this repo at
`configs/udev/99-galaxea-a1.rules`.

The USB CDC ACM module autoload file is tracked at
`configs/modules-load/cdc_acm.conf`.

Install it with:

```bash
scripts/runtime/install_a1_udev.sh
```

The installed rule content is:

```udev
# Galaxea A1 arm controller (STM32 Virtual ComPort) -> /dev/a1
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", ATTRS{serial}=="326D39703133", MODE:="0660", GROUP:="dialout", TAG+="uaccess", ENV{ID_MM_DEVICE_IGNORE}="1", SYMLINK+="a1"

# Secondary USB serial dongle seen on this machine
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5A7A016967", MODE:="0660", GROUP:="dialout", TAG+="uaccess", ENV{ID_MM_DEVICE_IGNORE}="1"

# Legacy alternate global-shutter camera -> OpenCV access without the video group
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="32e4", ATTRS{idProduct}=="2234", ATTRS{serial}=="01.00.00", MODE:="0666", GROUP:="plugdev"
```

The tracked default wrist camera is now the D405 RealSense selected by serial
in `configs/system/a1.toml`; it does not use the legacy global-shutter rule.
Keep that rule only for an explicitly tracked V4L2 alternate setup.

If you need to reload manually:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

## 3) Add user to serial group

`install_a1_udev.sh` also:

- adds the current user to `dialout` if needed
- installs `/etc/modules-load.d/cdc_acm.conf`
- runs `modprobe cdc_acm` so STM32 virtual serial devices enumerate as `ttyACM*`

Open a new terminal (or re-login) to ensure the group change applies in your shell.

## 4) Verify

```bash
ls -l /dev/a1 /dev/ttyACM*
id
```

Expected:

- `/dev/a1` is a symlink pointing to `ttyACM0` (or whichever port the arm is on)
- tty device owner/group includes `dialout`
- your user is in `dialout`

## 5) Run driver

Use the stable `/dev/a1` alias:

```bash
just eef-test
```

The alias is stable across reboot/replug — no matter which `ttyACM*` number the
kernel assigns, `/dev/a1` always points to the arm controller.
