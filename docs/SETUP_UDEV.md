# A1 Serial Setup (udev)

This setup keeps serial permissions persistent across reboot/replug, without custom device aliases.

## 1) Inspect device identity

Plug in devices, then:

```bash
ls -l /dev/ttyACM* /dev/serial/by-id
udevadm info -q property -n /dev/ttyACM0 | rg "ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_MODEL=|ID_VENDOR="
```

In this project machine:

- arm controller: `0483:5740` (`STM32 Virtual ComPort`)
- secondary serial: `1a86:55d3` (`USB Single Serial`)

## 2) Install udev rules (no alias)

Create `/etc/udev/rules.d/99-datacoach-a1.rules`:

```udev
# DataCoach: serial permissions for A1 setup (no custom symlink)
# Arm controller (STM32 Virtual ComPort)
SUBSYSTEM=="tty", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5740", ATTRS{serial}=="336635623233", MODE:="0660", GROUP:="dialout", TAG+="uaccess", ENV{ID_MM_DEVICE_IGNORE}="1"

# Secondary USB serial dongle seen on this machine
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", ATTRS{serial}=="5A7A016967", MODE:="0660", GROUP:="dialout", TAG+="uaccess", ENV{ID_MM_DEVICE_IGNORE}="1"
```

Then reload:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty
```

## 3) Add user to serial group

```bash
sudo usermod -aG dialout $USER
```

Open a new terminal (or re-login) to ensure group change applies in your shell.

## 4) Verify

```bash
ls -l /dev/ttyACM*
id
```

Expected:

- tty device owner/group includes `dialout`
- your user is in `dialout`

## 5) Run driver

`dragdatacoach.sh` defaults to `/dev/ttyACM0`:

```bash
scripts/collect_data/dragdatacoach.sh launch-driver
```

You can still override explicitly:

```bash
scripts/collect_data/dragdatacoach.sh launch-driver /dev/ttyACM1
```
