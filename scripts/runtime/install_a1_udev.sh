#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RULES_SRC="${PROJECT_ROOT}/configs/udev/99-galaxea-a1.rules"
RULES_DST="/etc/udev/rules.d/99-galaxea-a1.rules"
RS_RULES_SRC="${PROJECT_ROOT}/configs/udev/99-realsense-libusb.rules"
RS_RULES_DST="/etc/udev/rules.d/99-realsense-libusb.rules"
MODULES_SRC="${PROJECT_ROOT}/configs/modules-load/cdc_acm.conf"
MODULES_DST="/etc/modules-load.d/cdc_acm.conf"
NEEDS_RELOGIN=0

if [[ ! -f "${RULES_SRC}" ]]; then
  echo "[ERROR] Missing rules template: ${RULES_SRC}"
  exit 1
fi

if [[ ! -f "${MODULES_SRC}" ]]; then
  echo "[ERROR] Missing modules-load template: ${MODULES_SRC}"
  exit 1
fi

if [[ ! -f "${RS_RULES_SRC}" ]]; then
  echo "[ERROR] Missing RealSense rules template: ${RS_RULES_SRC}"
  exit 1
fi

if ! sudo -n true >/dev/null 2>&1; then
  echo "[ERROR] passwordless sudo is required to install udev rules."
  exit 1
fi

echo "[INFO] Installing udev rules:"
echo "       ${RULES_SRC} -> ${RULES_DST}"
sudo install -D -m 0644 "${RULES_SRC}" "${RULES_DST}"

echo "[INFO] Installing RealSense udev rules:"
echo "       ${RS_RULES_SRC} -> ${RS_RULES_DST}"
sudo install -D -m 0644 "${RS_RULES_SRC}" "${RS_RULES_DST}"

echo "[INFO] Installing persistent module load config:"
echo "       ${MODULES_SRC} -> ${MODULES_DST}"
sudo install -D -m 0644 "${MODULES_SRC}" "${MODULES_DST}"

echo "[INFO] Loading cdc_acm now..."
sudo modprobe cdc_acm

ensure_group() {
  local group="$1"
  if ! getent group "${group}" >/dev/null 2>&1; then
    return 0
  fi
  if id -nG "${USER}" | tr ' ' '\n' | grep -qx "${group}"; then
    return 0
  fi
  echo "[INFO] Adding ${USER} to ${group} group..."
  sudo usermod -aG "${group}" "${USER}"
  NEEDS_RELOGIN=1
}

ensure_group dialout
ensure_group video
ensure_group plugdev

echo "[INFO] Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --action=add || true
sudo udevadm trigger --subsystem-match=tty || true
sudo udevadm trigger --subsystem-match=video4linux || true

echo "[INFO] Current serial devices:"
shopt -s nullglob
serial_devices=(/dev/a1 /dev/ttyACM* /dev/ttyUSB*)
shopt -u nullglob
existing_serial_devices=()
for dev in "${serial_devices[@]}"; do
  [[ -e "${dev}" ]] && existing_serial_devices+=("${dev}")
done
if [[ "${#existing_serial_devices[@]}" -gt 0 ]]; then
  ls -l "${existing_serial_devices[@]}"
else
  echo "       No A1 serial device detected yet."
fi

echo "[INFO] Current video devices:"
shopt -s nullglob
video_devices=(/dev/video*)
shopt -u nullglob
if [[ "${#video_devices[@]}" -gt 0 ]]; then
  ls -l "${video_devices[@]}"
else
  echo "       No video devices detected yet."
fi

if [[ "${NEEDS_RELOGIN}" -eq 1 ]]; then
  echo "[INFO] Re-login or open a new shell before accessing device nodes as ${USER}."
fi
