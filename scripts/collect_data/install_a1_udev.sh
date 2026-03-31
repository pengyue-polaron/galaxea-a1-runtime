#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RULES_SRC="${PROJECT_ROOT}/configs/udev/99-datacoach-a1.rules"
RULES_DST="/etc/udev/rules.d/99-datacoach-a1.rules"
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

if ! sudo -n true >/dev/null 2>&1; then
  echo "[ERROR] passwordless sudo is required to install udev rules."
  exit 1
fi

echo "[INFO] Installing udev rules:"
echo "       ${RULES_SRC} -> ${RULES_DST}"
sudo install -D -m 0644 "${RULES_SRC}" "${RULES_DST}"

echo "[INFO] Installing persistent module load config:"
echo "       ${MODULES_SRC} -> ${MODULES_DST}"
sudo install -D -m 0644 "${MODULES_SRC}" "${MODULES_DST}"

echo "[INFO] Loading cdc_acm now..."
sudo modprobe cdc_acm

if ! id -nG "${USER}" | tr ' ' '\n' | grep -qx dialout; then
  echo "[INFO] Adding ${USER} to dialout group..."
  sudo usermod -aG dialout "${USER}"
  NEEDS_RELOGIN=1
fi

echo "[INFO] Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=usb --action=add || true
sudo udevadm trigger --subsystem-match=tty || true

echo "[INFO] Current serial devices:"
shopt -s nullglob
serial_devices=(/dev/a1 /dev/ttyACM* /dev/ttyUSB*)
shopt -u nullglob
if [[ "${#serial_devices[@]}" -gt 0 ]]; then
  ls -l "${serial_devices[@]}"
else
  echo "       No A1 serial device detected yet."
fi

if [[ "${NEEDS_RELOGIN}" -eq 1 ]]; then
  echo "[INFO] Re-login or open a new shell before accessing /dev/a1 as ${USER}."
fi
