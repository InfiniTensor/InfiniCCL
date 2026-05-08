#!/bin/bash

if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc" >/dev/null 2>&1
fi

SCRIPT_DIR=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
COMMON_DIR=$(dirname "$SCRIPT_DIR")
RELATIVE_EXE=$1
shift

if [ -c "/dev/nvidia0" ] || [ -x "$(command -v nvidia-smi)" ]; then
    ARCH="nvidia"
elif grep -l "9999" /sys/bus/pci/devices/*/vendor >/dev/null 2>&1 || [ -d "/opt/maca" ]; then
    ARCH="metax"
else
    ARCH="cpu"
fi

FULL_EXE_PATH="${COMMON_DIR}/build/${ARCH}/${RELATIVE_EXE}"

export LD_LIBRARY_PATH="${COMMON_DIR}/build/${ARCH}/src:${LD_LIBRARY_PATH:-}"

exec "$FULL_EXE_PATH" "$@"
