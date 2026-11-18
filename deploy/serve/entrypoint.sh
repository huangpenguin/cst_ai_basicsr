#!/bin/bash
# Adapted from lab-docker images/dl-gpu/lab-gpu-entrypoint.sh and common/lab-entrypoint.sh.
# Map NVIDIA device groups, then drop to app. Never change host device modes.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    exec "$@"
fi

for device in /dev/nvidia[0-9]* /dev/nvidiactl /dev/nvidia-uvm; do
    [[ -c "$device" ]] || continue
    device_gid="$(stat -c %g "$device")"
    # Do not grant membership in the root group.
    [[ "$device_gid" -ne 0 ]] || continue
    device_group="app-gpu-${device_gid}"
    if ! getent group "$device_group" >/dev/null; then
        groupadd -o --gid "$device_gid" "$device_group"
    fi
    usermod -aG "$device_group" app
done

export HOME=/home/app USER=app LOGNAME=app
cd /workspace
exec setpriv --reuid=app --regid=app --init-groups -- "$@"
