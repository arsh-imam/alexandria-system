#!/usr/bin/env bash
# The frozen modules use absolute paths under /media/pi/KINGSTON/local_ai.
# Rather than edit frozen code, recreate that path with a bind mount so the
# published modules stay byte-identical to the ones that were evaluated.
#
#   sudo ./setup_paths.sh /your/data/dir
set -euo pipefail
SRC="${1:?usage: setup_paths.sh /path/holding/local_ai}"
TARGET=/media/pi/KINGSTON
[ -d "$SRC/local_ai" ] || { echo "no local_ai/ under $SRC"; exit 1; }
mkdir -p "$TARGET"
mountpoint -q "$TARGET" && { echo "$TARGET already a mountpoint"; exit 0; }
mount --bind "$SRC" "$TARGET"
echo "bound $SRC -> $TARGET"
echo
echo "to persist, append to /etc/fstab:"
echo "  $SRC  $TARGET  none  bind  0  0"
