#!/usr/bin/env bash
set -euo pipefail

ARCHIVE_DIR="$1"
INFLATED_DIR="$2"
VIDEO_NAMES_FILE="$3"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$INFLATED_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  DATA_DIR="${ARCHIVE_DIR}/${BASE}"
  RAW_PATH="${INFLATED_DIR}/${BASE}.raw"
  echo "Inflating ${line} -> ${RAW_PATH}"
  python "${HERE}/inflate.py" "$DATA_DIR" "$RAW_PATH"
done < "$VIDEO_NAMES_FILE"
