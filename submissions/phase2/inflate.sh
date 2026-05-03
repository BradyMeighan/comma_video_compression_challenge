#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC_DIR="${DATA_DIR}/${BASE}"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -d "$SRC_DIR" ] && echo "ERROR: ${SRC_DIR} not found" >&2 && exit 1

  printf "Inflating %s via adversarial decode ... " "$line"
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC_DIR" "$DST"
done < "$FILE_LIST"
