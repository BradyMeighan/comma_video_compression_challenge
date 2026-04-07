#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

IN_DIR="${ROOT}/videos"
VIDEO_NAMES_FILE="${ROOT}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  VIDEO_PATH="${IN_DIR}/${line}"
  OUTPUT_SUBDIR="${ARCHIVE_DIR}/${BASE}"

  echo "→ Encoding ${line} → ${OUTPUT_SUBDIR}"
  cd "$ROOT"
  python -m submissions.adversarial_decode.encode "$VIDEO_PATH" "$OUTPUT_SUBDIR"
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
ls -lh "${HERE}/archive.zip"
