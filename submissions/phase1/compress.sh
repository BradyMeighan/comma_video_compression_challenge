#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR

CODEC="libsvtav1"
CRF=32
PRESET=0
GOP=180
SCALE_W=512
SCALE_H=384

while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "Encoding ${IN} -> ${OUT}"
  echo "  Codec: ${CODEC}, CRF: ${CRF}, Preset: ${PRESET}, GOP: ${GOP}, Scale: ${SCALE_W}x${SCALE_H}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=${SCALE_W}:${SCALE_H}:flags=lanczos" \
    -c:v ${CODEC} -preset ${PRESET} -crf ${CRF} \
    -g ${GOP} \
    -svtav1-params "enable-overlays=1:film-grain=0:fast-decode=0" \
    -r 20 "$OUT"

  echo "  Done."
done < "$VIDEO_NAMES_FILE"

cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
ZIPSIZE=$(stat --printf='%s' "${HERE}/archive.zip" 2>/dev/null || wc -c < "${HERE}/archive.zip" | tr -d ' ')
echo "Compressed to ${HERE}/archive.zip (${ZIPSIZE} bytes)"
