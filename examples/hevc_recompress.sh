#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/.." && pwd)"
LIST="${PD}/test_video_names.txt"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <crf> <in_dir> <jobs>" >&2
  exit 2
fi

CRF="$1"
IN_DIR="${2%/}"
JOBS="$3"

TMPDIR="$(mktemp -d)"
OUT_ZIP="$PD/comma2k19_submission.zip"

export CRF IN_DIR TMPDIR

xargs -a "$LIST" -n1 -P"$JOBS" -I{} bash -lc '
  rel="$1"
  [[ -z "$rel" ]] && exit 0

  IN="${IN_DIR}/${rel}"
  base="${rel##*/}"
  OUT="${TMPDIR}/${rel}"
  mkdir -p "$(dirname "$OUT")"

  echo "→ ${IN}  CRF=${CRF}  →  ${OUT}"
  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -c:v libx265 -preset fast -crf "$CRF" \
    -g 1 -bf 0 -x265-params "keyint=1:min-keyint=1:scenecut=0:pools=8:frame-threads=1:log-level=warning" \
    -r 20 -f hevc "$OUT"
' _ {}

rm -f "$OUT_ZIP"
(
  cd "$TMPDIR"
  zip -r "$OUT_ZIP" .
)
rm -rf "$TMPDIR"
echo "All done. Saved $OUT_ZIP"
