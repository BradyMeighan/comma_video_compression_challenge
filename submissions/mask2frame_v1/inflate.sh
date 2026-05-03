#!/bin/bash
set -e
ARCHIVE_DIR="$1"
INFLATED_DIR="$2"
VIDEO_NAMES_FILE="$3"
python -m submissions.mask2frame_v1.inflate "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES_FILE"
