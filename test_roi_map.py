#!/usr/bin/env python
import subprocess, os
from pathlib import Path

# Create a dummy ROI map file
# Format: 1 byte per 64x64 block.
# Video size 524x394 (approx)
# Width in 64x64 blocks: ceil(524/64) = 9
# Height in 64x64 blocks: ceil(394/64) = 7
# Total blocks = 9 * 7 = 63 bytes per frame
# For 1200 frames: 63 * 1200 = 75600 bytes
with open('test_roi.bin', 'wb') as f:
    f.write(b'\x00' * 75600)

cmd = [
    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'info',
    '-f', 'lavfi', '-i', 'color=c=black:s=524x394:d=5',
    '-pix_fmt', 'yuv420p',
    '-c:v', 'libsvtav1', '-preset', '12', '-crf', '33',
    '-svtav1-params', 'roi-map-file=test.roi:keyint=180',
    'test_out.mkv'
]

print("Running:", ' '.join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)
