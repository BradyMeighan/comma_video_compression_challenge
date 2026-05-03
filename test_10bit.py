#!/usr/bin/env python
import subprocess, sys, math
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb
import torch
import torch.nn.functional as F
import einops
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames

if __name__ == '__main__':
    frames = load_frames()[:60] # Just 60 frames for speed
    
    out_mkv = ROOT / '_test_10bit.mkv'
    
    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    
    # Encode 10-bit
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p10le',
        '-c:v', 'libsvtav1', '-preset', '4', '-crf', '33', # Preset 4 for speed
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    
    print(f"Encoded 10-bit MKV. Size: {out_mkv.stat().st_size} bytes")
    
    # Decode with PyAV
    import av
    container = av.open(str(out_mkv))
    dec_frames = []
    for frame in container.decode(container.streams.video[0]):
        # What format does PyAV decode 10-bit to?
        print(f"Decoded frame format: {frame.format.name}")
        f_np = yuv420_to_rgb(frame).numpy()
        dec_frames.append(f_np)
    container.close()
    
    f_np = dec_frames[0]
    print(f"RGB shape: {f_np.shape}, dtype: {f_np.dtype}, max: {f_np.max()}, min: {f_np.min()}")
    
    # If the max is ~255, PyAV automatically converted 10-bit YUV to 8-bit RGB.
    # If it's ~1023 or different, we need to handle it.
