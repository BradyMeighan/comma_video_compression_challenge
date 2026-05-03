#!/usr/bin/env python
import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb
import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size

def load_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames

if __name__ == '__main__':
    frames = load_frames()
    segment = np.array(frames[0:300]) # 300, H, W, C
    
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in segment]).astype(np.float32)
    temp_std = np.std(gray, axis=0) # H, W
    
    edges_accum = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    for g in gray:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        edges_accum += np.sqrt(gx**2 + gy**2)
    spatial_complexity = edges_accum / len(gray)
    
    print("Temporal Std Percentiles:")
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print(f"  {p}%: {np.percentile(temp_std, p):.2f}")
        
    print("\nSpatial Complexity Percentiles:")
    for p in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print(f"  {p}%: {np.percentile(spatial_complexity, p):.2f}")

    # Let's find the intersection: how many pixels have temp_std < X AND spatial_complexity < Y
    print("\nIntersection area (blur safe area):")
    for t_th in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for s_th in [10.0, 20.0, 30.0, 50.0]:
            mask = (temp_std < t_th) & (spatial_complexity < s_th)
            
            # Exclude hood (bottom 15%)
            hood_start = int(H_CAM * 0.85)
            mask[hood_start:, :] = False
            
            pct = mask.mean() * 100
            print(f"  temp_std < {t_th} AND spatial < {s_th} => {pct:.1f}% of image")
