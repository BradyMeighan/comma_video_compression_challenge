#!/usr/bin/env python
"""
Temporal Variance-based ROI Preprocessing.

Following the user's insight: "The most compressible regions are those that are 
both spatially simple AND temporally static. The dark nighttime sky in this video 
is the prime example... "

We compute the temporal variance for segments. Regions with near-zero variance
are completely static (like the sky). We blur these regions to save bits.
To protect PoseNet's ego-vehicle reference, we explicitly exclude the car hood.
"""
import subprocess, sys, math, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file
import einops

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

TMP = ROOT / '_tvar_roi_tmp'


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def encode_piped(frames, out_mkv, crf='33'):
    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', crf,
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def decode_and_eval(mkv_path, archive_size, segnet, posenet, unsharp=0.44):
    import av
    from PIL import Image
    container = av.open(str(mkv_path))
    decoded = []
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        decoded.append(f_np)
    container.close()

    raw = bytearray()
    for f_np in decoded:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())

    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw_np = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(N*2, H_CAM, W_CAM, 3)
    seg_dists, pose_dists = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw_np[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw_np[2*i+1:2*end:2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, 'b t h w c -> b t c h w')
            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            gt_seg = gt['seg'][i:end].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())
            pn_out = posenet(posenet.preprocess_input(x))['pose'][:, :6]
            gt_pose = gt['pose'][i:end].to(DEVICE)
            pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())
    s, p = np.mean(seg_dists), np.mean(pose_dists)
    r = archive_size / 37_545_489
    score = 100*s + math.sqrt(10*p) + 25*r
    return score, s, p, r


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    TMP.mkdir(exist_ok=True)
    out_mkv = TMP / '0.mkv'

    orig_frames = load_frames()
    print(f"Loaded {len(orig_frames)} frames.", flush=True)

    K = 4
    seg_len = len(orig_frames) // K
    
    def test(name, variance_thresh, blur_sigma, exclude_hood=True, crf='33'):
        processed_frames = []
        ksize = int(blur_sigma * 6) | 1
        
        for k in range(K):
            start_idx = k * seg_len
            end_idx = len(orig_frames) if k == K - 1 else (k + 1) * seg_len
            
            # Compute temporal standard deviation for the segment
            segment_gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) 
                                     for f in orig_frames[start_idx:end_idx]]).astype(np.float32)
            temp_std = np.std(segment_gray, axis=0)
            
            # Create mask: 1.0 means BLUR (variance < threshold), 0.0 means KEEP
            mask = (temp_std < variance_thresh).astype(np.float32)
            
            if exclude_hood:
                # Force hood (bottom 15%) to KEEP (mask=0.0)
                hood_start = int(H_CAM * 0.85)
                mask[hood_start:, :] = 0.0
                
            # Smooth the mask heavily so we don't have sharp artificial edges
            mask = cv2.GaussianBlur(mask, (91, 91), 30.0)
            
            cv2.imwrite(str(TMP / f'mask_{name}_seg{k}.png'), (mask * 255).astype(np.uint8))
            
            alpha = mask[..., np.newaxis]
            
            for i in range(start_idx, end_idx):
                f = orig_frames[i]
                blurred = cv2.GaussianBlur(f, (ksize, ksize), blur_sigma)
                processed = (f * (1.0 - alpha) + blurred * alpha).astype(np.uint8)
                processed_frames.append(processed)
                
        sz = encode_piped(processed_frames, out_mkv, crf=crf)
        score, s, p, r = decode_and_eval(out_mkv, sz, segnet, posenet)
        print(f"[{name:35s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={sz//1024}KB", flush=True)
        return score

    print("\n=== Testing Temporal Variance ROI ===", flush=True)
    
    # Test different variance thresholds
    test("var_th2.0_s5.0", 2.0, 5.0)
    test("var_th3.0_s5.0", 3.0, 5.0)
    test("var_th4.0_s5.0", 4.0, 5.0)
    test("var_th5.0_s5.0", 5.0, 5.0)
    
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
