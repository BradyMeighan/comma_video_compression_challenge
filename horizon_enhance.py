#!/usr/bin/env python
"""
Horizon-targeted decode enhancement.

Key insight: 5.66% of vehicle pixels (Class 1) get misclassified as sky (Class 0).
Vehicles are near the horizon where contrast is lowest after compression.

Approach:
1. Boost local contrast at the horizon band
2. Apply CLAHE (adaptive histogram equalization) to the horizon region
3. Gamma correction targeting dark areas (vehicles are dark against bright sky)
4. Multi-scale detail injection at horizon

Also: what about using a COMPLETELY different inflate approach?
- Skip unsharp mask entirely, use contrast enhancement instead
- Use edge-preserving filter (bilateral) as post-processing
"""
import subprocess, sys, os, time, math, zipfile, shutil
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
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

TMP = ROOT / '_horiz_tmp'


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def decode_frames_from_mkv(mkv_path):
    import av
    from PIL import Image
    container = av.open(str(mkv_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        frames.append(f_np)
    container.close()
    return frames


def standard_unsharp(decoded, strength=0.45):
    raw = bytearray()
    for f_np in decoded:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def clahe_enhance(decoded, clip_limit=2.0, tile_size=8,
                  region=(0.30, 0.60), unsharp=0.45):
    """Apply CLAHE to the horizon band, then standard unsharp everywhere."""
    raw = bytearray()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    y_start = int(H_CAM * region[0])
    y_end = int(H_CAM * region[1])

    for f_np in decoded:
        # Apply CLAHE to horizon band
        enhanced = f_np.copy()
        lab = cv2.cvtColor(enhanced[y_start:y_end], cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced[y_start:y_end] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Then apply standard unsharp
        x = torch.from_numpy(enhanced).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def contrast_boost(decoded, alpha=1.05, beta=0, region=None, unsharp=0.45):
    """Simple contrast boost: out = alpha * pixel + beta."""
    raw = bytearray()
    for f_np in decoded:
        enhanced = f_np.copy().astype(np.float32)
        if region:
            y_s, y_e = int(H_CAM * region[0]), int(H_CAM * region[1])
            enhanced[y_s:y_e] = enhanced[y_s:y_e] * alpha + beta
        else:
            enhanced = enhanced * alpha + beta
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        x = torch.from_numpy(enhanced).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def gamma_correct(decoded, gamma=0.9, unsharp=0.45):
    """Apply gamma correction before unsharp. gamma < 1 brightens darks."""
    table = np.array([(i/255.0)**(gamma) * 255 for i in range(256)], dtype=np.uint8)
    raw = bytearray()
    for f_np in decoded:
        corrected = table[f_np]
        x = torch.from_numpy(corrected).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def bilateral_post(decoded, d=5, sigma_color=50, sigma_space=50, unsharp=0.45):
    """Bilateral filter to reduce compression artifacts while preserving edges,
    then apply unsharp mask."""
    raw = bytearray()
    for f_np in decoded:
        filtered = cv2.bilateralFilter(f_np, d, sigma_color, sigma_space)
        x = torch.from_numpy(filtered).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def fast_eval(raw_bytes, archive_size, segnet, posenet):
    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)
    seg_dists, pose_dists = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(DEVICE).float()
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
    return 100*s + math.sqrt(10*p) + 25*r, s, p, r


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()

    mkv_path = ROOT / 'submissions' / 'av1_repro' / 'archive' / '0.mkv'
    archive_zip = ROOT / 'submissions' / 'av1_repro' / 'archive.zip'
    if not archive_zip.exists():
        print("archive.zip not found, please run compress first")
        sys.exit(1)
    archive_size = archive_zip.stat().st_size
    print(f"Archive: {archive_size/1024:.1f} KB ({archive_size} bytes)", flush=True)
    assert archive_size > 800_000, f"Archive too small ({archive_size}), something is wrong"

    decoded = decode_frames_from_mkv(mkv_path)
    print(f"Decoded {len(decoded)} frames.", flush=True)

    def test(name, raw_bytes):
        score, s, p, r = fast_eval(raw_bytes, archive_size, segnet, posenet)
        print(f"[{name:35s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f}", flush=True)
        return score

    print("\n=== Baseline ===", flush=True)
    test("baseline_0.45", standard_unsharp(decoded, 0.45))

    print("\n=== CLAHE at horizon ===", flush=True)
    for clip in [1.0, 2.0, 3.0, 5.0]:
        test(f"clahe_clip{clip}", clahe_enhance(decoded, clip))

    print("\n=== Contrast boost at horizon ===", flush=True)
    for alpha in [1.03, 1.05, 1.10, 1.15]:
        test(f"contrast_horiz_{alpha}", contrast_boost(decoded, alpha, region=(0.30, 0.55)))
    for alpha in [1.03, 1.05]:
        test(f"contrast_full_{alpha}", contrast_boost(decoded, alpha))

    print("\n=== Gamma correction ===", flush=True)
    for gamma in [0.85, 0.90, 0.95, 1.05, 1.10]:
        test(f"gamma_{gamma}", gamma_correct(decoded, gamma))

    print("\n=== Bilateral + unsharp ===", flush=True)
    test("bilateral_d5_sc50", bilateral_post(decoded, 5, 50, 50))
    test("bilateral_d3_sc30", bilateral_post(decoded, 3, 30, 30))
    test("bilateral_d7_sc75", bilateral_post(decoded, 7, 75, 75))

    print("\n=== No unsharp (just upscale) ===", flush=True)
    test("no_unsharp", standard_unsharp(decoded, 0.0))

    print("\nDone.", flush=True)
