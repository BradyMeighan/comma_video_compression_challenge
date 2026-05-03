#!/usr/bin/env python
"""
Adaptive unsharp mask guided by the spatial error analysis.

Key insight from analysis:
- Seg errors concentrate at the HORIZON LINE (center-vertical of frame)
- Class 1 (vehicles, 0.6% of pixels) has 5.66% error rate
- Errors are at thin class boundaries, not region interiors
- Bottom 1/3 (hood) and top 1/3 (sky interior) have almost zero errors

Approach:
1. Spatially-varying unsharp: stronger at horizon, normal elsewhere
2. Different kernel sizes: try larger kernels to capture boundary structure
3. Two-pass unsharp: first pass normal, second pass stronger at boundary region
4. Frequency-domain: preserve more high frequencies at horizon
5. Per-channel unsharp: different strength per RGB channel

Also test: the inflated frames currently use the SAME unsharp for both
frames in a pair. But SegNet only uses the LAST frame. What if we apply
STRONGER unsharp to odd frames (last in pair) and WEAKER to even frames?
PoseNet uses both but is less sensitive to individual frame quality.
"""
import subprocess, sys, os, time, math, zipfile, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file
import einops

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

_r5 = torch.tensor([1., 4., 6., 4., 1.])
KERNEL_5 = (torch.outer(_r5, _r5) / (_r5.sum()**2)).to(DEVICE).expand(3, 1, 5, 5)

TMP = ROOT / '_adap_tmp'


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_gt_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def make_sky_mask():
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * 0.15)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * 0.03)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t)
    return cv2.GaussianBlur(mask, (31, 31), 10)


def encode_sky_blur(frames_np, out_mkv):
    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    processed = [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - sky_alpha)
                  ).astype(np.uint8) for f in frames_np]

    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', '33',
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in processed:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def decode_frames(mkv_path):
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


def apply_unsharp(decoded_frames, strength=0.45, kernel=KERNEL_9, pad=4):
    raw = bytearray()
    for f_np in decoded_frames:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                        kernel, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def apply_adaptive_unsharp(decoded_frames, base_strength=0.45,
                            boost_strength=0.70, boost_region=(0.35, 0.55)):
    """Apply stronger unsharp at the horizon region (where seg errors concentrate)."""
    raw = bytearray()

    # Vertical strength profile
    strength_map = np.ones(H_CAM, dtype=np.float32) * base_strength
    y_start = int(H_CAM * boost_region[0])
    y_end = int(H_CAM * boost_region[1])

    for y in range(y_start, y_end):
        frac = (y - y_start) / max(y_end - y_start, 1)
        peak = 1.0 - 4.0 * (frac - 0.5)**2  # parabolic peak at center
        strength_map[y] = base_strength + (boost_strength - base_strength) * peak

    # Expand to 2D mask
    strength_2d = strength_map[:, np.newaxis, np.newaxis]  # H, 1, 1 for broadcasting

    for f_np in decoded_frames:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        detail = x - blur  # high-freq detail

        # Convert strength map to tensor
        s = torch.from_numpy(strength_map).float().to(DEVICE)
        s = s.view(1, 1, H_CAM, 1)  # 1, 1, H, 1

        result = x + s * detail
        t = result.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())

    return bytes(raw)


def apply_pair_asymmetric_unsharp(decoded_frames, even_strength=0.35,
                                   odd_strength=0.55):
    """Different unsharp for even (frame_0) and odd (frame_1) frames.

    SegNet only evaluates the odd (last) frame, so sharpen it more.
    PoseNet uses both but with sqrt dampening.
    """
    raw = bytearray()
    for i, f_np in enumerate(decoded_frames):
        strength = odd_strength if i % 2 == 1 else even_strength
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def apply_dual_kernel_unsharp(decoded_frames, coarse_strength=0.3,
                               fine_strength=0.2):
    """Two-pass: coarse (9-tap) + fine (5-tap) unsharp for multi-scale edge recovery."""
    raw = bytearray()
    for f_np in decoded_frames:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur9 = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                         KERNEL_9, padding=0, groups=3)
        x = x + coarse_strength * (x - blur9)
        blur5 = F.conv2d(F.pad(x, (2, 2, 2, 2), mode='reflect'),
                         KERNEL_5, padding=0, groups=3)
        x = x + fine_strength * (x - blur5)
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
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} frames.", flush=True)

    TMP.mkdir(exist_ok=True)
    out_mkv = TMP / '0.mkv'
    archive_zip = TMP / 'archive.zip'

    # Encode once (same video for all tests)
    print("Encoding with sky blur (one-time)...", flush=True)
    frames_np = [f.numpy() for f in gt_frames]
    encode_sky_blur(frames_np, out_mkv)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size
    print(f"  Archive: {archive_size/1024:.1f} KB", flush=True)

    # Decode once
    print("Decoding...", flush=True)
    decoded = decode_frames(out_mkv)
    print(f"  {len(decoded)} frames decoded.", flush=True)

    def test(name, raw_bytes):
        score, s, p, r = fast_eval(raw_bytes, archive_size, segnet, posenet)
        print(f"[{name:30s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f}", flush=True)
        return score

    # 1. BASELINE
    print("\n=== Tests ===", flush=True)
    test("baseline_0.45", apply_unsharp(decoded, 0.45))

    # 2. Different uniform strengths
    for s in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80]:
        test(f"uniform_{s}", apply_unsharp(decoded, s))

    # 3. Adaptive unsharp (boost at horizon)
    for boost in [0.60, 0.70, 0.80, 0.90]:
        test(f"adaptive_base0.45_boost{boost}", 
             apply_adaptive_unsharp(decoded, 0.45, boost))

    # 4. Pair-asymmetric unsharp
    configs = [(0.30, 0.60), (0.35, 0.55), (0.35, 0.65), (0.40, 0.55),
               (0.30, 0.70), (0.20, 0.70)]
    for even_s, odd_s in configs:
        test(f"asym_{even_s}_{odd_s}",
             apply_pair_asymmetric_unsharp(decoded, even_s, odd_s))

    # 5. Dual kernel
    for cs, fs in [(0.30, 0.15), (0.30, 0.20), (0.35, 0.15), (0.25, 0.25)]:
        test(f"dual_{cs}+{fs}", apply_dual_kernel_unsharp(decoded, cs, fs))

    # 6. 5-tap kernel instead of 9-tap
    test("5tap_0.45", apply_unsharp(decoded, 0.45, KERNEL_5, pad=2))
    test("5tap_0.55", apply_unsharp(decoded, 0.55, KERNEL_5, pad=2))
    test("5tap_0.65", apply_unsharp(decoded, 0.65, KERNEL_5, pad=2))

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
