#!/usr/bin/env python
"""
Decode-side upscale tricks.

Key insight: PyAV Lanczos beats FFmpeg Lanczos by 0.05 on score (all from PoseNet).
The upscale method is critical. Test more upscale approaches:

1. Oversample: upscale to 1.1x then crop/downscale
2. Torch bicubic + unsharp (since torch bicubic was close at 2.0241)
3. PyAV Lanczos at different output sizes
4. Multiple sequential upscales (e.g., 2x then to target)
5. Anti-aliased upscale
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
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size  # 1164, 874
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

TMP = ROOT / '_upscale_tmp'


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_and_encode():
    """Load, sky-blur, encode. Returns path to MKV and its archive size."""
    import av
    frames = [yuv420_to_rgb(f).numpy() for f in
              av.open(str(VIDEO)).decode(av.open(str(VIDEO)).streams.video[0])]
    
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()

    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * 0.15)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * 0.03)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t)
    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    alpha = mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    processed = [(f * alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - alpha)
                  ).astype(np.uint8) for f in frames]

    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    out_mkv = TMP / '0.mkv'
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', '33',
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    for f in processed:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    sz = out_mkv.stat().st_size
    return out_mkv, sz


def get_small_frames(mkv_path):
    """Get decoded small frames (not upscaled)."""
    import av
    container = av.open(str(mkv_path))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def unsharp(frame_t, strength=0.44):
    blur = F.conv2d(F.pad(frame_t, (4, 4, 4, 4), mode='reflect'),
                    KERNEL_9, padding=0, groups=3)
    return (frame_t + strength * (frame_t - blur)).clamp(0, 255)


def eval_frames(raw_bytes, archive_size, segnet, posenet):
    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N*2, H_CAM, W_CAM, 3)
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

    print("Encoding...", flush=True)
    mkv_path, archive_size = load_and_encode()
    print(f"  Archive: {archive_size//1024} KB", flush=True)

    small_frames = get_small_frames(mkv_path)
    sh, sw = small_frames[0].shape[:2]
    print(f"  Small frames: {sw}x{sh} ({len(small_frames)} frames)", flush=True)

    def test(name, raw_bytes):
        score, s, p, r = eval_frames(raw_bytes, archive_size, segnet, posenet)
        print(f"[{name:45s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f}", flush=True)
        return score

    # 1. BASELINE: PyAV Lanczos + 0.44 unsharp (PIL upscale)
    print("\n=== Baseline ===", flush=True)
    from PIL import Image
    raw = bytearray()
    for f_np in small_frames:
        pil = Image.fromarray(f_np)
        pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
        up = np.array(pil)
        x = torch.from_numpy(up).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        x = unsharp(x, 0.44)
        t = x.squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    test("pil_lanczos_ush0.44", bytes(raw))

    # 2. Torch bicubic + unsharp
    print("\n=== Torch upscale variants ===", flush=True)
    for mode in ['bicubic', 'bilinear']:
        for ush in [0.0, 0.30, 0.44, 0.55]:
            raw = bytearray()
            for f_np in small_frames:
                x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                x = F.interpolate(x, size=(H_CAM, W_CAM), mode=mode,
                                  align_corners=False)
                x = unsharp(x, ush)
                t = x.squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
                raw.extend(t.contiguous().numpy().tobytes())
            test(f"torch_{mode}_ush{ush}", bytes(raw))

    # 3. Oversample: upscale to 1.1x then lanczos downscale to target
    print("\n=== Oversample upscale ===", flush=True)
    for over in [1.05, 1.10, 1.15, 1.20]:
        oh = int(H_CAM * over) // 2 * 2
        ow = int(W_CAM * over) // 2 * 2
        raw = bytearray()
        for f_np in small_frames:
            pil = Image.fromarray(f_np)
            pil_big = pil.resize((ow, oh), Image.LANCZOS)
            pil_final = pil_big.resize((W_CAM, H_CAM), Image.LANCZOS)
            up = np.array(pil_final)
            x = torch.from_numpy(up).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            x = unsharp(x, 0.44)
            t = x.squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            raw.extend(t.contiguous().numpy().tobytes())
        test(f"oversample_{over}x_lanczos", bytes(raw))

    # 4. Two-step upscale: 2x first, then to target
    print("\n=== Two-step upscale ===", flush=True)
    raw = bytearray()
    for f_np in small_frames:
        pil = Image.fromarray(f_np)
        pil_2x = pil.resize((sw * 2, sh * 2), Image.LANCZOS)
        pil_final = pil_2x.resize((W_CAM, H_CAM), Image.LANCZOS)
        up = np.array(pil_final)
        x = torch.from_numpy(up).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        x = unsharp(x, 0.44)
        t = x.squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    test("two_step_2x_lanczos", bytes(raw))

    # 5. OpenCV upscale methods
    print("\n=== OpenCV upscale ===", flush=True)
    for interp, name in [(cv2.INTER_CUBIC, 'cv_cubic'),
                          (cv2.INTER_LANCZOS4, 'cv_lanczos4'),
                          (cv2.INTER_AREA, 'cv_area')]:
        raw = bytearray()
        for f_np in small_frames:
            up = cv2.resize(f_np, (W_CAM, H_CAM), interpolation=interp)
            x = torch.from_numpy(up).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            x = unsharp(x, 0.44)
            t = x.squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            raw.extend(t.contiguous().numpy().tobytes())
        test(f"{name}_ush0.44", bytes(raw))

    # 6. PIL different resamplers
    print("\n=== PIL resampler variants ===", flush=True)
    for method, mname in [(Image.BICUBIC, 'pil_bicubic'),
                           (Image.BILINEAR, 'pil_bilinear'),
                           (Image.NEAREST, 'pil_nearest'),
                           (Image.HAMMING, 'pil_hamming'),
                           (Image.BOX, 'pil_box')]:
        raw = bytearray()
        for f_np in small_frames:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), method)
            up = np.array(pil)
            x = torch.from_numpy(up).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            x = unsharp(x, 0.44)
            t = x.squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            raw.extend(t.contiguous().numpy().tobytes())
        test(f"{mname}_ush0.44", bytes(raw))

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
