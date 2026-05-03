#!/usr/bin/env python
"""
Per-segment color LUT optimization.

Idea: Store a small 3D color LUT in the archive that maps decoded pixel colors
to values that better match the original frame's model-relevant characteristics.

The LUT is computed at encode time by analyzing how the encode→decode pipeline
distorts colors, then creating a correction LUT that inverts this distortion.

Also test: per-frame global color shift (just 3 floats per frame = 3600 bytes total).
"""
import subprocess, sys, os, time, math, zipfile, shutil, struct, bz2
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

TMP = ROOT / '_lut_tmp'


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


def encode_piped(frames, out_mkv, scale=0.45, crf='33'):
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
    ksize = 31

    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
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
        blurred = cv2.GaussianBlur(f, (ksize, ksize), 5.0)
        processed = (f * alpha + blurred * (1 - alpha)).astype(np.uint8)
        proc.stdin.write(processed.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def decode_and_upscale(mkv_path, unsharp_strength=0.44):
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

    result = bytearray()
    for f_np in frames:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp_strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        result.extend(t.contiguous().numpy().tobytes())
    return bytes(result), frames


def eval_raw(raw_bytes, archive_size, segnet, posenet):
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
    return score, s, p, r, np.array(seg_dists), np.array(pose_dists)


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    TMP.mkdir(exist_ok=True)

    orig_frames = load_frames()
    print(f"Loaded {len(orig_frames)} original frames.", flush=True)

    out_mkv = TMP / '0.mkv'
    mkv_size = encode_piped(orig_frames, out_mkv)
    print(f"Encoded: {mkv_size//1024} KB", flush=True)

    # Get decoded frames (before unsharp) and final frames (after unsharp)
    raw_bytes, decoded_frames = decode_and_upscale(out_mkv, unsharp_strength=0.44)
    score, s, p, r, sd, pd = eval_raw(raw_bytes, mkv_size, segnet, posenet)
    print(f"\nBaseline: score={score:.4f} seg={100*s:.4f} "
          f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f}\n", flush=True)

    # Analyze color distribution difference between original and decoded
    print("=== Color distribution analysis ===", flush=True)
    per_frame_shift = np.zeros((len(orig_frames), 3), dtype=np.float64)
    per_frame_scale = np.zeros((len(orig_frames), 3), dtype=np.float64)
    
    for i in range(len(orig_frames)):
        orig = orig_frames[i].astype(np.float64)
        dec = decoded_frames[i].astype(np.float64)
        for c in range(3):
            o_mean = orig[:, :, c].mean()
            d_mean = dec[:, :, c].mean()
            per_frame_shift[i, c] = o_mean - d_mean
            o_std = orig[:, :, c].std()
            d_std = dec[:, :, c].std()
            per_frame_scale[i, c] = o_std / max(d_std, 1e-6)

    print(f"  Mean shift (orig - decoded): R={per_frame_shift[:,0].mean():.3f} "
          f"G={per_frame_shift[:,1].mean():.3f} B={per_frame_shift[:,2].mean():.3f}")
    print(f"  Mean scale: R={per_frame_scale[:,0].mean():.4f} "
          f"G={per_frame_scale[:,1].mean():.4f} B={per_frame_scale[:,2].mean():.4f}")
    print(f"  Shift std: R={per_frame_shift[:,0].std():.3f} "
          f"G={per_frame_shift[:,1].std():.3f} B={per_frame_shift[:,2].std():.3f}")

    # Test: apply global mean color shift at decode time
    print("\n=== Global color shift correction ===", flush=True)
    mean_shift = per_frame_shift.mean(axis=0)
    print(f"  Applying mean shift: {mean_shift}", flush=True)

    raw_shifted = bytearray()
    for f_np in decoded_frames:
        corrected = np.clip(f_np.astype(np.float32) + mean_shift[np.newaxis, np.newaxis, :],
                            0, 255).astype(np.uint8)
        x = torch.from_numpy(corrected).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + 0.44 * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_shifted.extend(t.contiguous().numpy().tobytes())
    
    score2, s2, p2, r2, _, _ = eval_raw(bytes(raw_shifted), mkv_size, segnet, posenet)
    print(f"  Global shift: score={score2:.4f} seg={100*s2:.4f} "
          f"pose={math.sqrt(10*p2):.4f}", flush=True)

    # Test: per-frame color shift (costs 1200*3*2 = 7.2KB if stored as int16)
    print("\n=== Per-frame color shift ===", flush=True)
    raw_pf = bytearray()
    for i, f_np in enumerate(decoded_frames):
        shift = per_frame_shift[i]
        corrected = np.clip(f_np.astype(np.float32) + shift[np.newaxis, np.newaxis, :],
                            0, 255).astype(np.uint8)
        x = torch.from_numpy(corrected).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + 0.44 * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_pf.extend(t.contiguous().numpy().tobytes())

    # Calculate cost of storing shifts
    shifts_q = np.round(per_frame_shift * 10).astype(np.int8)  # 0.1 precision
    shifts_compressed = bz2.compress(shifts_q.tobytes())
    extra_size = len(shifts_compressed)
    total_size = mkv_size + extra_size
    score3, s3, p3, r3, _, _ = eval_raw(bytes(raw_pf), total_size, segnet, posenet)
    print(f"  Per-frame shift: score={score3:.4f} seg={100*s3:.4f} "
          f"pose={math.sqrt(10*p3):.4f} extra={extra_size}B "
          f"total={total_size//1024}KB", flush=True)

    # Test: per-frame color scale+shift (mean+std matching)
    print("\n=== Per-frame color normalize (mean+std match) ===", flush=True)
    raw_norm = bytearray()
    for i, f_np in enumerate(decoded_frames):
        corrected = f_np.astype(np.float32)
        for c in range(3):
            corrected[:, :, c] = (corrected[:, :, c] - corrected[:, :, c].mean()
                                  ) * per_frame_scale[i, c] + (
                                  decoded_frames[i][:, :, c].mean() + per_frame_shift[i, c])
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        x = torch.from_numpy(corrected).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + 0.44 * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_norm.extend(t.contiguous().numpy().tobytes())
    
    scale_shift_data = np.column_stack([per_frame_shift, per_frame_scale]).astype(np.float32)
    extra2 = len(bz2.compress(scale_shift_data.tobytes()))
    total2 = mkv_size + extra2
    score4, s4, p4, r4, _, _ = eval_raw(bytes(raw_norm), total2, segnet, posenet)
    print(f"  Per-frame normalize: score={score4:.4f} seg={100*s4:.4f} "
          f"pose={math.sqrt(10*p4):.4f} extra={extra2}B", flush=True)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
