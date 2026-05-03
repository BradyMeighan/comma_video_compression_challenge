#!/usr/bin/env python
"""
Test different color conversion paths and intermediate representations.

Key observation: piped RGB path gets 2.02 vs direct MKV path gets 2.05.
The YUV→RGB→YUV round-trip adds quantization that helps models.

Hypotheses to test:
1. The specific PyAV YUV→RGB conversion matters (color matrix, rounding)
2. Different RGB→YUV mappings in ffmpeg affect quality differently
3. Going through a different intermediate (e.g., float32, specific color space)
4. Multiple round-trips might help even more (noise injection via quantization)
5. The scale factor in the piped path might differ slightly

Also test:
- Encode in YUV444 instead of YUV420 (less chroma subsampling)
- Different ffmpeg color space params (-colorspace, -color_primaries)
- Use 10-bit pipeline internally even if output is 8-bit
"""
import subprocess, sys, os, time, math, zipfile, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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

TMP = ROOT / '_cpath_tmp'


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_raw_frames():
    """Load with PyAV (same as piped path)."""
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def make_sky_mask():
    import cv2
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


def apply_sky_blur(frames_np, sigma=5.0):
    import cv2
    mask = make_sky_mask()
    alpha = mask[..., np.newaxis]
    ksize = int(sigma * 6) | 1
    return [(f * alpha + cv2.GaussianBlur(f, (ksize, ksize), sigma) * (1 - alpha)
             ).astype(np.uint8) for f in frames_np]


def encode_piped_rgb(frames, out_mkv, extra_params=None, pix_fmt='yuv420p',
                     scale=0.45, crf='33', in_pix_fmt='rgb24'):
    """Pipe raw RGB frames to ffmpeg. This is the 2.02 path."""
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    params = 'film-grain=22:keyint=180:scd=0'
    if extra_params:
        params += ':' + extra_params
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', in_pix_fmt,
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', pix_fmt,
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', crf,
        '-svtav1-params', params,
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def encode_mkv_direct(out_mkv, scale=0.45, crf='33', extra_vf=''):
    """Let ffmpeg read the MKV directly. This is the 2.05 path."""
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    vf = f'scale={w}:{h}:flags=lanczos'
    if extra_vf:
        vf = extra_vf + ',' + vf
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', str(VIDEO),
        '-vf', vf, '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', crf,
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_mkv.stat().st_size


def decode_and_eval(mkv_path, archive_size, segnet, posenet, unsharp=0.45):
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

    frames_raw = load_raw_frames()
    frames_np = [f.numpy() for f in frames_raw]
    frames_blur = apply_sky_blur(frames_np)
    print(f"Loaded {len(frames_raw)} frames.", flush=True)

    def test(name, mkv, size):
        score, s, p, r = decode_and_eval(mkv, size, segnet, posenet)
        print(f"[{name:40s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={size//1024}KB", flush=True)
        return score

    print("\n=== Color path comparison ===", flush=True)

    # Test 1: Direct MKV (no preprocessing)
    sz = encode_mkv_direct(out_mkv)
    test("mkv_direct_no_preprocess", out_mkv, sz)

    # Test 2: Piped RGB (no preprocessing)
    sz = encode_piped_rgb(frames_np, out_mkv)
    test("piped_rgb_no_preprocess", out_mkv, sz)

    # Test 3: Direct MKV with sky blur via ffmpeg filter
    # Can't easily do sky blur via ffmpeg filters. Skip.

    # Test 4: Piped RGB with sky blur
    sz = encode_piped_rgb(frames_blur, out_mkv)
    test("piped_rgb_sky_blur", out_mkv, sz)

    # Test 5: What if we do double round-trip? (PyAV decode → RGB → BGR → RGB)
    frames_drt = []
    for f in frames_np:
        # Additional round-trip: clip to uint8 again (should be no-op for uint8)
        frames_drt.append(np.clip(f.astype(np.float32) + 0.5, 0, 255).astype(np.uint8))
    frames_drt_blur = apply_sky_blur(frames_drt)
    sz = encode_piped_rgb(frames_drt_blur, out_mkv)
    test("piped_rgb_sky_blur_+0.5bias", out_mkv, sz)

    # Test 6: Subtract 0.5 bias
    frames_drt2 = []
    for f in frames_np:
        frames_drt2.append(np.clip(f.astype(np.float32) - 0.5, 0, 255).astype(np.uint8))
    frames_drt2_blur = apply_sky_blur(frames_drt2)
    sz = encode_piped_rgb(frames_drt2_blur, out_mkv)
    test("piped_rgb_sky_blur_-0.5bias", out_mkv, sz)

    # Test 7: Convert RGB to BGR before piping (mismatched color channels)
    frames_bgr = [f[:, :, ::-1].copy() for f in frames_blur]
    sz = encode_piped_rgb(frames_bgr, out_mkv, in_pix_fmt='bgr24')
    test("piped_bgr_sky_blur", out_mkv, sz)

    # Test 8: Use ffmpeg's colorspace filters on the MKV path
    for cs_filter in [
        'colorspace=bt601-6-625:bt709:bt709',
        'eq=contrast=1.01',
        'eq=brightness=0.005',
    ]:
        sz = encode_mkv_direct(out_mkv, extra_vf=cs_filter)
        test(f"mkv_vf_{cs_filter[:30]}", out_mkv, sz)

    # Test 9: Different ffmpeg Lanczos parameters
    # Test if using different resampling filter at encode time helps
    for flags in ['bicubic', 'spline', 'lanczos+accurate_rnd']:
        w = int(W_CAM * 0.45) // 2 * 2
        h = int(H_CAM * 0.45) // 2 * 2
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
            '-vf', f'scale={w}:{h}:flags={flags}', '-pix_fmt', 'yuv420p',
            '-c:v', 'libsvtav1', '-preset', '0', '-crf', '33',
            '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
            '-r', '20', str(out_mkv),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for f in frames_blur:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        proc.wait()
        sz = out_mkv.stat().st_size
        test(f"piped_sky_{flags[:25]}", out_mkv, sz)

    # Test 10: Decode with different method (ffmpeg instead of PyAV)
    print("\n=== Decode path comparison ===", flush=True)
    # Re-encode our best config
    sz = encode_piped_rgb(frames_blur, out_mkv)

    # Decode via ffmpeg to raw RGB then read
    raw_out = TMP / 'decoded.raw'
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', str(out_mkv),
        '-vf', f'scale={W_CAM}:{H_CAM}:flags=lanczos',
        '-pix_fmt', 'rgb24', '-f', 'rawvideo',
        str(raw_out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Read and apply unsharp
    raw_data = np.fromfile(raw_out, dtype=np.uint8).reshape(-1, H_CAM, W_CAM, 3)
    raw_bytes = bytearray()
    for f_np in raw_data:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + 0.45 * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())

    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw_np = np.frombuffer(bytes(raw_bytes), dtype=np.uint8).reshape(N*2, H_CAM, W_CAM, 3)
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
    r = sz / 37_545_489
    score = 100*s + math.sqrt(10*p) + 25*r
    print(f"[{'ffmpeg_decode_lanczos':40s}] score={score:.4f} seg={100*s:.4f} "
          f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
          f"size={sz//1024}KB", flush=True)

    # Clean up
    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
