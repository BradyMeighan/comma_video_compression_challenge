#!/usr/bin/env python
"""
Smooth tone curve encode/decode to preserve dark region detail.

Key finding: gamma 2.0 gives seg=0.414 and pose=0.748 (better than leader!)
but file size explodes. We need the quality without the rate cost.

Approaches:
1. Log tone curve (smoothly boosts darks, less extreme than gamma)
2. Piecewise linear: stretch dark region, compress bright region
3. Sigmoid/S-curve for balanced contrast
4. SVT-AV1 variance-boost for dark regions
5. Encode dark and bright separately with different CRF
"""
import subprocess, sys, time, math, shutil
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

TMP = ROOT / '_tone_tmp'


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


def apply_sky_blur(frames, sigma=5.0):
    mask = make_sky_mask()
    alpha = mask[..., np.newaxis]
    ksize = int(sigma * 6) | 1
    return [(f * alpha + cv2.GaussianBlur(f, (ksize, ksize), sigma) * (1 - alpha)
             ).astype(np.uint8) for f in frames]


def make_lut_pair(forward_fn):
    """Create encode and decode LUTs from a monotonic forward function.
    forward_fn maps [0,255] → [0,255] monotonically."""
    fwd = np.array([forward_fn(i) for i in range(256)], dtype=np.float64)
    fwd = np.clip(fwd, 0, 255)
    fwd_uint8 = np.round(fwd).astype(np.uint8)

    # Build inverse LUT
    inv = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        # Find the input value that maps closest to i
        diffs = np.abs(fwd - i)
        inv[i] = np.argmin(diffs)
    return fwd_uint8, inv


def encode(frames, out_mkv, crf='33', extra_params=''):
    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    params = 'film-grain=22:keyint=180:scd=0'
    if extra_params:
        params += ':' + extra_params
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', crf,
        '-svtav1-params', params,
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def full_pipeline(frames_to_encode, archive_size, inv_lut, segnet, posenet, unsharp=0.44):
    """Decode, apply inverse LUT, upscale, unsharp, evaluate."""
    import av
    from PIL import Image
    out_mkv = TMP / '0.mkv'
    container = av.open(str(out_mkv))
    decoded = []
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        if inv_lut is not None:
            f_np = inv_lut[f_np]
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
    blurred = apply_sky_blur(orig_frames)
    print(f"Loaded {len(orig_frames)} frames.", flush=True)

    def test(name, frames_in, crf='33', inv_lut=None, extra_params=''):
        sz = encode(frames_in, out_mkv, crf=crf, extra_params=extra_params)
        score, s, p, r = full_pipeline(frames_in, sz, inv_lut, segnet, posenet)
        print(f"[{name:45s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={sz//1024}KB", flush=True)
        return score

    # Baseline
    print("\n=== Baseline ===", flush=True)
    test("baseline", blurred)

    # Log tone curves: out = a * ln(1 + b*in) where b controls dark boost
    print("\n=== Log tone curves ===", flush=True)
    for b_val in [1.0, 2.0, 5.0, 10.0, 20.0]:
        def fwd(x, b=b_val):
            a = 255.0 / np.log(1 + b * 255)
            return a * np.log(1 + b * x)
        fwd_lut, inv_lut = make_lut_pair(fwd)
        encoded = [fwd_lut[f] for f in blurred]
        test(f"log_b{b_val}", encoded, inv_lut=inv_lut)

    # Piecewise linear: stretch [0, knee] to [0, stretch_to], compress [knee, 255] to [stretch_to, 255]
    print("\n=== Piecewise linear stretch ===", flush=True)
    for knee, stretch in [(30, 80), (50, 120), (80, 160), (30, 60), (50, 100)]:
        def fwd(x, k=knee, s=stretch):
            if x <= k:
                return x * s / k
            else:
                return s + (x - k) * (255 - s) / (255 - k)
        fwd_lut, inv_lut = make_lut_pair(fwd)
        encoded = [fwd_lut[f] for f in blurred]
        test(f"plinear_k{knee}_s{stretch}", encoded, inv_lut=inv_lut)

    # SVT-AV1 variance-boost (internal parameter for dark region handling)
    print("\n=== SVT-AV1 variance boost ===", flush=True)
    for vbs in [1, 2, 3, 4]:
        test(f"varboost_{vbs}", blurred, extra_params=f'variance-boost-strength={vbs}')

    # Test: what about lowering film-grain to save bits, then using
    # the saved bits for lower CRF?
    print("\n=== Film-grain vs CRF tradeoff ===", flush=True)
    for fg, crf in [(18, '32'), (15, '31'), (12, '30'), (22, '33'),
                     (22, '32'), (22, '31')]:
        params = f'film-grain={fg}:keyint=180:scd=0'
        sz = encode(blurred, out_mkv, crf=crf, extra_params=f'film-grain={fg}')
        score, s, p, r = full_pipeline(blurred, sz, None, segnet, posenet)
        nm = f"fg{fg}_crf{crf}"
        print(f"[{nm:45s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={sz//1024}KB", flush=True)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
