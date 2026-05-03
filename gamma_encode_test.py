#!/usr/bin/env python
"""
Gamma encode/decode trick to preserve dark region detail.

Key insight: 99.9% of pose error is in tx (lateral translation), and
worst pairs are in dark frames (brightness 18-23/255). The codec quantizes
dark pixels too aggressively, destroying the subtle lateral features PoseNet needs.

Approach: Apply gamma compression BEFORE encoding (boosts dark pixels above noise floor),
then apply inverse gamma at decode (restores original brightness).

This gives the codec more "room" in the dark regions — instead of trying to encode
pixel values 0-30 with only a few quantization levels, we stretch them to 0-120+
which gives the codec much more precision.

The gamma encode/decode is purely mathematical (no model needed at decode time).
The decode-side just needs to know the gamma value.
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
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

TMP = ROOT / '_gamma_enc_tmp'


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


def gamma_encode(frames, gamma):
    """Apply gamma compression: out = (in/255)^(1/gamma) * 255"""
    lut = np.array([(i / 255.0) ** (1.0 / gamma) * 255.0
                     for i in range(256)], dtype=np.uint8)
    return [lut[f] for f in frames]


def gamma_decode(frames_uint8, gamma):
    """Apply gamma expansion: out = (in/255)^gamma * 255"""
    lut = np.array([(i / 255.0) ** gamma * 255.0
                     for i in range(256)], dtype=np.uint8)
    return [lut[f] for f in frames_uint8]


def encode(frames, out_mkv, crf='33'):
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


def decode_and_inflate(mkv_path, gamma=None, unsharp=0.44):
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

    # Apply inverse gamma if encode used gamma
    if gamma is not None:
        decoded = gamma_decode(decoded, gamma)

    raw = bytearray()
    for f_np in decoded:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


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
    return score, s, p, r


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    TMP.mkdir(exist_ok=True)
    out_mkv = TMP / '0.mkv'

    orig_frames = load_frames()
    print(f"Loaded {len(orig_frames)} frames.", flush=True)

    # Apply sky blur
    blurred = apply_sky_blur(orig_frames)

    def test(name, frames_to_encode, crf='33', gamma=None, unsharp=0.44):
        sz = encode(frames_to_encode, out_mkv, crf=crf)
        raw = decode_and_inflate(out_mkv, gamma=gamma, unsharp=unsharp)
        score, s, p, r = eval_raw(raw, sz, segnet, posenet)
        print(f"[{name:45s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={sz//1024}KB", flush=True)
        return score

    # 1. Baseline
    print("\n=== Baseline ===", flush=True)
    test("baseline_sky_blur", blurred)

    # 2. Gamma encode/decode with sky blur
    print("\n=== Gamma encode/decode ===", flush=True)
    for gamma in [0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
        gamma_frames = gamma_encode(blurred, gamma)
        test(f"gamma_{gamma}_sky_blur", gamma_frames, gamma=gamma)

    # 3. Gamma encode with different CRF (since gamma changes the rate)
    print("\n=== Gamma with CRF adjustment ===", flush=True)
    for gamma, crf in [(0.7, '35'), (0.7, '37'), (0.8, '34'), (0.8, '35'),
                        (1.5, '31'), (1.5, '30'), (2.0, '30'), (2.0, '28')]:
        gamma_frames = gamma_encode(blurred, gamma)
        test(f"gamma_{gamma}_crf{crf}", gamma_frames, crf=crf, gamma=gamma)

    # 4. Partial gamma: only boost dark pixels (below threshold)
    print("\n=== Partial gamma (dark regions only) ===", flush=True)
    for threshold, gamma in [(50, 0.5), (50, 0.7), (80, 0.5), (80, 0.7), (30, 0.5)]:
        partial_frames = []
        for f in blurred:
            out = f.copy().astype(np.float32)
            dark_mask = f < threshold
            out[dark_mask] = (out[dark_mask] / 255.0) ** (1.0/gamma) * 255.0
            partial_frames.append(np.clip(out, 0, 255).astype(np.uint8))
        
        # At decode, reverse: dark pixels get gamma applied
        # But we need to know which pixels WERE dark. Without storing the mask,
        # we use the decoded values and threshold them (imprecise but workable)
        sz = encode(partial_frames, out_mkv)
        
        # For decode, we need the inverse but we'll be approximate
        import av as _av
        from PIL import Image as _Image
        container = _av.open(str(out_mkv))
        dec_frames = []
        for frame in container.decode(container.streams.video[0]):
            f_np = yuv420_to_rgb(frame).numpy()
            h, w, _ = f_np.shape
            if h != H_CAM or w != W_CAM:
                pil = _Image.fromarray(f_np)
                pil = pil.resize((W_CAM, H_CAM), _Image.LANCZOS)
                f_np = np.array(pil)
            # Inverse: pixels that decode above threshold^(1/gamma)*255 were originally dark
            # This is the gamma-compressed threshold
            compressed_thresh = int((threshold / 255.0) ** (1.0/gamma) * 255.0)
            out = f_np.astype(np.float32)
            bright_mask = f_np < compressed_thresh
            # These were originally dark — apply inverse
            out[bright_mask] = (out[bright_mask] / 255.0) ** gamma * 255.0
            dec_frames.append(np.clip(out, 0, 255).astype(np.uint8))
        container.close()

        raw = bytearray()
        for f_np in dec_frames:
            x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                            KERNEL_9, padding=0, groups=3)
            x = x + 0.44 * (x - blur)
            t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            raw.extend(t.contiguous().numpy().tobytes())

        score, s, p, r = eval_raw(bytes(raw), sz, segnet, posenet)
        nm = f"partial_t{threshold}_g{gamma}"
        print(f"[{nm:45s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f}", flush=True)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
