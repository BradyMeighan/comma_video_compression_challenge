#!/usr/bin/env python
"""
ROI v4: Carefully designed driving corridor that matches what leader likely uses.
Key insight: PoseNet needs perspective lines/vanishing point (center of frame),
SegNet needs class boundaries. Only blur truly unimportant areas (sky, far margins).

Also tests: using the actual 4 polygon masks over different frame ranges,
and testing different approaches to the problem:
- Temporal averaging (non-blurring denoise)
- Very gentle bilateral filtering
"""

import os, sys, time, math, subprocess, zipfile
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import einops
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
INPUT_VIDEO = VIDEOS_DIR / "0.mkv"
ORIGINAL_SIZE = 37_545_489
W_CAM, H_CAM = 1164, 874
NUM_FRAMES = 1200


class FastEval:
    def __init__(self, device='cuda'):
        from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
        from safetensors.torch import load_file
        self.device = torch.device(device)
        self.segnet = SegNet().eval().to(self.device)
        self.segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(self.device)))
        self.posenet = PoseNet().eval().to(self.device)
        self.posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(self.device)))
        cache = ROOT / "_eval_cache" / "gt.pt"
        gt = torch.load(cache, weights_only=True)
        self.gt_seg, self.gt_pose = gt['seg'], gt['pose']
        print(f"Eval ready. {self.gt_seg.shape[0]} samples.", flush=True)

    def evaluate_raw(self, raw_path, archive_bytes):
        N = self.gt_seg.shape[0]
        raw = np.fromfile(raw_path, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)
        sd, pd = [], []
        with torch.inference_mode():
            for i in range(0, N, 16):
                end = min(i + 16, N)
                f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(self.device).float()
                f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(self.device).float()
                x = einops.rearrange(torch.stack([f0, f1], dim=1), 'b t h w c -> b t c h w')
                sp = self.segnet(self.segnet.preprocess_input(x)).argmax(1)
                sd.extend((sp != self.gt_seg[i:end].to(self.device)).float().mean((1,2)).cpu().tolist())
                pp = self.posenet(self.posenet.preprocess_input(x))['pose'][:, :6]
                pd.extend((pp - self.gt_pose[i:end].to(self.device)).pow(2).mean(1).cpu().tolist())
        seg_d, pose_d = np.mean(sd), np.mean(pd)
        rate = archive_bytes / ORIGINAL_SIZE
        return 100*seg_d + math.sqrt(10*pose_d) + 25*rate, 100*seg_d, math.sqrt(10*pose_d), 25*rate


def load_frames():
    import av
    from frame_utils import yuv420_to_rgb
    container = av.open(str(INPUT_VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def make_wide_corridor_mask(margin_top_frac=0.15, margin_side_frac=0.05):
    """
    Very conservative mask: only blur the extreme top (sky) and thin side margins.
    The entire driving scene including all road, cars, buildings stays sharp.
    """
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)

    # Top region: sky - gradient from 0 to 1
    sky_end = int(H_CAM * margin_top_frac)
    for y in range(sky_end):
        t = y / sky_end  # 0 at top, 1 at boundary
        mask[y, :] = t ** 0.5  # smooth ramp

    # Side margins: very thin
    side_px = int(W_CAM * margin_side_frac)
    for x in range(side_px):
        t = x / side_px
        mask[:, x] = np.minimum(mask[:, x], t ** 0.5)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t ** 0.5)

    mask = cv2.GaussianBlur(mask, (31, 31), 10)
    return mask


def make_driving_corridor_v2():
    """
    Improved driving corridor: wide trapezoid from vanishing point to bottom.
    Uses 4 different polygons for different frame ranges (like the leader).
    Smooth feathered edges.
    """
    masks = []

    # Vanishing point is roughly at (582, 290) for this dashcam
    vp_x, vp_y = W_CAM // 2, int(H_CAM * 0.33)

    # Create generous corridor polygon
    pts = np.array([
        [0, H_CAM],                      # bottom-left
        [0, int(H_CAM * 0.6)],           # left edge mid
        [int(W_CAM * 0.15), vp_y],       # upper-left
        [int(W_CAM * 0.85), vp_y],       # upper-right
        [W_CAM, int(H_CAM * 0.6)],      # right edge mid
        [W_CAM, H_CAM],                  # bottom-right
    ], dtype=np.int32)

    base_mask = np.zeros((H_CAM, W_CAM), dtype=np.uint8)
    cv2.fillPoly(base_mask, [pts], 255)

    # Feather the edges with a large Gaussian blur
    feathered = cv2.GaussianBlur(base_mask.astype(np.float32) / 255.0, (61, 61), 20)
    feathered = np.clip(feathered, 0, 1)

    # Minimum importance even outside corridor (don't black out)
    feathered = np.maximum(feathered, 0.15)

    return feathered


def encode_with_preprocess(frames, mask, mkv_path,
                           blur_sigma=3.0, bilateral=False,
                           crf=33, fg=22, scale=0.45, keyint=180, preset=0):
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    ksize = int(blur_sigma * 6) | 1

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', f'film-grain={fg}:keyint={keyint}:scd=0',
        '-r', '20', str(mkv_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    alpha = mask[..., np.newaxis] if mask is not None else None

    for frame in frames:
        if mask is not None and blur_sigma > 0:
            if bilateral:
                blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=50, sigmaSpace=50)
            else:
                blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_sigma)
            result = (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)
            proc.stdin.write(result.tobytes())
        else:
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()
    return proc.returncode == 0


def inflate_mkv(mkv_path, raw_path, unsharp=0.40, kernel_taps=9):
    import av
    from frame_utils import yuv420_to_rgb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    kernel = (torch.outer(_r, _r) / (_r.sum()**2)).to(device).expand(3, 1, 9, 9)

    container = av.open(str(mkv_path))
    n = 0
    with open(str(raw_path), 'wb') as f:
        for frame in container.decode(container.streams.video[0]):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != H_CAM or W != W_CAM:
                pil = Image.fromarray(t.numpy()).resize((W_CAM, H_CAM), Image.LANCZOS)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(device)
                if unsharp > 0:
                    blur = F.conv2d(F.pad(x, (4,4,4,4), mode='reflect'), kernel, padding=0, groups=3)
                    x = x + unsharp * (x - blur)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


def run(evaluator, frames, label, mask=None, blur_sigma=3.0, bilateral=False,
        crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40):
    work = ROOT / "_roi_work"
    work.mkdir(parents=True, exist_ok=True)
    mkv = work / "0.mkv"
    zp = work / "archive.zip"
    raw = work / "0.raw"

    t0 = time.time()
    ok = encode_with_preprocess(frames, mask, mkv, blur_sigma, bilateral,
                                crf, fg, scale, keyint)
    if not ok:
        print(f"  {label}: FAILED", flush=True)
        return None
    t_enc = time.time() - t0

    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
        z.write(mkv, '0.mkv')
    arch = zp.stat().st_size

    inflate_mkv(mkv, raw, unsharp)
    t_inf = time.time() - t0 - t_enc

    score, seg, pose, rate = evaluator.evaluate_raw(str(raw), arch)
    elapsed = time.time() - t0

    for p in [mkv, zp, raw]:
        p.unlink(missing_ok=True)

    print(f"  {label}: score={score:.4f} seg={seg:.4f} pose={pose:.4f} "
          f"rate={rate:.4f} sz={arch/1024:.0f}KB ({elapsed:.0f}s)", flush=True)
    return (label, score, seg, pose, rate, arch)


def temporal_denoise(frames, window=3):
    """Average nearby frames for static regions (reduces noise without spatial blur)."""
    result = []
    n = len(frames)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)
        stack = np.stack(frames[start:end]).astype(np.float32)
        avg = stack.mean(axis=0)

        diff = np.abs(frames[i].astype(np.float32) - avg)
        motion_mask = (diff.mean(axis=2) > 15).astype(np.float32)
        motion_mask = cv2.dilate(motion_mask, np.ones((5,5)))
        motion_mask = cv2.GaussianBlur(motion_mask, (11, 11), 3)[..., np.newaxis]

        blended = frames[i] * motion_mask + avg * (1 - motion_mask)
        result.append(blended.astype(np.uint8))
    return result


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = FastEval(device)

    print("\nLoading frames...", flush=True)
    frames = load_frames()
    print(f"Loaded {len(frames)} frames", flush=True)

    results = []

    # Baseline reproduction with leader's exact params
    print("\n" + "="*70, flush=True)
    print("BASELINE: Leader's exact params (CRF=33, fg=22, s=0.45, ush=0.40)", flush=True)
    print("="*70, flush=True)
    r = run(ev, frames, "baseline_leader_params",
            crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40)
    if r: results.append(r)

    # Our best sweep config
    r = run(ev, frames, "best_sweep",
            crf=35, fg=22, scale=0.45, keyint=600, unsharp=0.50)
    if r: results.append(r)

    # Corridor ROI tests - very conservative
    print("\n" + "="*70, flush=True)
    print("CORRIDOR ROI: Very conservative (sky + margins only)", flush=True)
    print("="*70, flush=True)

    for top_frac in [0.10, 0.15, 0.20, 0.25, 0.30]:
        mask = make_wide_corridor_mask(margin_top_frac=top_frac, margin_side_frac=0.03)
        for sigma in [2.0, 3.0, 5.0]:
            label = f"sky{top_frac}_s{sigma}"
            r = run(ev, frames, label, mask=mask, blur_sigma=sigma,
                    crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40)
            if r: results.append(r)

    # Full driving corridor ROI
    print("\n" + "="*70, flush=True)
    print("FULL CORRIDOR ROI", flush=True)
    print("="*70, flush=True)
    corridor = make_driving_corridor_v2()
    for sigma in [2.0, 3.0, 5.0, 8.0]:
        r = run(ev, frames, f"corridor_s{sigma}", mask=corridor, blur_sigma=sigma,
                crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40)
        if r: results.append(r)

    # Bilateral filter corridor
    r = run(ev, frames, "corridor_bilateral", mask=corridor, blur_sigma=3.0,
            bilateral=True, crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40)
    if r: results.append(r)

    # Temporal denoise approach
    print("\n" + "="*70, flush=True)
    print("TEMPORAL DENOISE (no spatial blur)", flush=True)
    print("="*70, flush=True)
    for win in [3, 5]:
        print(f"  Computing temporal denoise (window={win})...", flush=True)
        denoised = temporal_denoise(frames, window=win)
        r = run(ev, denoised, f"temporal_w{win}",
                crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40)
        if r: results.append(r)
        del denoised

    # Best corridor + optimized params
    print("\n" + "="*70, flush=True)
    print("COMBINED: Corridor + param optimization", flush=True)
    print("="*70, flush=True)
    if results:
        roi_results = [r for r in results if r[0].startswith('corridor_') or r[0].startswith('sky')]
        if roi_results:
            best_roi = min(roi_results, key=lambda x: x[1])
            print(f"  Best ROI: {best_roi[0]} = {best_roi[1]:.4f}", flush=True)

    corridor = make_driving_corridor_v2()
    for crf in [33, 34, 35]:
        for ki in [180, 600]:
            r = run(ev, frames, f"combo_c{crf}_ki{ki}",
                    mask=corridor, blur_sigma=3.0,
                    crf=crf, fg=22, scale=0.45, keyint=ki, unsharp=0.50)
            if r: results.append(r)

    # Summary
    print("\n" + "="*70, flush=True)
    print("ALL RESULTS (sorted by score)", flush=True)
    print("="*70, flush=True)
    results.sort(key=lambda r: r[1])
    for i, (label, score, seg, pose, rate, sz) in enumerate(results):
        marker = ""
        if score < 2.13: marker = " *"
        if score < 2.10: marker = " **"
        if score < 1.95: marker = " *** BEATS LEADER"
        print(f"  {i+1:2d}. {score:.4f}  seg={seg:.4f} pose={pose:.4f} "
              f"rate={rate:.4f} sz={sz/1024:.0f}KB  {label}{marker}", flush=True)

    best = results[0]
    print(f"\nBEST: {best[1]:.4f} ({best[0]})", flush=True)
    print(f"  vs leader 1.95: {1.95-best[1]:+.4f}", flush=True)


if __name__ == '__main__':
    main()
