#!/usr/bin/env python
"""
Deep tuning: test advanced encoding strategies to close the gap to 1.95.
Current best: 2.0223 (sky blur + piped RGB + CRF=33, fg=22, ush=0.45)
Target: 1.95

Strategies:
1. Different codecs (vvenc, x265)
2. SVT-AV1 advanced params (sharpness, variance boost, QP comp)
3. 10-bit encoding
4. Pre-sharpen before encode (boost edges the encoder should preserve)
5. Model-native resolution (512x384)
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


def make_sky_mask(top_frac=0.15, side_frac=0.03):
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * top_frac)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * side_frac)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM-1-x] = np.minimum(mask[:, W_CAM-1-x], t)
    return cv2.GaussianBlur(mask, (31, 31), 10)


def apply_sky_blur(frames, mask, sigma=5.0):
    ksize = int(sigma * 6) | 1
    alpha = mask[..., np.newaxis]
    out = []
    for frame in frames:
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        out.append((frame * alpha + blurred * (1 - alpha)).astype(np.uint8))
    return out


def encode_generic(frames, mkv_path, codec_cmd):
    """Encode frames using arbitrary ffmpeg command (codec part)."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
    ] + codec_cmd + ['-r', '20', str(mkv_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        print(f"  ENCODE FAIL: {stderr.decode()[:200]}", flush=True)
    return proc.returncode == 0


def inflate_mkv(mkv_path, raw_path, unsharp=0.45, kernel_taps=9):
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


def run(ev, frames, label, codec_cmd, unsharp=0.45):
    work = ROOT / "_dt_work"
    work.mkdir(parents=True, exist_ok=True)
    mkv = work / "0.mkv"
    zp = work / "archive.zip"
    raw = work / "0.raw"

    t0 = time.time()
    ok = encode_generic(frames, mkv, codec_cmd)
    if not ok:
        return None

    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
        z.write(mkv, '0.mkv')
    arch = zp.stat().st_size

    inflate_mkv(mkv, raw, unsharp)
    score, seg, pose, rate = ev.evaluate_raw(str(raw), arch)
    elapsed = time.time() - t0

    for p in [mkv, zp, raw]:
        p.unlink(missing_ok=True)

    print(f"  {label}: score={score:.4f} seg={seg:.4f} pose={pose:.4f} "
          f"rate={rate:.4f} sz={arch/1024:.0f}KB ({elapsed:.0f}s)", flush=True)
    return (label, score, seg, pose, rate, arch)


def svt_cmd(w, h, crf=33, fg=22, keyint=180, preset=0, extra_svt="",
            pix_fmt='yuv420p', extra_vf=""):
    vf = f'scale={w}:{h}:flags=lanczos'
    if extra_vf:
        vf = f'{extra_vf},{vf}'

    svt_parts = [f'film-grain={fg}', f'keyint={keyint}', 'scd=0']
    if extra_svt:
        svt_parts.append(extra_svt)

    return [
        '-vf', vf,
        '-pix_fmt', pix_fmt,
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', ':'.join(svt_parts),
    ]


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = FastEval(device)

    print("\nLoading frames...", flush=True)
    raw_frames = load_frames()
    print(f"Loaded {len(raw_frames)} frames", flush=True)

    mask = make_sky_mask(0.15, 0.03)
    frames = apply_sky_blur(raw_frames, mask, sigma=5.0)
    print("Applied sky blur", flush=True)

    results = []

    # Baseline: current best config
    print("\n" + "="*70, flush=True)
    print("BASELINE: sky blur + CRF=33, fg=22", flush=True)
    print("="*70, flush=True)
    r = run(ev, frames, "baseline", svt_cmd(522, 392, crf=33, fg=22), unsharp=0.45)
    if r: results.append(r)

    # SVT-AV1 advanced params
    print("\n" + "="*70, flush=True)
    print("SVT-AV1 ADVANCED PARAMS", flush=True)
    print("="*70, flush=True)

    # Sharpness param (0-7, higher = stronger sharpening in loop filter)
    for sharp in [1, 2, 3, 4]:
        r = run(ev, frames, f"sharp{sharp}",
                svt_cmd(522, 392, crf=33, fg=22, extra_svt=f"sharpness={sharp}"))
        if r: results.append(r)

    # Variance boost (strength for adaptive QP based on variance)
    for vb in [1, 2, 3]:
        r = run(ev, frames, f"varboost{vb}",
                svt_cmd(522, 392, crf=33, fg=22, extra_svt=f"variance-boost-strength={vb}"))
        if r: results.append(r)

    # QP scale compression (controls QP range compression)
    for qpsc in [1, 2, 3]:
        r = run(ev, frames, f"qpsc{qpsc}",
                svt_cmd(522, 392, crf=33, fg=22, extra_svt=f"qp-scale-compress-strength={qpsc}"))
        if r: results.append(r)

    # Enable overlays (may help with compression)
    r = run(ev, frames, "overlays",
            svt_cmd(522, 392, crf=33, fg=22, extra_svt="enable-overlays=1"))
    if r: results.append(r)

    # 10-bit encoding
    print("\n" + "="*70, flush=True)
    print("10-BIT ENCODING", flush=True)
    print("="*70, flush=True)
    for crf10 in [33, 35, 37]:
        r = run(ev, frames, f"10bit_c{crf10}",
                svt_cmd(522, 392, crf=crf10, fg=22, pix_fmt='yuv420p10le'))
        if r: results.append(r)

    # Model-native resolution 512x384
    print("\n" + "="*70, flush=True)
    print("RESOLUTION TESTS", flush=True)
    print("="*70, flush=True)
    for w, h in [(512, 384), (524, 394), (530, 398), (548, 412)]:
        r = run(ev, frames, f"res{w}x{h}",
                svt_cmd(w, h, crf=33, fg=22))
        if r: results.append(r)

    # Pre-sharpen before encode (boost edges)
    print("\n" + "="*70, flush=True)
    print("PRE-SHARPEN BEFORE ENCODE", flush=True)
    print("="*70, flush=True)
    for amt in ['0.5', '1.0', '1.5']:
        r = run(ev, frames, f"presharp_{amt}",
                svt_cmd(522, 392, crf=33, fg=22,
                        extra_vf=f'unsharp=5:5:{amt}:5:5:0'))
        if r: results.append(r)

    # x265 codec comparison
    print("\n" + "="*70, flush=True)
    print("X265 CODEC", flush=True)
    print("="*70, flush=True)
    for crf265 in [26, 28, 30, 32]:
        cmd265 = [
            '-vf', 'scale=522:392:flags=lanczos',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libx265', '-preset', 'veryslow', '-crf', str(crf265),
            '-x265-params', 'keyint=180:min-keyint=60:bframes=3:b-adapt=2:ref=5:me=umh:subme=7:rd=6:aq-mode=3:log-level=warning',
        ]
        r = run(ev, frames, f"x265_c{crf265}", cmd265)
        if r: results.append(r)

    # VVC codec (libvvenc)
    print("\n" + "="*70, flush=True)
    print("VVC CODEC (libvvenc)", flush=True)
    print("="*70, flush=True)
    for qp in [30, 32, 34, 36]:
        cmd_vvc = [
            '-vf', 'scale=522:392:flags=lanczos',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libvvenc', '-qp', str(qp),
            '-vvenc-params', 'preset=slow',
        ]
        r = run(ev, frames, f"vvc_q{qp}", cmd_vvc)
        if r: results.append(r)

    # Combining best SVT params
    print("\n" + "="*70, flush=True)
    print("COMBINED BEST SVT PARAMS", flush=True)
    print("="*70, flush=True)
    if results:
        svt_results = [r for r in results if r[0].startswith('sharp') or r[0].startswith('var') or r[0].startswith('qpsc')]
        if svt_results:
            best_svt = min(svt_results, key=lambda x: x[1])
            print(f"  Best SVT param: {best_svt[0]} = {best_svt[1]:.4f}", flush=True)

    for combo in [
        "sharpness=2:variance-boost-strength=1",
        "sharpness=1:qp-scale-compress-strength=1",
        "sharpness=2:enable-overlays=1",
    ]:
        r = run(ev, frames, f"combo_{combo[:30]}",
                svt_cmd(522, 392, crf=33, fg=22, extra_svt=combo))
        if r: results.append(r)

    # Summary
    print("\n" + "="*70, flush=True)
    print("ALL RESULTS (sorted)", flush=True)
    print("="*70, flush=True)
    results.sort(key=lambda r: r[1])
    for i, (label, score, seg, pose, rate, sz) in enumerate(results[:30]):
        marker = " ***" if score < 1.95 else (" **" if score < 2.00 else (" *" if score < 2.05 else ""))
        print(f"  {i+1:2d}. {score:.4f}  seg={seg:.4f} pose={pose:.4f} "
              f"rate={rate:.4f} sz={sz/1024:.0f}KB  {label}{marker}", flush=True)

    best = results[0]
    print(f"\nBEST: {best[1]:.4f} ({best[0]})", flush=True)
    print(f"  vs leader 1.95: {1.95-best[1]:+.4f}", flush=True)


if __name__ == '__main__':
    main()
