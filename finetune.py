#!/usr/bin/env python
"""
Fine-tuning sweep around best config (sky blur + leader params).
Focus on: CRF fine-tune, unsharp fine-tune, sky params, ffmpeg denoise, 
keyint, and combining multiple improvements.

Best so far: 2.0236 (sky0.15_s5.0 with CRF=33, fg=22, s=0.45, ush=0.40, ki=180)
Target: < 1.95
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


def encode_piped(frames, mkv_path, mask=None, blur_sigma=5.0,
                 crf=33, fg=22, scale=0.45, keyint=180, preset=0,
                 extra_vf="", extra_svt=""):
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    ksize = int(blur_sigma * 6) | 1

    vf = f'scale={w}:{h}:flags=lanczos'
    if extra_vf:
        vf = f'{extra_vf},{vf}'

    svt_parts = [f'film-grain={fg}', f'keyint={keyint}', 'scd=0']
    if extra_svt:
        svt_parts.append(extra_svt)

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', vf,
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', ':'.join(svt_parts),
        '-r', '20', str(mkv_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    alpha = mask[..., np.newaxis] if mask is not None else None

    for frame in frames:
        if alpha is not None and blur_sigma > 0:
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


def run(ev, frames, label, mask=None, blur_sigma=5.0,
        crf=33, fg=22, scale=0.45, keyint=180, unsharp=0.40,
        extra_vf="", extra_svt=""):
    work = ROOT / "_ft_work"
    work.mkdir(parents=True, exist_ok=True)
    mkv = work / "0.mkv"
    zp = work / "archive.zip"
    raw = work / "0.raw"

    t0 = time.time()
    ok = encode_piped(frames, mkv, mask, blur_sigma,
                      crf, fg, scale, keyint, extra_vf=extra_vf, extra_svt=extra_svt)
    if not ok:
        print(f"  {label}: FAIL", flush=True)
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev = FastEval(device)

    print("\nLoading frames...", flush=True)
    frames = load_frames()
    print(f"Loaded {len(frames)} frames", flush=True)

    results = []
    mask = make_sky_mask(top_frac=0.15, side_frac=0.03)

    # Reproduce best: sky0.15_s5.0 with CRF=33
    print("\n" + "="*70, flush=True)
    print("REPRODUCE BEST + FINE-TUNE CRF", flush=True)
    print("="*70, flush=True)
    for crf in [30, 31, 32, 33, 34]:
        r = run(ev, frames, f"sky_c{crf}", mask=mask, crf=crf)
        if r: results.append(r)

    # Fine-tune unsharp with best sky config
    print("\n" + "="*70, flush=True)
    print("UNSHARP FINE-TUNE with sky blur", flush=True)
    print("="*70, flush=True)
    for ush in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]:
        r = run(ev, frames, f"sky_u{ush}", mask=mask, crf=33, unsharp=ush)
        if r: results.append(r)

    # Sky mask params fine-tune
    print("\n" + "="*70, flush=True)
    print("SKY MASK PARAMS", flush=True)
    print("="*70, flush=True)
    for top in [0.12, 0.18, 0.20]:
        for sigma in [4.0, 6.0, 8.0]:
            m = make_sky_mask(top_frac=top, side_frac=0.03)
            r = run(ev, frames, f"top{top}_sig{sigma}", mask=m, blur_sigma=sigma, crf=33)
            if r: results.append(r)

    # ffmpeg filter: hqdn3d (hardware-optimized denoiser)
    print("\n" + "="*70, flush=True)
    print("FFMPEG DENOISE FILTERS (no sky mask, denoise in ffmpeg)", flush=True)
    print("="*70, flush=True)
    for strength in ['2:2:3:3', '4:3:6:4.5', '1.5:1.5:2:2']:
        r = run(ev, frames, f"hqdn3d_{strength.replace(':','_')}",
                extra_vf=f'hqdn3d={strength}', crf=33, blur_sigma=0)
        if r: results.append(r)

    # nlmeans (non-local means denoiser) 
    for strength in ['3:1:7:3', '5:3:7:3', '2:1:7:3']:
        r = run(ev, frames, f"nlmeans_{strength.replace(':','_')}",
                extra_vf=f'nlmeans={strength}', crf=33, blur_sigma=0)
        if r: results.append(r)

    # Combine sky mask + ffmpeg denoise
    print("\n" + "="*70, flush=True)
    print("COMBINED: Sky mask + ffmpeg denoise", flush=True)
    print("="*70, flush=True)
    for dn in ['hqdn3d=2:2:3:3', 'hqdn3d=4:3:6:4.5', 'nlmeans=3:1:7:3']:
        r = run(ev, frames, f"sky+{dn.replace('=','_').replace(':','_')}",
                mask=mask, extra_vf=dn, crf=33)
        if r: results.append(r)

    # Combine best sky + best unsharp + CRF fine-tune
    print("\n" + "="*70, flush=True)
    print("BEST COMBINATIONS", flush=True)
    print("="*70, flush=True)
    best_so_far = min(results, key=lambda x: x[1]) if results else None
    if best_so_far:
        print(f"  Current best: {best_so_far[0]} = {best_so_far[1]:.4f}", flush=True)

    # Try with keyint variations
    for ki in [120, 240, 600]:
        r = run(ev, frames, f"sky_ki{ki}", mask=mask, crf=33, keyint=ki)
        if r: results.append(r)

    # Try scale variations with sky mask
    for sc in [0.43, 0.44, 0.46, 0.47, 0.48]:
        r = run(ev, frames, f"sky_sc{sc}", mask=mask, crf=33, scale=sc)
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
