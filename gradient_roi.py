#!/usr/bin/env python
"""
Gradient-based saliency ROI: compute per-pixel gradient magnitudes from SegNet
to determine which pixels actually affect model output. Blur only truly 
unimportant pixels (zero/low gradient). Also test spatially-varying post-processing.
"""

import os, sys, time, math, subprocess, zipfile, gc
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


def compute_saliency_maps(device='cuda'):
    """Compute per-pixel gradient saliency from SegNet for each frame."""
    cache = ROOT / "_eval_cache" / "saliency.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)

    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file

    dev = torch.device(device)
    segnet = SegNet().eval().to(dev)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(dev)))

    # Enable gradient computation
    for p in segnet.parameters():
        p.requires_grad_(False)

    import av
    from frame_utils import yuv420_to_rgb
    container = av.open(str(INPUT_VIDEO))
    frames_rgb = []
    for frame in container.decode(container.streams.video[0]):
        frames_rgb.append(yuv420_to_rgb(frame))
    container.close()

    print(f"Computing saliency for {len(frames_rgb)} frames...", flush=True)
    saliency_maps = []

    for idx in range(len(frames_rgb)):
        frame_tensor = frames_rgb[idx].float().permute(2, 0, 1).unsqueeze(0).to(dev)
        frame_tensor.requires_grad_(True)

        seg_in = F.interpolate(frame_tensor, size=(384, 512), mode='bilinear', align_corners=False)
        seg_out = segnet(seg_in)

        # Gradient of max logit sum w.r.t. input
        loss = seg_out.max(1).values.sum()
        loss.backward()

        grad = frame_tensor.grad.abs().mean(1).squeeze()
        grad_full = grad

        # Normalize to [0, 1]
        gmax = grad_full.max()
        if gmax > 0:
            grad_full = grad_full / gmax

        saliency_maps.append(grad_full.detach().cpu())

        frame_tensor.grad = None
        segnet.zero_grad()

        if (idx + 1) % 200 == 0:
            print(f"  {idx+1}/{len(frames_rgb)}", flush=True)
            torch.cuda.empty_cache()

    result = torch.stack(saliency_maps)
    torch.save(result, cache)
    del segnet; torch.cuda.empty_cache()
    print(f"Saved saliency maps: {result.shape}", flush=True)
    return result


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


def encode_piped(frames, mkv_path, crf=33, fg=22, scale=0.45, keyint=180):
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', str(crf),
        '-svtav1-params', f'film-grain={fg}:keyint={keyint}:scd=0',
        '-r', '20', str(mkv_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    _, stderr = proc.communicate()
    return proc.returncode == 0


def inflate_with_saliency_unsharp(mkv_path, raw_path, saliency_maps,
                                   base_unsharp=0.45, edge_boost=0.3):
    """Spatially-varying unsharp: stronger at high-saliency regions."""
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

                blur = F.conv2d(F.pad(x, (4,4,4,4), mode='reflect'), kernel, padding=0, groups=3)
                detail = x - blur

                if saliency_maps is not None and n < saliency_maps.shape[0]:
                    sal = saliency_maps[n].to(device)
                    # Higher saliency → more sharpening
                    strength = base_unsharp + edge_boost * sal
                    strength = strength.unsqueeze(0).unsqueeze(0)
                    x = x + strength * detail
                else:
                    x = x + base_unsharp * detail

                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


def inflate_standard(mkv_path, raw_path, unsharp=0.45):
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


def run(ev, label, frames, saliency_maps=None, inflate_fn=None, inflate_kw=None,
        crf=33, fg=22, scale=0.45, keyint=180):
    work = ROOT / "_grad_work"
    work.mkdir(parents=True, exist_ok=True)
    mkv = work / "0.mkv"
    zp = work / "archive.zip"
    raw = work / "0.raw"

    t0 = time.time()
    ok = encode_piped(frames, mkv, crf, fg, scale, keyint)
    if not ok: return None

    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
        z.write(mkv, '0.mkv')
    arch = zp.stat().st_size

    if inflate_fn:
        inflate_fn(mkv, raw, **(inflate_kw or {}))
    else:
        inflate_standard(mkv, raw)

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
    raw_frames = load_frames()

    mask = make_sky_mask(0.15, 0.03)
    alpha = mask[..., np.newaxis]
    ksize = 31
    frames_sky = []
    for f in raw_frames:
        bl = cv2.GaussianBlur(f, (ksize, ksize), 5.0)
        frames_sky.append((f * alpha + bl * (1 - alpha)).astype(np.uint8))

    print("Computing saliency maps...", flush=True)
    saliency = compute_saliency_maps(device)

    results = []

    # Baseline
    print("\n" + "="*70, flush=True)
    print("BASELINE", flush=True)
    print("="*70, flush=True)
    r = run(ev, "baseline_sky", frames_sky)
    if r: results.append(r)

    # Saliency-guided pre-blur
    print("\n" + "="*70, flush=True)
    print("SALIENCY-GUIDED PRE-BLUR", flush=True)
    print("="*70, flush=True)

    for percentile in [10, 20, 30, 50]:
        threshold = np.percentile(saliency.numpy(), percentile)
        frames_sal = []
        for i, f in enumerate(raw_frames):
            sal = saliency[i].numpy()
            importance = (sal > threshold).astype(np.float32)
            importance = cv2.GaussianBlur(importance, (31, 31), 10)
            importance = np.clip(importance, 0.1, 1.0)
            bl = cv2.GaussianBlur(f, (ksize, ksize), 5.0)
            frames_sal.append((f * importance[..., None] + bl * (1 - importance[..., None])).astype(np.uint8))
        r = run(ev, f"saliency_p{percentile}", frames_sal)
        if r: results.append(r)
        del frames_sal

    # Combined: saliency + sky mask
    print("\n" + "="*70, flush=True)
    print("COMBINED: Saliency + Sky mask", flush=True)
    print("="*70, flush=True)
    for percentile in [20, 30]:
        threshold = np.percentile(saliency.numpy(), percentile)
        frames_combo = []
        for i, f in enumerate(raw_frames):
            sal = saliency[i].numpy()
            sal_importance = (sal > threshold).astype(np.float32)
            sal_importance = cv2.GaussianBlur(sal_importance, (31, 31), 10)
            sal_importance = np.clip(sal_importance, 0.1, 1.0)
            combined_importance = np.minimum(mask, sal_importance)
            combined_importance = np.maximum(combined_importance, 0.1)
            bl = cv2.GaussianBlur(f, (ksize, ksize), 5.0)
            frames_combo.append((f * combined_importance[..., None] + bl * (1 - combined_importance[..., None])).astype(np.uint8))
        r = run(ev, f"sal+sky_p{percentile}", frames_combo)
        if r: results.append(r)
        del frames_combo

    # Spatially-varying unsharp (post-processing)
    print("\n" + "="*70, flush=True)
    print("SPATIALLY-VARYING UNSHARP", flush=True)
    print("="*70, flush=True)
    for base, boost in [(0.40, 0.20), (0.35, 0.30), (0.30, 0.40), (0.45, 0.30)]:
        r = run(ev, f"svar_u{base}_b{boost}", frames_sky,
                inflate_fn=inflate_with_saliency_unsharp,
                inflate_kw={'saliency_maps': saliency, 'base_unsharp': base, 'edge_boost': boost})
        if r: results.append(r)

    # Summary
    print("\n" + "="*70, flush=True)
    print("RESULTS", flush=True)
    print("="*70, flush=True)
    results.sort(key=lambda r: r[1])
    for i, (label, score, seg, pose, rate, sz) in enumerate(results):
        marker = " ***" if score < 1.95 else (" **" if score < 2.00 else " *" if score < 2.03 else "")
        print(f"  {i+1:2d}. {score:.4f}  seg={seg:.4f} pose={pose:.4f} "
              f"rate={rate:.4f} sz={sz/1024:.0f}KB  {label}{marker}", flush=True)

    best = results[0]
    print(f"\nBEST: {best[1]:.4f} ({best[0]})", flush=True)
    print(f"  vs leader 1.95: {1.95-best[1]:+.4f}", flush=True)


if __name__ == '__main__':
    main()
