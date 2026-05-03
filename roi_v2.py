#!/usr/bin/env python
"""
ROI preprocessing v2: Multiple strategies, sweep blur params.
Key insight: we need to denoise regions that DON'T matter to SegNet/PoseNet.
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
        cache = ROOT / "_sweep_cache" / "gt.pt"
        gt = torch.load(cache, weights_only=True)
        self.gt_seg, self.gt_pose = gt['seg'], gt['pose']
        print(f"Eval ready. {self.gt_seg.shape[0]} samples.", flush=True)

    def evaluate(self, raw_path, archive_bytes):
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
        score = 100*seg_d + math.sqrt(10*pose_d) + 25*rate
        return score, 100*seg_d, math.sqrt(10*pose_d), 25*rate, archive_bytes


def load_video_frames():
    """Load all frames from the original video."""
    import av
    from frame_utils import yuv420_to_rgb
    container = av.open(str(INPUT_VIDEO))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return frames


def get_segmentation_maps(device='cuda'):
    """Get per-frame segmentation predictions."""
    cache = ROOT / "_roi_cache" / "seg_maps.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file
    from frame_utils import AVVideoDataset

    dev = torch.device(device)
    segnet = SegNet().eval().to(dev)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(dev)))

    ds = AVVideoDataset(['0.mkv'], data_dir=VIDEOS_DIR, batch_size=8, device=torch.device('cpu'))
    ds.prepare_data()

    all_maps = []
    with torch.inference_mode():
        for _, _, batch in ds:
            batch = batch.to(dev)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
            for t in range(x.shape[1]):
                frame = x[:, t]
                seg_in = F.interpolate(frame, size=(384, 512), mode='bilinear')
                seg_out = segnet(seg_in).argmax(1)
                seg_full = F.interpolate(seg_out.float().unsqueeze(1), size=(H_CAM, W_CAM), mode='nearest')
                for b in range(seg_full.shape[0]):
                    all_maps.append(seg_full[b, 0].byte().cpu())

    seg_maps = torch.stack(all_maps[:NUM_FRAMES])
    torch.save(seg_maps, cache)
    del segnet; torch.cuda.empty_cache()
    print(f"Cached {seg_maps.shape[0]} seg maps", flush=True)
    return seg_maps


def make_roi_from_segmaps(seg_maps, edge_dilate=21, smooth_size=41):
    """
    Create ROI mask from segmentation maps.
    Key insight: class BOUNDARIES are what SegNet cares about most.
    So we protect boundaries with high importance, and let flat regions be blurred.
    """
    rois = []
    for i in range(seg_maps.shape[0]):
        seg = seg_maps[i].float()

        importance = torch.ones_like(seg) * 0.5

        # Detect class boundaries using gradient
        sobel_x = F.conv2d(
            seg.unsqueeze(0).unsqueeze(0),
            torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            padding=1
        )
        sobel_y = F.conv2d(
            seg.unsqueeze(0).unsqueeze(0),
            torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            padding=1
        )
        edges = (sobel_x.abs() + sobel_y.abs()).squeeze() > 0

        # Dilate edges - protect a wide band around class boundaries
        edge_float = edges.float().unsqueeze(0).unsqueeze(0)
        edge_dilated = F.max_pool2d(edge_float, kernel_size=edge_dilate,
                                     stride=1, padding=edge_dilate//2).squeeze()

        importance = torch.where(edge_dilated > 0, torch.ones_like(importance), importance)

        # Smooth the importance map
        importance_4d = importance.unsqueeze(0).unsqueeze(0)
        importance = F.avg_pool2d(importance_4d, kernel_size=smooth_size,
                                  stride=1, padding=smooth_size//2).squeeze()

        rois.append(importance.clamp(0, 1))

    return torch.stack(rois)


def preprocess_frames(frames, roi_maps, blur_sigma=3.0, bilateral=False):
    """Apply ROI-guided denoising to frames."""
    ksize = int(blur_sigma * 6) | 1
    out_frames = []

    for i, frame in enumerate(frames):
        importance = roi_maps[i].numpy()

        if bilateral:
            blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_sigma)

        alpha = importance[..., np.newaxis]
        result = (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)
        out_frames.append(result)

    return out_frames


def write_raw(frames, path):
    with open(str(path), 'wb') as f:
        for frame in frames:
            f.write(frame.tobytes())


def encode_raw_to_mkv(raw_path, mkv_path, w, h, crf=35, fg=22, keyint=600, preset=0):
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', str(raw_path),
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', f'film-grain={fg}:keyint={keyint}:scd=0',
        '-r', '20', str(mkv_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return r.returncode == 0


def inflate_mkv(mkv_path, raw_path, unsharp=0.50, kernel_taps=9):
    import av
    from frame_utils import yuv420_to_rgb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if kernel_taps == 9: _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    elif kernel_taps == 11: _r = torch.tensor([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
    else: _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    kernel = (torch.outer(_r, _r) / (_r.sum()**2)).to(device).expand(3, 1, kernel_taps, kernel_taps)
    pad = kernel_taps // 2

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
                    blur = F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'), kernel, padding=0, groups=3)
                    x = x + unsharp * (x - blur)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


def run_one_config(evaluator, frames, roi_maps, label,
                   blur_sigma=3.0, bilateral=False,
                   crf=35, fg=22, scale=0.45, keyint=600,
                   unsharp=0.50, kernel_taps=9):
    work = ROOT / "_roi_tmp"
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2

    preprocessed = preprocess_frames(frames, roi_maps, blur_sigma, bilateral)
    preproc_raw = work / "preproc.raw"
    write_raw(preprocessed, preproc_raw)
    del preprocessed

    mkv_path = work / "0.mkv"
    ok = encode_raw_to_mkv(preproc_raw, mkv_path, w, h, crf, fg, keyint)
    preproc_raw.unlink(missing_ok=True)
    if not ok:
        print(f"  {label}: ENCODE FAILED", flush=True)
        return None

    zip_path = work / "archive.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(mkv_path, '0.mkv')
    arch_bytes = zip_path.stat().st_size

    inflated_raw = work / "inflated.raw"
    n = inflate_mkv(mkv_path, inflated_raw, unsharp, kernel_taps)

    score, seg, pose, rate, _ = evaluator.evaluate(str(inflated_raw), arch_bytes)
    elapsed = time.time() - t0

    for p in [mkv_path, zip_path, inflated_raw]:
        p.unlink(missing_ok=True)

    print(f"  {label}: score={score:.4f} seg={seg:.4f} pose={pose:.4f} "
          f"rate={rate:.4f} sz={arch_bytes/1024:.0f}KB ({elapsed:.0f}s)", flush=True)

    return score, seg, pose, rate, arch_bytes, elapsed


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = FastEval(device)

    print("\nLoading video frames...", flush=True)
    frames = load_video_frames()
    print(f"Loaded {len(frames)} frames", flush=True)

    print("\nGenerating segmentation maps...", flush=True)
    seg_maps = get_segmentation_maps(device)
    print(f"Seg maps: {seg_maps.shape}", flush=True)

    results = []

    # Test 1: ROI from seg boundaries with various blur strengths
    print("\n" + "="*70, flush=True)
    print("TEST 1: Segmentation boundary ROI, varying blur", flush=True)
    print("="*70, flush=True)

    for edge_dilate in [11, 21, 41]:
        roi_maps = make_roi_from_segmaps(seg_maps, edge_dilate=edge_dilate, smooth_size=41)
        for sigma in [2.0, 3.0, 5.0, 8.0]:
            label = f"seg_ed{edge_dilate}_s{sigma}"
            r = run_one_config(evaluator, frames, roi_maps, label,
                               blur_sigma=sigma, crf=35, scale=0.45, keyint=600)
            if r:
                results.append((label, *r))

    # Test 2: Bilateral filter instead of Gaussian
    print("\n" + "="*70, flush=True)
    print("TEST 2: Bilateral filter", flush=True)
    print("="*70, flush=True)

    roi_maps = make_roi_from_segmaps(seg_maps, edge_dilate=21, smooth_size=41)
    label = "bilateral_ed21"
    r = run_one_config(evaluator, frames, roi_maps, label,
                       bilateral=True, crf=35, scale=0.45, keyint=600)
    if r:
        results.append((label, *r))

    # Test 3: Higher film-grain values (trend suggests more is better)
    print("\n" + "="*70, flush=True)
    print("TEST 3: Higher film-grain values (no ROI)", flush=True)
    print("="*70, flush=True)

    no_blur_roi = torch.ones(NUM_FRAMES, H_CAM, W_CAM)
    for fg_val in [26, 30, 35, 40, 50]:
        label = f"fg{fg_val}_c35"
        r = run_one_config(evaluator, frames, no_blur_roi, label,
                           blur_sigma=0, crf=35, fg=fg_val, scale=0.45, keyint=600)
        if r:
            results.append((label, *r))

    # Test 4: Best ROI + higher film-grain
    if results:
        # Find best ROI result
        roi_results = [r for r in results if r[0].startswith('seg_') or r[0].startswith('bilateral')]
        if roi_results:
            best_roi = min(roi_results, key=lambda r: r[1])
            print(f"\nBest ROI config: {best_roi[0]} score={best_roi[1]:.4f}", flush=True)

        fg_results = [r for r in results if r[0].startswith('fg')]
        if fg_results:
            best_fg = min(fg_results, key=lambda r: r[1])
            print(f"Best film-grain: {best_fg[0]} score={best_fg[1]:.4f}", flush=True)

    # Test 5: Combined best ROI + best fg + best params from sweep
    print("\n" + "="*70, flush=True)
    print("TEST 5: Combined optimizations", flush=True)
    print("="*70, flush=True)

    roi_maps = make_roi_from_segmaps(seg_maps, edge_dilate=21, smooth_size=41)
    for crf in [33, 34, 35]:
        for fg in [22, 26, 30]:
            for sigma in [2.0, 3.0]:
                label = f"combo_c{crf}_fg{fg}_s{sigma}"
                r = run_one_config(evaluator, frames, roi_maps, label,
                                   blur_sigma=sigma, crf=crf, fg=fg,
                                   scale=0.45, keyint=600)
                if r:
                    results.append((label, *r))

    # Summary
    print("\n" + "="*70, flush=True)
    print("RESULTS SUMMARY (sorted by score)", flush=True)
    print("="*70, flush=True)
    results.sort(key=lambda r: r[1])
    for i, (label, score, seg, pose, rate, sz, elapsed) in enumerate(results):
        marker = " ***" if score < 2.1038 else ""
        print(f"  {i+1:2d}. {score:.4f}  seg={seg:.4f} pose={pose:.4f} "
              f"rate={rate:.4f} sz={sz/1024:.0f}KB  {label}{marker}", flush=True)

    if results:
        best = results[0]
        print(f"\nBEST: {best[1]:.4f} - {best[0]}", flush=True)
        print(f"  vs baseline 2.1304: delta={2.1304 - best[1]:.4f}", flush=True)
        print(f"  vs param sweep best 2.1038: delta={2.1038 - best[1]:.4f}", flush=True)
        print(f"  vs leader 1.95: delta={1.95 - best[1]:.4f}", flush=True)


if __name__ == '__main__':
    main()
