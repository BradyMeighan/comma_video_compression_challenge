#!/usr/bin/env python
"""
ROI preprocessing v3: Memory-efficient pipeline using stdin pipe to ffmpeg.
No large intermediate raw files on disk.
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
        if cache.exists():
            gt = torch.load(cache, weights_only=True)
        else:
            cache.parent.mkdir(parents=True, exist_ok=True)
            print("Caching GT...", flush=True)
            from frame_utils import AVVideoDataset
            ds = AVVideoDataset(['0.mkv'], data_dir=VIDEOS_DIR, batch_size=16, device=torch.device('cpu'))
            ds.prepare_data()
            sl, pl = [], []
            with torch.inference_mode():
                for _, _, batch in ds:
                    batch = batch.to(self.device)
                    x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
                    sl.append(self.segnet(self.segnet.preprocess_input(x)).argmax(1).cpu())
                    pl.append(self.posenet(self.posenet.preprocess_input(x))['pose'][:, :6].cpu())
            gt = {'seg': torch.cat(sl), 'pose': torch.cat(pl)}
            torch.save(gt, cache)
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
        score = 100*seg_d + math.sqrt(10*pose_d) + 25*rate
        return score, 100*seg_d, math.sqrt(10*pose_d), 25*rate


def load_video_frames():
    import av
    from frame_utils import yuv420_to_rgb
    container = av.open(str(INPUT_VIDEO))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return frames


def get_seg_maps(device='cuda'):
    cache = ROOT / "_eval_cache" / "seg_maps.pt"
    if cache.exists():
        return torch.load(cache, weights_only=True)

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
                seg_full = F.interpolate(seg_out.float().unsqueeze(1),
                                         size=(H_CAM, W_CAM), mode='nearest')
                for b in range(seg_full.shape[0]):
                    all_maps.append(seg_full[b, 0].byte().cpu())

    seg_maps = torch.stack(all_maps[:NUM_FRAMES])
    torch.save(seg_maps, cache)
    del segnet; torch.cuda.empty_cache()
    return seg_maps


def make_boundary_roi(seg_maps, edge_dilate=21, base_importance=0.3):
    """Protect class boundaries; allow blur in flat regions."""
    rois = []
    sobel_x_k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y_k = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    for i in range(seg_maps.shape[0]):
        seg = seg_maps[i].float().unsqueeze(0).unsqueeze(0)
        sx = F.conv2d(seg, sobel_x_k, padding=1)
        sy = F.conv2d(seg, sobel_y_k, padding=1)
        edges = ((sx.abs() + sy.abs()) > 0).float()
        dilated = F.max_pool2d(edges, kernel_size=edge_dilate,
                               stride=1, padding=edge_dilate//2)
        importance = torch.full_like(dilated, base_importance)
        importance = torch.max(importance, dilated)
        importance = F.avg_pool2d(importance, kernel_size=31, stride=1, padding=15)
        rois.append(importance.squeeze().clamp(0, 1))

    return torch.stack(rois)


def encode_with_roi(frames, roi_maps, mkv_path, blur_sigma=3.0,
                    crf=35, fg=22, scale=0.45, keyint=600, preset=0, bilateral=False):
    """Preprocess frames and pipe directly to ffmpeg (no raw file on disk)."""
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

    for i, frame in enumerate(frames):
        if blur_sigma > 0 and roi_maps is not None:
            importance = roi_maps[i].numpy()
            if bilateral:
                blurred = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
            else:
                blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_sigma)
            alpha = importance[..., np.newaxis]
            result = (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)
            proc.stdin.write(result.tobytes())
        else:
            proc.stdin.write(frame.tobytes())

    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        print(f"  ENCODE FAIL: {proc.stderr.read().decode()[:200]}", flush=True)
        return False
    return True


def inflate_mkv(mkv_path, raw_path, unsharp=0.50, kernel_taps=9):
    import av
    from frame_utils import yuv420_to_rgb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    if kernel_taps == 11:
        _r = torch.tensor([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
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
                    blur = F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'),
                                    kernel, padding=0, groups=3)
                    x = x + unsharp * (x - blur)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


def run_config(evaluator, frames, label, roi_maps=None, blur_sigma=3.0,
               crf=35, fg=22, scale=0.45, keyint=600, unsharp=0.50,
               kernel_taps=9, bilateral=False):
    work = ROOT / "_roi_work"
    work.mkdir(parents=True, exist_ok=True)

    mkv_path = work / "0.mkv"
    zip_path = work / "archive.zip"
    raw_path = work / "0.raw"

    t0 = time.time()

    ok = encode_with_roi(frames, roi_maps, mkv_path,
                         blur_sigma=blur_sigma, crf=crf, fg=fg,
                         scale=scale, keyint=keyint, bilateral=bilateral)
    if not ok:
        return None
    t_enc = time.time() - t0

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(mkv_path, '0.mkv')
    arch_bytes = zip_path.stat().st_size

    n = inflate_mkv(mkv_path, raw_path, unsharp, kernel_taps)
    t_inf = time.time() - t0 - t_enc

    score, seg, pose, rate = evaluator.evaluate_raw(str(raw_path), arch_bytes)
    elapsed = time.time() - t0

    for p in [mkv_path, zip_path, raw_path]:
        p.unlink(missing_ok=True)

    print(f"  {label}: score={score:.4f} seg={seg:.4f} pose={pose:.4f} "
          f"rate={rate:.4f} sz={arch_bytes/1024:.0f}KB "
          f"({t_enc:.0f}s+{t_inf:.0f}s+{elapsed-t_enc-t_inf:.0f}s)", flush=True)

    return (label, score, seg, pose, rate, arch_bytes, elapsed)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = FastEval(device)

    print("\nLoading frames...", flush=True)
    frames = load_video_frames()
    print(f"Loaded {len(frames)} frames", flush=True)

    print("Generating seg maps...", flush=True)
    seg_maps = get_seg_maps(device)
    print(f"Seg maps: {seg_maps.shape}", flush=True)

    results = []

    # Baseline (no ROI)
    print("\n" + "="*70, flush=True)
    print("BASELINE (no ROI, best sweep params)", flush=True)
    print("="*70, flush=True)
    r = run_config(evaluator, frames, "baseline_c35_fg22_ki600",
                   crf=35, fg=22, keyint=600, scale=0.45, unsharp=0.50)
    if r: results.append(r)

    # Test higher film-grain
    print("\n" + "="*70, flush=True)
    print("HIGHER FILM-GRAIN (no ROI)", flush=True)
    print("="*70, flush=True)
    for fg_val in [26, 30, 35, 40]:
        r = run_config(evaluator, frames, f"fg{fg_val}_c35_ki600",
                       crf=35, fg=fg_val, keyint=600, scale=0.45, unsharp=0.50)
        if r: results.append(r)

    # ROI with boundary protection, varying blur and edge_dilate
    print("\n" + "="*70, flush=True)
    print("ROI: Boundary-based, varying blur strength", flush=True)
    print("="*70, flush=True)

    for ed in [11, 21, 41]:
        roi = make_boundary_roi(seg_maps, edge_dilate=ed, base_importance=0.3)
        for sigma in [2.0, 3.0, 5.0]:
            r = run_config(evaluator, frames, f"roi_ed{ed}_s{sigma}",
                           roi_maps=roi, blur_sigma=sigma,
                           crf=35, fg=22, keyint=600, scale=0.45)
            if r: results.append(r)

    # ROI with varying base importance (how much to preserve in flat regions)
    print("\n" + "="*70, flush=True)
    print("ROI: Varying base importance", flush=True)
    print("="*70, flush=True)
    for base in [0.1, 0.2, 0.4, 0.5, 0.6, 0.7]:
        roi = make_boundary_roi(seg_maps, edge_dilate=21, base_importance=base)
        r = run_config(evaluator, frames, f"roi_base{base}_ed21_s3",
                       roi_maps=roi, blur_sigma=3.0,
                       crf=35, fg=22, keyint=600, scale=0.45)
        if r: results.append(r)

    # Bilateral filter ROI
    print("\n" + "="*70, flush=True)
    print("ROI: Bilateral filter", flush=True)
    print("="*70, flush=True)
    roi = make_boundary_roi(seg_maps, edge_dilate=21, base_importance=0.3)
    r = run_config(evaluator, frames, "roi_bilateral_ed21",
                   roi_maps=roi, blur_sigma=3.0, bilateral=True,
                   crf=35, fg=22, keyint=600, scale=0.45)
    if r: results.append(r)

    # Best ROI + CRF combinations
    print("\n" + "="*70, flush=True)
    print("COMBINED: Best ROI + CRF sweep", flush=True)
    print("="*70, flush=True)
    roi_results = [r for r in results if r[0].startswith('roi_')]
    if roi_results:
        best_roi_label = min(roi_results, key=lambda r: r[1])[0]
        print(f"Best ROI so far: {best_roi_label}", flush=True)

    roi = make_boundary_roi(seg_maps, edge_dilate=21, base_importance=0.3)
    for crf in [31, 32, 33, 34]:
        for fg in [22, 26]:
            r = run_config(evaluator, frames, f"combo_c{crf}_fg{fg}_roi",
                           roi_maps=roi, blur_sigma=3.0,
                           crf=crf, fg=fg, keyint=600, scale=0.45)
            if r: results.append(r)

    # Summary
    print("\n" + "="*70, flush=True)
    print("ALL RESULTS (sorted by score)", flush=True)
    print("="*70, flush=True)
    results.sort(key=lambda r: r[1])
    for i, (label, score, seg, pose, rate, sz, el) in enumerate(results):
        marker = " *** BEATS BASELINE" if score < 2.1038 else ""
        if score < 1.95:
            marker = " *** BEATS LEADER!"
        print(f"  {i+1:2d}. {score:.4f}  seg={seg:.4f} pose={pose:.4f} "
              f"rate={rate:.4f} {label}{marker}", flush=True)

    if results:
        best = results[0]
        print(f"\nBEST: {best[1]:.4f} ({best[0]})", flush=True)
        print(f"  vs baseline 2.1304: {2.1304-best[1]:+.4f}", flush=True)
        print(f"  vs sweep best 2.1038: {2.1038-best[1]:+.4f}", flush=True)
        print(f"  vs leader 1.95: {1.95-best[1]:+.4f}", flush=True)


if __name__ == '__main__':
    main()
