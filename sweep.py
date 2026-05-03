#!/usr/bin/env python
"""
Systematic parameter sweep for video compression optimization.
Sweeps encode params (CRF, scale, film-grain, tune, aq-mode) AND decode params
(unsharp strength, kernel size) jointly. Uses in-process GPU eval for speed.

Usage:
    python sweep.py                          # full sweep
    python sweep.py --phase 1               # quick: film-grain & CRF only
    python sweep.py --phase 2               # unsharp + tune + aq
    python sweep.py --phase 3               # fine-tune around best
"""

import os, sys, csv, time, math, gc, subprocess, zipfile, argparse, tempfile
from pathlib import Path
from itertools import product

os.environ["PYTHONUTF8"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import einops
from PIL import Image

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
INPUT_VIDEO = VIDEOS_DIR / "0.mkv"
ORIGINAL_SIZE = 37_545_489
W_CAM, H_CAM, C = 1164, 874, 3
FRAME_BYTES = H_CAM * W_CAM * C
NUM_FRAMES = 1200

WORK_DIR = ROOT / "_sweep_tmp"
RESULTS_CSV = ROOT / "sweep_results.csv"


def get_resolution(scale):
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    return w, h


class Evaluator:
    """Loads SegNet+PoseNet once, caches ground truth, evaluates quickly."""

    def __init__(self, device):
        from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
        from safetensors.torch import load_file
        from frame_utils import camera_size, segnet_model_input_size

        self.device = torch.device(device)
        print(f"Loading models on {self.device}...")

        self.segnet = SegNet().eval().to(self.device)
        self.segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(self.device)))
        self.posenet = PoseNet().eval().to(self.device)
        self.posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(self.device)))

        self._cache_ground_truth()
        print(f"Evaluator ready. GT: {self.gt_seg.shape[0]} samples.")

    def _cache_ground_truth(self):
        cache_path = ROOT / "_sweep_cache" / "gt.pt"
        if cache_path.exists():
            gt = torch.load(cache_path, weights_only=True)
            self.gt_seg = gt['seg']
            self.gt_pose = gt['pose']
            return

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print("Caching ground truth (one-time)...")
        from frame_utils import AVVideoDataset
        ds = AVVideoDataset(['0.mkv'], data_dir=VIDEOS_DIR,
                            batch_size=16, device=torch.device('cpu'))
        ds.prepare_data()

        seg_list, pose_list = [], []
        with torch.inference_mode():
            for _, idx, batch in ds:
                batch = batch.to(self.device)
                x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
                seg_in = self.segnet.preprocess_input(x)
                seg_list.append(self.segnet(seg_in).argmax(1).cpu())
                pn_in = self.posenet.preprocess_input(x)
                pose_list.append(self.posenet(pn_in)['pose'][:, :6].cpu())

        self.gt_seg = torch.cat(seg_list)
        self.gt_pose = torch.cat(pose_list)
        torch.save({'seg': self.gt_seg, 'pose': self.gt_pose}, cache_path)

    def evaluate_raw(self, raw_path, archive_bytes):
        N = self.gt_seg.shape[0]
        raw = np.fromfile(raw_path, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)

        seg_dists, pose_dists = [], []
        bs = 16

        with torch.inference_mode():
            for i in range(0, N, bs):
                end = min(i + bs, N)
                f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(self.device).float()
                f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(self.device).float()
                x = torch.stack([f0, f1], dim=1)
                x = einops.rearrange(x, 'b t h w c -> b t c h w')

                seg_in = self.segnet.preprocess_input(x)
                seg_pred = self.segnet(seg_in).argmax(1)
                gt_seg = self.gt_seg[i:end].to(self.device)
                seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())

                pn_in = self.posenet.preprocess_input(x)
                pn_out = self.posenet(pn_in)['pose'][:, :6]
                gt_pose = self.gt_pose[i:end].to(self.device)
                pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())

        seg_d = np.mean(seg_dists)
        pose_d = np.mean(pose_dists)
        rate = archive_bytes / ORIGINAL_SIZE
        score = 100 * seg_d + math.sqrt(10 * pose_d) + 25 * rate

        return {
            'seg_dist': seg_d,
            'pose_dist': pose_d,
            'rate': rate,
            'seg_score': 100 * seg_d,
            'pose_score': math.sqrt(10 * pose_d),
            'rate_score': 25 * rate,
            'score': score,
            'archive_bytes': archive_bytes,
        }


def compress(input_path, output_path, crf, scale, film_grain, preset=0,
             tune=0, aq_mode=0, keyint=180, enable_qm=0, extra_svt_params=""):
    w, h = get_resolution(scale)
    svt_parts = [f"film-grain={film_grain}", f"keyint={keyint}", "scd=0"]
    if tune != 0:
        svt_parts.append(f"tune={tune}")
    if aq_mode != 0:
        svt_parts.append(f"aq-mode={aq_mode}")
    if enable_qm:
        svt_parts.append("enable-qm=1")
    if extra_svt_params:
        svt_parts.append(extra_svt_params)

    svt_str = ":".join(svt_parts)

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-r', '20', '-fflags', '+genpts',
        '-i', str(input_path),
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', svt_str,
        '-r', '20',
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  COMPRESS FAILED: {r.stderr[:300]}")
    return r.returncode == 0


def make_archive(mkv_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(mkv_path, '0.mkv')
    return zip_path.stat().st_size


def inflate(mkv_path, raw_path, unsharp_strength=0.40, kernel_taps=9,
            upscale_method='lanczos'):
    """Decode, upscale, and apply unsharp mask."""
    import av
    from frame_utils import yuv420_to_rgb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if kernel_taps == 5:
        _r = torch.tensor([1., 4., 6., 4., 1.])
    elif kernel_taps == 7:
        _r = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    elif kernel_taps == 9:
        _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    elif kernel_taps == 11:
        _r = torch.tensor([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
    else:
        raise ValueError(f"Unsupported kernel_taps={kernel_taps}")

    kernel = (torch.outer(_r, _r) / (_r.sum()**2)).to(device).expand(3, 1, kernel_taps, kernel_taps)
    pad = kernel_taps // 2

    if upscale_method == 'lanczos':
        pil_method = Image.LANCZOS
    elif upscale_method == 'bicubic':
        pil_method = Image.BICUBIC
    else:
        pil_method = Image.LANCZOS

    container = av.open(str(mkv_path))
    stream = container.streams.video[0]
    n = 0
    with open(str(raw_path), 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != H_CAM or W != W_CAM:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((W_CAM, H_CAM), pil_method)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(device)
                if unsharp_strength > 0:
                    blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                                    kernel, padding=0, groups=3)
                    x = x + unsharp_strength * (x - blur)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


def run_config(cfg, evaluator):
    """Run a single configuration: compress -> inflate -> evaluate."""
    work = WORK_DIR / f"run_{os.getpid()}"
    work.mkdir(parents=True, exist_ok=True)

    mkv_path = work / "0.mkv"
    zip_path = work / "archive.zip"
    raw_path = work / "0.raw"

    t0 = time.time()

    ok = compress(
        INPUT_VIDEO, mkv_path,
        crf=cfg['crf'], scale=cfg['scale'], film_grain=cfg['film_grain'],
        preset=cfg.get('preset', 0), tune=cfg.get('tune', 0),
        aq_mode=cfg.get('aq_mode', 0), keyint=cfg.get('keyint', 180),
        enable_qm=cfg.get('enable_qm', 0),
        extra_svt_params=cfg.get('extra_svt_params', ''),
    )
    if not ok:
        return None

    archive_bytes = make_archive(mkv_path, zip_path)

    n = inflate(
        mkv_path, raw_path,
        unsharp_strength=cfg.get('unsharp_strength', 0.40),
        kernel_taps=cfg.get('kernel_taps', 9),
        upscale_method=cfg.get('upscale_method', 'lanczos'),
    )
    if n != NUM_FRAMES:
        print(f"  WARNING: got {n} frames, expected {NUM_FRAMES}")
        return None

    results = evaluator.evaluate_raw(str(raw_path), archive_bytes)
    results['elapsed'] = time.time() - t0

    # Clean up
    for p in [mkv_path, zip_path, raw_path]:
        if p.exists():
            p.unlink()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, default=0,
                        help='1=film-grain+CRF, 2=unsharp+tune+aq, 3=fine-tune, 0=full')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    evaluator = Evaluator(args.device)

    if args.phase == 1:
        configs = []
        for crf in [30, 31, 32, 33, 34, 35]:
            for fg in [0, 4, 8, 12, 16, 22]:
                for scale in [0.45, 0.48, 0.50]:
                    configs.append({
                        'crf': crf, 'scale': scale, 'film_grain': fg,
                        'label': f'p1_crf{crf}_fg{fg}_s{scale}'
                    })

    elif args.phase == 2:
        configs = []
        for unsharp in [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]:
            for kernel in [5, 7, 9, 11]:
                configs.append({
                    'crf': 33, 'scale': 0.45, 'film_grain': 8,
                    'unsharp_strength': unsharp, 'kernel_taps': kernel,
                    'label': f'p2_ush{unsharp}_k{kernel}'
                })
        for tune in [0, 1, 2]:
            for aq in [0, 1, 2]:
                configs.append({
                    'crf': 33, 'scale': 0.45, 'film_grain': 8,
                    'tune': tune, 'aq_mode': aq,
                    'label': f'p2_tune{tune}_aq{aq}'
                })

    elif args.phase == 3:
        configs = []
        # will be filled after phases 1+2 reveal best params
        # For now, placeholder fine-tune grid
        for crf in [31, 32, 33]:
            for fg in [4, 6, 8, 10]:
                for ush in [0.30, 0.35, 0.40, 0.45, 0.50]:
                    for scale in [0.44, 0.45, 0.46, 0.47]:
                        configs.append({
                            'crf': crf, 'scale': scale, 'film_grain': fg,
                            'unsharp_strength': ush,
                            'label': f'p3_c{crf}_fg{fg}_s{scale}_u{ush}'
                        })

    else:
        configs = []
        for crf in [30, 31, 32, 33, 34, 35]:
            for fg in [0, 4, 8, 12, 16, 22]:
                for scale in [0.43, 0.45, 0.47, 0.50]:
                    for ush in [0.0, 0.30, 0.40, 0.50, 0.60]:
                        configs.append({
                            'crf': crf, 'scale': scale, 'film_grain': fg,
                            'unsharp_strength': ush,
                            'label': f'full_c{crf}_fg{fg}_s{scale}_u{ush}'
                        })

    existing = set()
    fieldnames = [
        'label', 'crf', 'scale', 'film_grain', 'preset', 'tune', 'aq_mode',
        'keyint', 'enable_qm', 'unsharp_strength', 'kernel_taps', 'upscale_method',
        'archive_bytes', 'rate', 'seg_dist', 'pose_dist',
        'seg_score', 'pose_score', 'rate_score', 'score', 'elapsed',
    ]

    csv_mode = 'w'
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                existing.add(row.get('label', ''))
        csv_mode = 'a'
        print(f"Resuming: {len(existing)} configs already done")

    remaining = [c for c in configs if c.get('label', '') not in existing]
    print(f"\nConfigs to run: {len(remaining)} (of {len(configs)} total)")

    best_score = float('inf')
    best_cfg = None

    with open(RESULTS_CSV, csv_mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csv_mode == 'w':
            writer.writeheader()

        for i, cfg in enumerate(remaining):
            label = cfg.get('label', f'cfg_{i}')
            print(f"\n[{i+1}/{len(remaining)}] {label}")

            results = run_config(cfg, evaluator)

            if results is None:
                print("  FAILED")
                continue

            row = {
                'label': label,
                'crf': cfg.get('crf', ''),
                'scale': cfg.get('scale', ''),
                'film_grain': cfg.get('film_grain', ''),
                'preset': cfg.get('preset', 0),
                'tune': cfg.get('tune', 0),
                'aq_mode': cfg.get('aq_mode', 0),
                'keyint': cfg.get('keyint', 180),
                'enable_qm': cfg.get('enable_qm', 0),
                'unsharp_strength': cfg.get('unsharp_strength', 0.40),
                'kernel_taps': cfg.get('kernel_taps', 9),
                'upscale_method': cfg.get('upscale_method', 'lanczos'),
                'archive_bytes': results['archive_bytes'],
                'rate': f"{results['rate']:.8f}",
                'seg_dist': f"{results['seg_dist']:.8f}",
                'pose_dist': f"{results['pose_dist']:.8f}",
                'seg_score': f"{results['seg_score']:.4f}",
                'pose_score': f"{results['pose_score']:.4f}",
                'rate_score': f"{results['rate_score']:.4f}",
                'score': f"{results['score']:.4f}",
                'elapsed': f"{results['elapsed']:.1f}",
            }

            writer.writerow(row)
            csvfile.flush()

            s = results['score']
            tag = ""
            if s < best_score:
                best_score = s
                best_cfg = label
                tag = " *** NEW BEST ***"

            print(f"  score={s:.4f} seg={results['seg_score']:.4f} "
                  f"pose={results['pose_score']:.4f} rate={results['rate_score']:.4f} "
                  f"size={results['archive_bytes']/1024:.0f}KB "
                  f"({results['elapsed']:.0f}s){tag}")

    print(f"\n{'='*60}")
    print(f"Sweep complete. Best: {best_score:.4f} ({best_cfg})")
    print(f"Results: {RESULTS_CSV}")
    print(f"{'='*60}")

    # Print top results
    if RESULTS_CSV.exists():
        rows = list(csv.DictReader(open(RESULTS_CSV)))
        rows = [r for r in rows if r.get('score')]
        rows.sort(key=lambda r: float(r['score']))
        print(f"\n=== Top 15 ===")
        for i, r in enumerate(rows[:15]):
            print(f"  {i+1}. {r['score']:>7s}  {r['label']:<40s}  "
                  f"seg={r['seg_score']} pose={r['pose_score']} rate={r['rate_score']} "
                  f"sz={r['archive_bytes']}")


if __name__ == '__main__':
    main()
