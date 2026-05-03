#!/usr/bin/env python
"""
Targeted parameter sweep - tests specific high-impact configurations.
Each run: compress ~90s, inflate ~40s, eval ~50s = ~3 min per config.
"""

import os, sys, csv, time, math, gc, subprocess, zipfile
from pathlib import Path

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
NUM_FRAMES = 1200
WORK_DIR = ROOT / "_sweep_tmp"
RESULTS_CSV = ROOT / "targeted_sweep_results.csv"


def get_resolution(scale):
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    return w, h


class FastEval:
    def __init__(self, device='cuda'):
        from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
        from safetensors.torch import load_file

        self.device = torch.device(device)
        print(f"Loading models on {self.device}...", flush=True)
        self.segnet = SegNet().eval().to(self.device)
        self.segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(self.device)))
        self.posenet = PoseNet().eval().to(self.device)
        self.posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(self.device)))

        cache = ROOT / "_sweep_cache" / "gt.pt"
        if cache.exists():
            gt = torch.load(cache, weights_only=True)
            self.gt_seg, self.gt_pose = gt['seg'], gt['pose']
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
            self.gt_seg, self.gt_pose = torch.cat(sl), torch.cat(pl)
            torch.save({'seg': self.gt_seg, 'pose': self.gt_pose}, cache)

        print(f"Ready. {self.gt_seg.shape[0]} samples.", flush=True)

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
                seg_pred = self.segnet(self.segnet.preprocess_input(x)).argmax(1)
                sd.extend((seg_pred != self.gt_seg[i:end].to(self.device)).float().mean((1,2)).cpu().tolist())
                pose_pred = self.posenet(self.posenet.preprocess_input(x))['pose'][:, :6]
                pd.extend((pose_pred - self.gt_pose[i:end].to(self.device)).pow(2).mean(1).cpu().tolist())
        seg_d, pose_d = np.mean(sd), np.mean(pd)
        rate = archive_bytes / ORIGINAL_SIZE
        return seg_d, pose_d, rate, 100*seg_d + math.sqrt(10*pose_d) + 25*rate


def compress_and_eval(evaluator, label, crf=33, scale=0.45, film_grain=22,
                      preset=0, tune=0, aq_mode=0, keyint=180,
                      unsharp_strength=0.40, kernel_taps=9,
                      enable_qm=0, extra_svt='',
                      preprocess_cmd=None, input_video=None):
    work = WORK_DIR / "run"
    work.mkdir(parents=True, exist_ok=True)
    for f in work.iterdir():
        f.unlink()

    src = input_video or str(INPUT_VIDEO)
    mkv = work / "0.mkv"
    raw = work / "0.raw"
    zp = work / "archive.zip"

    w, h = get_resolution(scale)
    svt = [f"film-grain={film_grain}", f"keyint={keyint}", "scd=0"]
    if tune: svt.append(f"tune={tune}")
    if aq_mode: svt.append(f"aq-mode={aq_mode}")
    if enable_qm: svt.append("enable-qm=1")
    if extra_svt: svt.append(extra_svt)

    t0 = time.time()
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-r', '20', '-fflags', '+genpts', '-i', src,
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', ':'.join(svt), '-r', '20', str(mkv),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  ENCODE FAIL: {r.stderr[:200]}", flush=True)
        return None

    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_STORED) as z:
        z.write(mkv, '0.mkv')
    arch_bytes = zp.stat().st_size
    t_enc = time.time() - t0

    import av
    from frame_utils import yuv420_to_rgb
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ktaps = kernel_taps
    if ktaps == 5: _r = torch.tensor([1., 4., 6., 4., 1.])
    elif ktaps == 7: _r = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    elif ktaps == 9: _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    elif ktaps == 11: _r = torch.tensor([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
    else: _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    kernel = (torch.outer(_r, _r) / (_r.sum()**2)).to(device).expand(3, 1, ktaps, ktaps)
    pad = ktaps // 2

    container = av.open(str(mkv))
    n = 0
    with open(str(raw), 'wb') as f:
        for frame in container.decode(container.streams.video[0]):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != H_CAM or W != W_CAM:
                pil = Image.fromarray(t.numpy()).resize((W_CAM, H_CAM), Image.LANCZOS)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(device)
                if unsharp_strength > 0:
                    blur = F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'), kernel, padding=0, groups=3)
                    x = x + unsharp_strength * (x - blur)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    t_inf = time.time() - t0 - t_enc

    seg_d, pose_d, rate, score = evaluator.evaluate(str(raw), arch_bytes)
    elapsed = time.time() - t0

    print(f"  {label}: score={score:.4f} seg={100*seg_d:.4f} pose={math.sqrt(10*pose_d):.4f} "
          f"rate={25*rate:.4f} sz={arch_bytes/1024:.0f}KB "
          f"(enc={t_enc:.0f}s inf={t_inf:.0f}s eval={elapsed-t_enc-t_inf:.0f}s)",
          flush=True)

    return {
        'label': label, 'score': score,
        'seg': 100*seg_d, 'pose': math.sqrt(10*pose_d), 'rate': 25*rate,
        'seg_d': seg_d, 'pose_d': pose_d, 'archive_bytes': arch_bytes,
        'crf': crf, 'scale': scale, 'film_grain': film_grain,
        'unsharp': unsharp_strength, 'kernel': kernel_taps,
        'tune': tune, 'aq_mode': aq_mode, 'elapsed': elapsed,
    }


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    ev = FastEval('cuda' if torch.cuda.is_available() else 'cpu')

    all_results = []

    def run(label, **kw):
        r = compress_and_eval(ev, label, **kw)
        if r:
            all_results.append(r)
        return r

    print("\n" + "="*70, flush=True)
    print("STAGE 1: Film-grain sweep (biggest suspected impact)", flush=True)
    print("="*70, flush=True)
    # Baseline uses fg=22. Research suggests lower is better for machine vision.
    for fg in [0, 4, 8, 12, 16, 22]:
        run(f"fg={fg}", film_grain=fg)

    print("\n" + "="*70, flush=True)
    print("STAGE 2: CRF sweep with best film-grain", flush=True)
    print("="*70, flush=True)
    best_fg = min([r for r in all_results if r['label'].startswith('fg=')],
                  key=lambda r: r['score'])
    fg_best = best_fg['film_grain']
    print(f"Best film-grain: {fg_best} (score={best_fg['score']:.4f})", flush=True)

    for crf in [29, 30, 31, 32, 34, 35, 36]:
        run(f"crf={crf}_fg={fg_best}", crf=crf, film_grain=fg_best)

    print("\n" + "="*70, flush=True)
    print("STAGE 3: Unsharp strength sweep", flush=True)
    print("="*70, flush=True)
    best_crf_r = min([r for r in all_results if r['label'].startswith('crf=') or r['label'] == f'fg={fg_best}'],
                     key=lambda r: r['score'])
    crf_best = best_crf_r['crf']
    print(f"Best CRF: {crf_best} (score={best_crf_r['score']:.4f})", flush=True)

    for ush in [0.0, 0.15, 0.25, 0.30, 0.35, 0.45, 0.50, 0.60, 0.70]:
        run(f"ush={ush}_c{crf_best}_fg{fg_best}",
            crf=crf_best, film_grain=fg_best, unsharp_strength=ush)

    print("\n" + "="*70, flush=True)
    print("STAGE 4: Scale sweep", flush=True)
    print("="*70, flush=True)
    best_ush_r = min([r for r in all_results if 'ush=' in r['label']],
                     key=lambda r: r['score'])
    ush_best = best_ush_r['unsharp']
    print(f"Best unsharp: {ush_best} (score={best_ush_r['score']:.4f})", flush=True)

    for sc in [0.40, 0.42, 0.43, 0.44, 0.46, 0.47, 0.48, 0.50]:
        run(f"sc={sc}_c{crf_best}_fg{fg_best}_u{ush_best}",
            crf=crf_best, film_grain=fg_best, unsharp_strength=ush_best, scale=sc)

    print("\n" + "="*70, flush=True)
    print("STAGE 5: Tune mode + AQ mode", flush=True)
    print("="*70, flush=True)
    best_sc_r = min([r for r in all_results if 'sc=' in r['label']],
                    key=lambda r: r['score'], default=best_ush_r)
    sc_best = best_sc_r['scale'] if 'sc=' in best_sc_r['label'] else 0.45
    print(f"Best scale: {sc_best} (score={best_sc_r['score']:.4f})", flush=True)

    for tune in [1, 2]:
        run(f"tune={tune}_c{crf_best}_fg{fg_best}_u{ush_best}_s{sc_best}",
            crf=crf_best, film_grain=fg_best, unsharp_strength=ush_best,
            scale=sc_best, tune=tune)

    for aq in [1, 2]:
        run(f"aq={aq}_c{crf_best}_fg{fg_best}_u{ush_best}_s{sc_best}",
            crf=crf_best, film_grain=fg_best, unsharp_strength=ush_best,
            scale=sc_best, aq_mode=aq)

    print("\n" + "="*70, flush=True)
    print("STAGE 6: Kernel size sweep", flush=True)
    print("="*70, flush=True)
    for k in [5, 7, 11]:
        run(f"k={k}_c{crf_best}_fg{fg_best}_u{ush_best}_s{sc_best}",
            crf=crf_best, film_grain=fg_best, unsharp_strength=ush_best,
            scale=sc_best, kernel_taps=k)

    print("\n" + "="*70, flush=True)
    print("STAGE 7: Enable QM + keyint variations", flush=True)
    print("="*70, flush=True)
    run(f"qm=1_c{crf_best}_fg{fg_best}_u{ush_best}_s{sc_best}",
        crf=crf_best, film_grain=fg_best, unsharp_strength=ush_best,
        scale=sc_best, enable_qm=1)

    for ki in [120, 240, 300, 600]:
        run(f"ki={ki}_c{crf_best}_fg{fg_best}_u{ush_best}_s{sc_best}",
            crf=crf_best, film_grain=fg_best, unsharp_strength=ush_best,
            scale=sc_best, keyint=ki)

    # Summary
    print("\n" + "="*70, flush=True)
    print("RESULTS SUMMARY (sorted by score)", flush=True)
    print("="*70, flush=True)
    all_results.sort(key=lambda r: r['score'])
    for i, r in enumerate(all_results):
        marker = " ***" if r['score'] < 2.13 else ""
        print(f"  {i+1:2d}. {r['score']:.4f}  seg={r['seg']:.4f} pose={r['pose']:.4f} "
              f"rate={r['rate']:.4f}  {r['label']}{marker}", flush=True)

    best = all_results[0]
    print(f"\nBEST: {best['score']:.4f} - {best['label']}", flush=True)
    print(f"  vs baseline 2.1304: improvement = {2.1304 - best['score']:.4f}", flush=True)

    # Save to CSV
    with open(RESULTS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\nResults saved to {RESULTS_CSV}", flush=True)


if __name__ == '__main__':
    main()
