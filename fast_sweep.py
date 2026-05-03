#!/usr/bin/env python
"""
Fast two-phase codec parameter sweep.

Phase 1 (Proxy): Encode a 10-second clip with fast preset, decode + upscale +
evaluate entirely in-process on GPU. No .raw disk round-trip. Tests hundreds
of configs in minutes instead of hours.

Phase 2 (Validation): Re-encode top N candidates with preset 0 on the full
60-second video. Full in-process GPU evaluation with archive.zip rate.

Usage:
    python fast_sweep.py --device cuda --benchmark
    python fast_sweep.py --device cuda --proxy-only
    python fast_sweep.py --device cuda                # both phases
    python fast_sweep.py --device cuda --validate-only # from existing proxy CSV
"""

import os, sys, csv, time, math, gc, shutil, subprocess, argparse, zipfile
from pathlib import Path
from itertools import product

os.environ["PYTHONUTF8"] = "1"

_FFMPEG_FULL = Path(os.path.expanduser(
    "~/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
    "ffmpeg-8.1-full_build/bin"
))
if _FFMPEG_FULL.is_dir() and str(_FFMPEG_FULL) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_FFMPEG_FULL) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
import torch.nn.functional as F
import av

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
INPUT_VIDEO = VIDEOS_DIR / "0.mkv"
ORIGINAL_SIZE = 37_545_489
W_CAM, H_CAM = 1164, 874
NUM_FRAMES = 1200
SEQ_LEN = 2
FPS = 20
PROXY_FRAMES = 200  # 10 seconds


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def decode_video_frames(video_path, n_frames=None):
    """Decode video via PyAV + yuv420_to_rgb. Returns list of (H,W,3) uint8 tensors."""
    from frame_utils import yuv420_to_rgb

    fmt = 'hevc' if str(video_path).endswith('.hevc') else None
    container = av.open(str(video_path), format=fmt)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))
        if n_frames and len(frames) >= n_frames:
            break
    container.close()
    return frames


def upscale_frames_gpu(frames_list, target_h, target_w, device, batch_size=64):
    """
    Upscale (H,W,3) uint8 tensor list to (target_h, target_w) on GPU.
    Matches inflate.py: permute -> float -> bicubic -> clamp -> round -> uint8.
    Returns a single (N, target_h, target_w, 3) uint8 CPU tensor.
    """
    stacked = torch.stack(frames_list)
    n, h, w, _ = stacked.shape
    if h == target_h and w == target_w:
        return stacked

    parts = []
    for i in range(0, n, batch_size):
        batch = stacked[i:i+batch_size].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
        batch = F.interpolate(batch, size=(target_h, target_w), mode='bicubic', align_corners=False)
        parts.append(batch.clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu())
    return torch.cat(parts, dim=0)


def encode_video(input_path, output_path, codec, crf, preset, gop, width, height):
    """Encode with ffmpeg. Returns (success: bool, elapsed_seconds: float)."""
    if codec == "svtav1":
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
            "-r", str(FPS), "-fflags", "+genpts", "-i", str(input_path),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libsvtav1", "-preset", str(preset), "-crf", str(crf),
            "-g", str(gop),
            "-svtav1-params", "enable-overlays=1:film-grain=0:fast-decode=0",
            "-r", str(FPS), str(output_path),
        ]
    elif codec == "x265":
        cmd = [
            "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
            "-r", str(FPS), "-fflags", "+genpts", "-i", str(input_path),
            "-vf", f"scale={width}:{height}:flags=lanczos",
            "-c:v", "libx265", "-preset", str(preset), "-crf", str(crf),
            "-g", str(gop), "-bf", "3",
            "-x265-params",
            f"keyint={gop}:min-keyint={min(gop, 60)}:scenecut=40:bframes=3:"
            f"b-adapt=2:ref=5:me=umh:subme=7:rd=6:aq-mode=3:log-level=warning",
            "-r", str(FPS), str(output_path),
        ]
    else:
        return False, 0.0

    t0 = time.time()
    result = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True, timeout=3600
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"    ENCODE ERR: {result.stderr[:200]}")
    return result.returncode == 0, elapsed


def create_proxy_clip(source_path, dest_path, n_frames):
    """Copy first n_frames from source without re-encoding."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(source_path), "-frames:v", str(n_frames), "-c", "copy",
        str(dest_path),
    ]
    result = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=60)
    return result.returncode == 0


def check_encoder(name):
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        return name in r.stdout
    except Exception:
        return False


def safe_rmtree(path):
    for _ in range(3):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# FastEvaluator — the core optimisation: everything stays in memory
# ---------------------------------------------------------------------------

class FastEvaluator:
    """
    In-process GPU evaluator that eliminates the 3.4 GB .raw disk round-trip.
    Decode → GPU upscale → GPU inference, all in one process.
    """

    def __init__(self, device, gt_video_path, n_frames=None, batch_size=16):
        from modules import DistortionNet, segnet_sd_path, posenet_sd_path

        self.device = torch.device(device)
        self.batch_size = batch_size

        t0 = time.time()
        self.net = DistortionNet().eval().to(self.device)
        self.net.load_state_dicts(str(posenet_sd_path), str(segnet_sd_path), self.device)
        self._model_load_time = time.time() - t0
        print(f"  Models loaded in {self._model_load_time:.1f}s")

        t0 = time.time()
        gt_list = decode_video_frames(gt_video_path, n_frames=n_frames)
        self.gt_frames = torch.stack(gt_list)
        self.n_frames = self.gt_frames.shape[0]
        self._gt_decode_time = time.time() - t0
        print(f"  GT decoded: {self.n_frames} frames in {self._gt_decode_time:.1f}s")

        # Warmup CUDA kernels so first eval isn't 5x slower
        if self.device.type == 'cuda':
            print("  Warming up CUDA kernels...")
            t0 = time.time()
            with torch.inference_mode():
                dummy_gt = self.gt_frames[:SEQ_LEN].unsqueeze(0).to(self.device)
                self.net.compute_distortion(dummy_gt, dummy_gt)
            del dummy_gt
            torch.cuda.empty_cache()
            print(f"  Warmup done in {time.time()-t0:.1f}s")

    def evaluate_compressed(self, compressed_path, compressed_size=None,
                            scale_rate_to_frames=None):
        """
        Full eval pipeline in-process:
        1. Decode compressed MKV (PyAV, CPU)
        2. Upscale to 1164x874 (GPU, batched bicubic)
        3. Run DistortionNet (GPU)

        Returns dict with posenet_dist, segnet_dist, rate, score, timings.
        """
        timings = {}

        if compressed_size is None:
            compressed_size = os.path.getsize(compressed_path)

        # 1. Decode
        t0 = time.time()
        comp_list = decode_video_frames(compressed_path, n_frames=self.n_frames)
        timings["decode"] = time.time() - t0

        n_eval = min(len(comp_list), self.n_frames)

        # 2. GPU upscale
        t0 = time.time()
        comp_up = upscale_frames_gpu(
            comp_list[:n_eval], H_CAM, W_CAM, self.device, batch_size=64
        )
        timings["upscale"] = time.time() - t0
        del comp_list

        # Rate
        if scale_rate_to_frames is not None:
            est_full = compressed_size * (scale_rate_to_frames / n_eval)
            rate = est_full / ORIGINAL_SIZE
        else:
            rate = compressed_size / ORIGINAL_SIZE

        # 3. GPU inference
        t0 = time.time()
        n_seq = n_eval // SEQ_LEN
        psum = torch.zeros([], device=self.device)
        ssum = torch.zeros([], device=self.device)
        total = 0

        with torch.inference_mode():
            for start in range(0, n_seq, self.batch_size):
                end = min(start + self.batch_size, n_seq)
                gt_b, comp_b = [], []
                for s in range(start, end):
                    f0, f1 = s * SEQ_LEN, s * SEQ_LEN + SEQ_LEN
                    gt_b.append(self.gt_frames[f0:f1])
                    comp_b.append(comp_up[f0:f1])
                bg = torch.stack(gt_b).to(self.device)
                bc = torch.stack(comp_b).to(self.device)
                p, seg = self.net.compute_distortion(bg, bc)
                psum += p.sum()
                ssum += seg.sum()
                total += end - start

        timings["inference"] = time.time() - t0

        del comp_up
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        pd = (psum / total).item()
        sd = (ssum / total).item()
        score = 100 * sd + math.sqrt(pd * 10) + 25 * rate

        return {
            "posenet_dist": pd, "segnet_dist": sd,
            "rate": rate, "score": score,
            "compressed_size": compressed_size,
            "n_frames": n_eval,
            "timings": timings,
        }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark(evaluator, proxy_clip, device):
    print("\n" + "=" * 60)
    print("BENCHMARK — timing each pipeline component")
    print("=" * 60)

    tmp = ROOT / "_bench_tmp"
    tmp.mkdir(exist_ok=True)
    tw, th, crf, gop = 512, 384, 32, 180
    enc_times = {}

    for preset in [0, 2, 4, 6, 8]:
        out = tmp / f"p{preset}.mkv"
        ok, et = encode_video(proxy_clip, out, "svtav1", crf, preset, gop, tw, th)
        if ok:
            sz = out.stat().st_size
            enc_times[preset] = et
            print(f"  Encode preset {preset}: {et:6.2f}s   {sz:>10,} bytes")

    test_file = tmp / "p4.mkv"
    decode_t = upscale_t = infer_t = total_t = 0.0
    if test_file.exists():
        t0 = time.time()
        frames = decode_video_frames(test_file)
        decode_t = time.time() - t0
        print(f"  Decode {len(frames):>4d} frames   : {decode_t:6.2f}s")

        t0 = time.time()
        up = upscale_frames_gpu(frames, H_CAM, W_CAM, device, batch_size=64)
        upscale_t = time.time() - t0
        print(f"  GPU upscale            : {upscale_t:6.2f}s")

        n_seq = len(frames) // SEQ_LEN
        t0 = time.time()
        with torch.inference_mode():
            for s0 in range(0, n_seq, evaluator.batch_size):
                s1 = min(s0 + evaluator.batch_size, n_seq)
                gb, cb = [], []
                for s in range(s0, s1):
                    f0, f1 = s * SEQ_LEN, s * SEQ_LEN + SEQ_LEN
                    gb.append(evaluator.gt_frames[f0:f1])
                    cb.append(up[f0:f1])
                bg = torch.stack(gb).to(device)
                bc = torch.stack(cb).to(device)
                evaluator.net.compute_distortion(bg, bc)
        infer_t = time.time() - t0
        print(f"  GPU inference          : {infer_t:6.2f}s")

        del frames, up
        gc.collect()
        torch.cuda.empty_cache()

        t0 = time.time()
        evaluator.evaluate_compressed(
            str(test_file), test_file.stat().st_size, scale_rate_to_frames=NUM_FRAMES
        )
        total_t = time.time() - t0
        print(f"  Total in-process eval  : {total_t:6.2f}s")

    safe_rmtree(tmp)
    return enc_times, decode_t, upscale_t, infer_t, total_t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fast two-phase codec sweep")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark only, no sweep")
    parser.add_argument("--proxy-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--proxy-frames", type=int, default=PROXY_FRAMES)
    parser.add_argument("--proxy-preset", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--val-preset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--codec", default="svtav1", choices=["svtav1", "x265", "all"])
    parser.add_argument("--proxy-csv", type=Path, default=ROOT / "proxy_sweep_results.csv")
    parser.add_argument("--final-csv", type=Path, default=ROOT / "final_validation_results.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    has_svtav1 = check_encoder("libsvtav1")
    has_x265 = check_encoder("libx265")
    print(f"Encoders: svtav1={has_svtav1}  x265={has_x265}")
    print(f"Device:   {device}")

    if not INPUT_VIDEO.exists():
        print(f"ERROR: {INPUT_VIDEO} not found")
        sys.exit(1)

    # ----- proxy clip -----
    proxy_dir = ROOT / "proxy_cache"
    proxy_dir.mkdir(exist_ok=True)
    proxy_clip = proxy_dir / f"proxy_{args.proxy_frames}f.mkv"
    if not proxy_clip.exists():
        print(f"Creating proxy clip ({args.proxy_frames} frames)...")
        ok = create_proxy_clip(INPUT_VIDEO, proxy_clip, args.proxy_frames)
        if not ok:
            print("ERROR: proxy clip creation failed")
            sys.exit(1)
    print(f"Proxy clip: {proxy_clip}  ({proxy_clip.stat().st_size:,} bytes)")

    # ----- evaluator -----
    need_proxy_eval = not args.validate_only
    if need_proxy_eval:
        print("\nInitialising proxy evaluator...")
        proxy_eval = FastEvaluator(
            device, INPUT_VIDEO, n_frames=args.proxy_frames, batch_size=args.batch_size
        )

    # ----- benchmark -----
    if args.benchmark:
        enc_times, dec_t, up_t, inf_t, tot_t = run_benchmark(
            proxy_eval, proxy_clip, device
        )
        p4_enc = enc_times.get(4, 3.0)
        per_cfg = p4_enc + tot_t
        resolutions = [
            (384, 288), (416, 312), (440, 330), (464, 348),
            (480, 360), (496, 372), (512, 384), (522, 392),
        ]
        crfs = list(range(26, 42))
        proxy_gops = [g for g in [60, 120, 180, 200] if g <= args.proxy_frames]
        n_proxy = len(resolutions) * len(crfs) * len(proxy_gops)
        print(f"\n{'='*60}")
        print(f"ESTIMATED TIMES")
        print(f"{'='*60}")
        print(f"  Per proxy config : ~{per_cfg:.1f}s  (encode {p4_enc:.1f}s + eval {tot_t:.1f}s)")
        print(f"  Proxy grid       : {len(resolutions)} res x {len(crfs)} CRF x {len(proxy_gops)} GOP = {n_proxy}")
        print(f"  Proxy total      : ~{n_proxy * per_cfg / 60:.0f} min  ({n_proxy * per_cfg / 3600:.1f} hr)")
        frame_ratio = NUM_FRAMES / args.proxy_frames
        p0_enc_full = enc_times.get(0, 30.0) * frame_ratio
        full_eval_est = tot_t * frame_ratio
        val_per = p0_enc_full + full_eval_est
        n_val = args.top_n * 4  # top_n unique base configs × 4 GOPs
        print(f"  Per validation   : ~{val_per:.0f}s  (encode {p0_enc_full:.0f}s + eval {full_eval_est:.1f}s)")
        print(f"  Validation       : ~{n_val} configs => ~{n_val * val_per / 60:.0f} min  ({n_val * val_per / 3600:.1f} hr)")
        print(f"  Grand total      : ~{(n_proxy * per_cfg + n_val * val_per) / 3600:.1f} hr")
        return

    # ===================== PHASE 1 — PROXY SWEEP =====================
    resolutions = [
        (384, 288), (416, 312), (440, 330), (464, 348),
        (480, 360), (496, 372), (512, 384), (522, 392),
    ]
    crfs = list(range(26, 42))
    proxy_gops = [g for g in [60, 120, 180, 200] if g <= args.proxy_frames]
    if not proxy_gops:
        proxy_gops = [args.proxy_frames]
    validation_gops = [120, 180, 300, 600]

    if need_proxy_eval:
        proxy_configs = []
        if args.codec in ("svtav1", "all") and has_svtav1:
            for res, crf, gop in product(resolutions, crfs, proxy_gops):
                proxy_configs.append({
                    "codec": "svtav1", "width": res[0], "height": res[1],
                    "crf": crf, "preset": args.proxy_preset, "gop": gop,
                    "scale": f"{res[0]}x{res[1]}",
                })
        if args.codec in ("x265", "all") and has_x265:
            x265_presets = ["veryslow"]
            for res, crf, preset, gop in product(resolutions, crfs, x265_presets, proxy_gops):
                proxy_configs.append({
                    "codec": "x265", "width": res[0], "height": res[1],
                    "crf": crf, "preset": preset, "gop": gop,
                    "scale": f"{res[0]}x{res[1]}",
                })

        # resume support
        existing = set()
        if args.proxy_csv.exists():
            with open(args.proxy_csv) as f:
                for row in csv.DictReader(f):
                    k = (row["codec"], row["scale"], row["crf"],
                         str(row["preset"]), row["gop"])
                    existing.add(k)
            csv_mode = "a"
        else:
            csv_mode = "w"

        todo = [c for c in proxy_configs
                if (c["codec"], c["scale"], str(c["crf"]),
                    str(c["preset"]), str(c["gop"])) not in existing]

        print(f"\n{'='*60}")
        print(f"PHASE 1 — PROXY SWEEP")
        print(f"  Total grid : {len(proxy_configs)}")
        print(f"  Already done: {len(existing)}")
        print(f"  Remaining  : {len(todo)}")
        print(f"{'='*60}")

        fields = [
            "codec", "scale", "width", "height", "crf", "preset", "gop",
            "compressed_size", "rate", "segnet_dist", "posenet_dist",
            "segnet_score", "posenet_score", "rate_score", "total_score",
            "encode_time", "decode_time", "upscale_time", "inference_time",
            "total_time", "status",
        ]

        tmp = ROOT / "_sweep_tmp"
        tmp.mkdir(exist_ok=True)
        completed = 0
        times_hist = []
        best_score = 999.0

        with open(args.proxy_csv, csv_mode, newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if csv_mode == "w":
                writer.writeheader()

            for cfg in todo:
                t0_all = time.time()
                row = {k: cfg.get(k, "") for k in fields}
                out_path = tmp / "comp.mkv"

                ok, enc_t = encode_video(
                    proxy_clip, out_path, cfg["codec"],
                    cfg["crf"], cfg["preset"], cfg["gop"],
                    cfg["width"], cfg["height"],
                )
                row["encode_time"] = f"{enc_t:.2f}"
                if not ok:
                    row["status"] = "encode_failed"
                    row["total_time"] = f"{time.time()-t0_all:.2f}"
                    writer.writerow(row)
                    fh.flush()
                    continue

                comp_sz = out_path.stat().st_size
                try:
                    res = proxy_eval.evaluate_compressed(
                        str(out_path), comp_sz, scale_rate_to_frames=NUM_FRAMES
                    )
                except Exception as e:
                    row["status"] = f"eval_failed: {str(e)[:80]}"
                    row["total_time"] = f"{time.time()-t0_all:.2f}"
                    writer.writerow(row)
                    fh.flush()
                    continue

                total_t = time.time() - t0_all
                row.update({
                    "compressed_size": comp_sz,
                    "rate":           f"{res['rate']:.8f}",
                    "segnet_dist":    f"{res['segnet_dist']:.8f}",
                    "posenet_dist":   f"{res['posenet_dist']:.8f}",
                    "segnet_score":   f"{100*res['segnet_dist']:.4f}",
                    "posenet_score":  f"{math.sqrt(10*res['posenet_dist']):.4f}",
                    "rate_score":     f"{25*res['rate']:.4f}",
                    "total_score":    f"{res['score']:.4f}",
                    "decode_time":    f"{res['timings']['decode']:.2f}",
                    "upscale_time":   f"{res['timings']['upscale']:.2f}",
                    "inference_time": f"{res['timings']['inference']:.2f}",
                    "total_time":     f"{total_t:.2f}",
                    "status":         "ok",
                })
                writer.writerow(row)
                fh.flush()

                completed += 1
                times_hist.append(total_t)
                if res['score'] < best_score:
                    best_score = res['score']

                remaining = len(todo) - completed
                avg = sum(times_hist[-30:]) / len(times_hist[-30:])
                eta = remaining * avg / 60

                if completed <= 5 or completed % 20 == 0 or res['score'] < best_score + 0.01:
                    print(
                        f"  [{completed:>4d}/{len(todo)}] "
                        f"{cfg['scale']:>7s} crf={cfg['crf']:>2} gop={cfg['gop']:>3}  "
                        f"score={res['score']:.4f}  "
                        f"best={best_score:.4f}  "
                        f"{total_t:.1f}s  ETA {eta:.0f}m"
                    )

        safe_rmtree(tmp)
        print(f"\nProxy sweep done: {completed} configs, avg {sum(times_hist)/max(len(times_hist),1):.1f}s/cfg")

    # ===================== LOAD PROXY RESULTS =====================
    if not args.proxy_csv.exists():
        print("ERROR: no proxy results CSV found")
        sys.exit(1)

    proxy_rows = []
    with open(args.proxy_csv) as f:
        for r in csv.DictReader(f):
            if r.get("status") == "ok":
                proxy_rows.append(r)
    proxy_rows.sort(key=lambda r: float(r["total_score"]))

    print(f"\n{'='*60}")
    print(f"TOP {min(args.top_n, len(proxy_rows))} PROXY RESULTS")
    print(f"{'='*60}")
    for i, r in enumerate(proxy_rows[:args.top_n]):
        print(
            f"  {i+1:3d}. {float(r['total_score']):7.4f}  "
            f"{r['codec']} {r['scale']} crf={r['crf']} gop={r['gop']}  "
            f"seg={r['segnet_score']} pose={r['posenet_score']} rate={r['rate_score']}"
        )

    if args.proxy_only:
        print("\n--proxy-only flag set, skipping validation.")
        return

    # ===================== PHASE 2 — FULL VALIDATION =====================
    if need_proxy_eval:
        del proxy_eval
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"PHASE 2 — FULL VALIDATION")
    print(f"{'='*60}")
    print("Loading full-video evaluator...")
    full_eval = FastEvaluator(device, INPUT_VIDEO, n_frames=None, batch_size=args.batch_size)

    seen_base = set()
    val_configs = []
    for r in proxy_rows[:args.top_n]:
        base = (r["codec"], r["scale"], r["crf"])
        if base in seen_base:
            continue
        seen_base.add(base)
        for gop in validation_gops:
            val_configs.append({
                "codec": r["codec"],
                "width": int(r["width"]), "height": int(r["height"]),
                "crf": int(r["crf"]), "preset": args.val_preset,
                "gop": gop, "scale": r["scale"],
            })

    existing_val = set()
    if args.final_csv.exists():
        with open(args.final_csv) as f:
            for row in csv.DictReader(f):
                k = (row["codec"], row["scale"], row["crf"],
                     str(row["preset"]), row["gop"])
                existing_val.add(k)
        csv_mode = "a"
    else:
        csv_mode = "w"

    val_todo = [c for c in val_configs
                if (c["codec"], c["scale"], str(c["crf"]),
                    str(c["preset"]), str(c["gop"])) not in existing_val]

    print(f"  Unique base configs: {len(seen_base)}")
    print(f"  x {len(validation_gops)} GOPs = {len(val_configs)} total")
    print(f"  Already done: {len(existing_val)}")
    print(f"  Remaining   : {len(val_todo)}")

    val_fields = [
        "codec", "scale", "width", "height", "crf", "preset", "gop",
        "compressed_size", "archive_size", "rate",
        "segnet_dist", "posenet_dist",
        "segnet_score", "posenet_score", "rate_score", "total_score",
        "encode_time", "decode_time", "upscale_time", "inference_time",
        "total_time", "status",
    ]

    tmp = ROOT / "_val_tmp"
    completed = 0
    times_hist = []
    best_score = 999.0

    with open(args.final_csv, csv_mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=val_fields)
        if csv_mode == "w":
            writer.writeheader()

        for cfg in val_todo:
            t0_all = time.time()
            row = {k: cfg.get(k, "") for k in val_fields}

            safe_rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)
            out_path = tmp / "0.mkv"

            ok, enc_t = encode_video(
                INPUT_VIDEO, out_path, cfg["codec"],
                cfg["crf"], cfg["preset"], cfg["gop"],
                cfg["width"], cfg["height"],
            )
            row["encode_time"] = f"{enc_t:.2f}"
            if not ok:
                row["status"] = "encode_failed"
                row["total_time"] = f"{time.time()-t0_all:.2f}"
                writer.writerow(row)
                fh.flush()
                continue

            comp_sz = out_path.stat().st_size

            zip_path = tmp / "archive.zip"
            with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(out_path, "0.mkv")
            archive_sz = zip_path.stat().st_size

            try:
                res = full_eval.evaluate_compressed(str(out_path), archive_sz)
            except Exception as e:
                row["status"] = f"eval_failed: {str(e)[:80]}"
                row["total_time"] = f"{time.time()-t0_all:.2f}"
                writer.writerow(row)
                fh.flush()
                continue

            total_t = time.time() - t0_all
            row.update({
                "compressed_size": comp_sz,
                "archive_size":    archive_sz,
                "rate":           f"{res['rate']:.8f}",
                "segnet_dist":    f"{res['segnet_dist']:.8f}",
                "posenet_dist":   f"{res['posenet_dist']:.8f}",
                "segnet_score":   f"{100*res['segnet_dist']:.4f}",
                "posenet_score":  f"{math.sqrt(10*res['posenet_dist']):.4f}",
                "rate_score":     f"{25*res['rate']:.4f}",
                "total_score":    f"{res['score']:.4f}",
                "decode_time":    f"{res['timings']['decode']:.2f}",
                "upscale_time":   f"{res['timings']['upscale']:.2f}",
                "inference_time": f"{res['timings']['inference']:.2f}",
                "total_time":     f"{total_t:.2f}",
                "status":         "ok",
            })
            writer.writerow(row)
            fh.flush()

            completed += 1
            times_hist.append(total_t)
            if res['score'] < best_score:
                best_score = res['score']

            remaining = len(val_todo) - completed
            avg = sum(times_hist[-10:]) / len(times_hist[-10:])
            eta = remaining * avg / 60

            print(
                f"  [{completed:>3d}/{len(val_todo)}] "
                f"{cfg['scale']:>7s} crf={cfg['crf']:>2} p={cfg['preset']} gop={cfg['gop']:>3}  "
                f"score={res['score']:.4f}  best={best_score:.4f}  "
                f"{total_t:.0f}s  ETA {eta:.0f}m"
            )

    safe_rmtree(tmp)

    # ===================== FINAL RESULTS =====================
    val_rows = []
    if args.final_csv.exists():
        with open(args.final_csv) as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    val_rows.append(r)
    val_rows.sort(key=lambda r: float(r["total_score"]))

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — validated on full 60s video")
    print(f"{'='*60}")
    for i, r in enumerate(val_rows[:20]):
        print(
            f"  {i+1:3d}. {float(r['total_score']):7.4f}  "
            f"{r['codec']} {r['scale']} crf={r['crf']} p={r['preset']} gop={r['gop']}  "
            f"seg={r['segnet_score']} pose={r['posenet_score']} rate={r['rate_score']}  "
            f"archive={r['archive_size']}"
        )

    if val_rows:
        best = val_rows[0]
        bs = float(best['total_score'])
        print(f"\n  BEST: {bs:.4f}  {best['codec']} {best['scale']} "
              f"crf={best['crf']} gop={best['gop']}")
        if bs < 2.90:
            print(f"  >>> BEATS LEADER (2.90) by {2.90 - bs:.4f} <<<")


if __name__ == "__main__":
    main()
