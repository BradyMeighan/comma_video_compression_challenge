#!/usr/bin/env python
"""
Automated grid search for optimal codec parameters.
Runs full evaluation pipeline (compress -> inflate -> evaluate) for each configuration.
Outputs results to grid_search_results.csv.

Uses in-process GPU-accelerated evaluation (loads models once, reuses across configs).
Falls back to CPU if no GPU is available.

Usage:
    python grid_search.py --device cuda --quick --codec svtav1
    python grid_search.py --device cpu --quick
"""

import os
import sys
import csv
import time
import math
import shutil
import subprocess
import argparse
import zipfile
from pathlib import Path
from itertools import product

os.environ["PYTHONUTF8"] = "1"

# Ensure the full ffmpeg build (with libsvtav1) is on PATH
_FFMPEG_FULL = Path(os.path.expanduser(
    "~/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
    "ffmpeg-8.1-full_build/bin"
))
if _FFMPEG_FULL.is_dir() and str(_FFMPEG_FULL) not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(_FFMPEG_FULL) + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
VIDEO_NAMES_FILE = ROOT / "public_test_video_names.txt"
INPUT_VIDEO = VIDEOS_DIR / "0.mkv"
SUBMISSION_DIR = ROOT / "submissions" / "phase1"
ARCHIVE_DIR = SUBMISSION_DIR / "archive"
INFLATED_DIR = SUBMISSION_DIR / "inflated"
RESULTS_CSV = ROOT / "grid_search_results.csv"

ORIGINAL_SIZE = 37_545_489
W, H, C = 1164, 874, 3
FRAME_BYTES = H * W * C
NUM_FRAMES = 1200
SEQ_LEN = 2


def run_cmd(cmd, timeout=3600):
    """Run a shell command and return (returncode, stdout, stderr)."""
    abbrev = " ".join(str(c) for c in cmd[:8])
    print(f"  CMD: {abbrev}...")
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[:500]}")
    return result.returncode, result.stdout, result.stderr


def compress_svtav1(input_path, output_path, crf, preset, gop, width, height):
    """Compress with SVT-AV1."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-r", "20", "-fflags", "+genpts", "-i", str(input_path),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libsvtav1", "-preset", str(preset), "-crf", str(crf),
        "-g", str(gop),
        "-svtav1-params", "enable-overlays=1:film-grain=0:fast-decode=0",
        "-r", "20", str(output_path),
    ]
    return run_cmd(cmd)


def compress_x265(input_path, output_path, crf, preset, gop, width, height):
    """Compress with x265."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-r", "20", "-fflags", "+genpts", "-i", str(input_path),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx265", "-preset", str(preset), "-crf", str(crf),
        "-g", str(gop), "-bf", "3",
        "-x265-params",
        f"keyint={gop}:min-keyint={min(gop, 60)}:scenecut=40:bframes=3:"
        f"b-adapt=2:ref=5:me=umh:subme=7:rd=6:aq-mode=3:log-level=warning",
        "-r", "20", str(output_path),
    ]
    return run_cmd(cmd)


def create_archive_zip(archive_dir, zip_path):
    """Create archive.zip from the archive directory."""
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(archive_dir.rglob('*')):
            if f.is_file():
                zf.write(f, f.relative_to(archive_dir))


def inflate_video(archive_dir, inflated_dir):
    """Run inflation: decode + upscale compressed video to raw."""
    inflated_dir.mkdir(parents=True, exist_ok=True)
    compressed_files = list(archive_dir.glob("0.*"))
    if not compressed_files:
        return False, "No compressed file found in archive"

    src = compressed_files[0]
    dst = inflated_dir / "0.raw"

    cmd = [sys.executable, "-m", "submissions.phase1.inflate", str(src), str(dst)]
    rc, stdout, stderr = run_cmd(cmd, timeout=600)
    return rc == 0, stderr if rc != 0 else stdout


def safe_rmtree(path):
    """rmtree with retry for Windows file locking."""
    import gc
    for attempt in range(3):
        try:
            if path.exists():
                shutil.rmtree(path)
            return
        except PermissionError:
            gc.collect()
            time.sleep(1)


class InProcessEvaluator:
    """
    GPU-accelerated evaluator that loads models once and reuses them.
    Avoids the overhead of spawning evaluate.py per config and bypasses
    the DALI dependency issue on Windows.
    """

    def __init__(self, device, batch_size=16):
        from modules import DistortionNet, segnet_sd_path, posenet_sd_path
        from frame_utils import camera_size, seq_len

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.seq_len = seq_len

        print(f"Loading distortion networks on {self.device}...")
        self.net = DistortionNet().eval().to(self.device)
        self.net.load_state_dicts(str(posenet_sd_path), str(segnet_sd_path), self.device)

        self._uncompressed_size = sum(
            f.stat().st_size for f in VIDEOS_DIR.rglob('*') if f.is_file()
        )
        self._load_ground_truth()
        print(f"Evaluator ready. GT loaded: {self.gt_frames.shape[0]} frames.")

    def _load_ground_truth(self):
        """Load and cache ground truth video frames (decoded via PyAV)."""
        import av
        from frame_utils import yuv420_to_rgb

        gt_path = str(VIDEOS_DIR / "0.mkv")
        container = av.open(gt_path)
        stream = container.streams.video[0]
        frames = []
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            frames.append(t)
        container.close()
        self.gt_frames = torch.stack(frames)  # (1200, 874, 1164, 3) uint8

    def evaluate(self, raw_path, archive_zip_path):
        """
        Run evaluation on an inflated .raw file.
        Returns dict with posenet_dist, segnet_dist, rate, score.
        """
        import gc

        file_size = os.path.getsize(raw_path)
        n_frames = file_size // FRAME_BYTES
        assert n_frames == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {n_frames}"

        compressed_size = os.path.getsize(archive_zip_path)
        rate = compressed_size / self._uncompressed_size

        posenet_sum = torch.zeros([], device=self.device)
        segnet_sum = torch.zeros([], device=self.device)
        total_samples = 0

        n_sequences = n_frames // self.seq_len

        # Read raw file via mmap, process in batches, close promptly
        mm = np.memmap(raw_path, dtype=np.uint8, mode='r', shape=(n_frames, H, W, C))
        with torch.inference_mode():
            for start in range(0, n_sequences, self.batch_size):
                end = min(start + self.batch_size, n_sequences)
                batch_size_actual = end - start

                gt_batch = []
                comp_batch = []
                for s in range(start, end):
                    f0 = s * self.seq_len
                    f1 = f0 + self.seq_len
                    gt_batch.append(self.gt_frames[f0:f1])
                    comp_batch.append(torch.from_numpy(np.array(mm[f0:f1])))

                batch_gt = torch.stack(gt_batch).to(self.device)
                batch_comp = torch.stack(comp_batch).to(self.device)

                p_dist, s_dist = self.net.compute_distortion(batch_gt, batch_comp)
                posenet_sum += p_dist.sum()
                segnet_sum += s_dist.sum()
                total_samples += batch_size_actual

        del mm
        gc.collect()

        posenet_dist = (posenet_sum / total_samples).item()
        segnet_dist = (segnet_sum / total_samples).item()
        score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate

        return {
            "posenet_dist": posenet_dist,
            "segnet_dist": segnet_dist,
            "rate": rate,
            "score": score,
            "archive_size": compressed_size,
        }


def get_resolution(scale):
    """Compute even-aligned resolution from scale factor or explicit tuple."""
    if isinstance(scale, tuple):
        return scale
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    return w, h


def scale_label(scale):
    if isinstance(scale, tuple):
        return f"{scale[0]}x{scale[1]}"
    return str(scale)


def check_encoder_available(encoder_name):
    """Check if an ffmpeg encoder is available."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        return encoder_name in result.stdout
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Grid search over codec parameters")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced grid for quick testing")
    parser.add_argument("--codec", default="svtav1", choices=["svtav1", "x265", "all"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: grid_search_results.csv)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    results_csv = args.output or RESULTS_CSV

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    has_svtav1 = check_encoder_available("libsvtav1")
    has_x265 = check_encoder_available("libx265")

    print(f"Available encoders: svtav1={has_svtav1}, x265={has_x265}")
    print(f"Device: {args.device}")

    if not INPUT_VIDEO.exists():
        print(f"ERROR: Input video not found: {INPUT_VIDEO}")
        sys.exit(1)

    evaluator = InProcessEvaluator(args.device, batch_size=args.batch_size)

    # --- Define search grid ---
    # 45% scale = 522x392 (leader's setting, for comparison)
    if args.quick:
        scales = [(512, 384), (522, 392), (480, 360), (440, 330)]
        svtav1_crfs = [28, 30, 32, 35]
        svtav1_presets = [0]
        gops = [180, 300]
        x265_crfs = [28, 32]
        x265_presets = ["veryslow"]
    else:
        scales = [(512, 384), (522, 392), (480, 360), (440, 330)]
        svtav1_crfs = [25, 27, 29, 30, 31, 32, 33, 35, 38, 40]
        svtav1_presets = [0, 2, 4]
        gops = [120, 180, 300, 600]
        x265_crfs = [25, 28, 30, 32, 35]
        x265_presets = ["slow", "veryslow"]

    configs = []

    if has_svtav1 and args.codec in ("all", "svtav1"):
        for scale, crf, preset, gop in product(scales, svtav1_crfs, svtav1_presets, gops):
            w, h = get_resolution(scale)
            configs.append({
                "codec": "svtav1",
                "crf": crf,
                "preset": preset,
                "gop": gop,
                "width": w,
                "height": h,
                "scale": scale_label(scale),
            })

    if has_x265 and args.codec in ("all", "x265"):
        for scale, crf, preset, gop in product(scales, x265_crfs, x265_presets, gops):
            w, h = get_resolution(scale)
            configs.append({
                "codec": "x265",
                "crf": crf,
                "preset": str(preset),
                "gop": gop,
                "width": w,
                "height": h,
                "scale": scale_label(scale),
            })

    print(f"\nTotal configurations to test: {len(configs)}")
    print(f"Results will be written to: {results_csv}")

    fieldnames = [
        "codec", "scale", "width", "height", "crf", "preset", "gop",
        "archive_size_bytes", "rate", "segnet_dist", "posenet_dist",
        "segnet_score", "posenet_score", "rate_score", "total_score",
        "elapsed_seconds", "status",
    ]

    # Resume support: load existing results
    existing = set()
    if results_csv.exists():
        with open(results_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["codec"], row["scale"], row["crf"],
                       str(row["preset"]), row["gop"])
                existing.add(key)
        print(f"Resuming: {len(existing)} configurations already completed")
        csv_mode = "a"
    else:
        csv_mode = "w"

    with open(results_csv, csv_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        for i, cfg in enumerate(configs):
            key = (cfg["codec"], cfg["scale"], str(cfg["crf"]),
                   str(cfg["preset"]), str(cfg["gop"]))
            if key in existing:
                continue

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(configs)}] {cfg['codec']} scale={cfg['scale']} "
                  f"crf={cfg['crf']} preset={cfg['preset']} gop={cfg['gop']}")
            print(f"{'='*60}")

            t0 = time.time()
            row = {k: cfg.get(k, "") for k in fieldnames}

            # Clean up previous artifacts
            safe_rmtree(ARCHIVE_DIR)
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            safe_rmtree(INFLATED_DIR)

            output_path = ARCHIVE_DIR / "0.mkv"

            # --- Compress ---
            if cfg["codec"] == "svtav1":
                rc, _, _ = compress_svtav1(
                    INPUT_VIDEO, output_path,
                    cfg["crf"], cfg["preset"], cfg["gop"],
                    cfg["width"], cfg["height"],
                )
            elif cfg["codec"] == "x265":
                rc, _, _ = compress_x265(
                    INPUT_VIDEO, output_path,
                    cfg["crf"], cfg["preset"], cfg["gop"],
                    cfg["width"], cfg["height"],
                )
            else:
                row["status"] = "unknown_codec"
                writer.writerow(row)
                csvfile.flush()
                continue

            if rc != 0:
                row["status"] = "compress_failed"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            compressed_size = output_path.stat().st_size
            print(f"  Compressed video: {compressed_size:,} bytes")

            # --- Create archive.zip ---
            zip_path = SUBMISSION_DIR / "archive.zip"
            create_archive_zip(ARCHIVE_DIR, zip_path)
            archive_size = zip_path.stat().st_size
            print(f"  Archive.zip: {archive_size:,} bytes")

            # --- Inflate ---
            ok, msg = inflate_video(ARCHIVE_DIR, INFLATED_DIR)
            if not ok:
                row["status"] = f"inflate_failed: {msg[:100]}"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            raw_path = INFLATED_DIR / "0.raw"
            if not raw_path.exists():
                row["status"] = "inflate_no_raw"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            expected_raw = NUM_FRAMES * FRAME_BYTES
            actual_raw = raw_path.stat().st_size
            if actual_raw != expected_raw:
                row["status"] = f"raw_size_mismatch: {actual_raw} vs {expected_raw}"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            # --- Evaluate (in-process, GPU-accelerated) ---
            try:
                results = evaluator.evaluate(str(raw_path), str(zip_path))
            except Exception as e:
                row["status"] = f"eval_failed: {str(e)[:100]}"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            elapsed = time.time() - t0

            row["archive_size_bytes"] = archive_size
            row["rate"] = f"{results['rate']:.8f}"
            row["segnet_dist"] = f"{results['segnet_dist']:.8f}"
            row["posenet_dist"] = f"{results['posenet_dist']:.8f}"
            row["segnet_score"] = f"{100 * results['segnet_dist']:.4f}"
            row["posenet_score"] = f"{math.sqrt(10 * results['posenet_dist']):.4f}"
            row["rate_score"] = f"{25 * results['rate']:.4f}"
            row["total_score"] = f"{results['score']:.4f}"
            row["elapsed_seconds"] = f"{elapsed:.1f}"
            row["status"] = "ok"

            writer.writerow(row)
            csvfile.flush()

            print(f"\n  >>> Score: {results['score']:.4f} <<<")
            print(f"      SegNet:  {100*results['segnet_dist']:.3f}  "
                  f"(dist={results['segnet_dist']:.6f})")
            print(f"      PoseNet: {math.sqrt(10*results['posenet_dist']):.3f}  "
                  f"(dist={results['posenet_dist']:.6f})")
            print(f"      Rate:    {25*results['rate']:.3f}  "
                  f"(size={archive_size:,} bytes)")
            print(f"      Time:    {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"Grid search complete. Results in {results_csv}")
    print(f"{'='*60}")

    # Print top 10 results
    if results_csv.exists():
        rows = []
        with open(results_csv) as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    rows.append(row)
        rows.sort(key=lambda r: float(r["total_score"]))
        print(f"\n=== Top {min(10, len(rows))} Configurations ===")
        for i, r in enumerate(rows[:10]):
            print(f"  {i+1}. score={r['total_score']}  "
                  f"codec={r['codec']} scale={r['scale']} "
                  f"crf={r['crf']} preset={r['preset']} gop={r['gop']}  "
                  f"size={r['archive_size_bytes']}  "
                  f"seg={r['segnet_score']} pose={r['posenet_score']} "
                  f"rate={r['rate_score']}")


if __name__ == "__main__":
    main()
