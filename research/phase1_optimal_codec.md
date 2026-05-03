# Phase 1: Optimal Traditional Codec Submission

## Goal

Beat the current leader score of **2.90** (PR #17 by EthanYangTW). Target score range: **2.3 -- 2.7**.

This phase uses only traditional video codec optimization -- no neural network decoders, no learned compression. It is the safest, fastest path to a competitive score because the entire pipeline is well-understood and every parameter change can be evaluated in a single run.

---

## Background

### Scoring Formula

The score is computed as (lower is better):

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * rate
```

Source: `evaluate.py:92`

| Component | Formula | Weight | Sensitivity |
|-----------|---------|--------|-------------|
| SegNet distortion | Fraction of pixels where argmax class flips | 100x | **Dominant.** 1% pixel disagreement = 1.0 points. |
| PoseNet distortion | MSE of first 6 of 12 pose output dims | sqrt(10x) | Diminishing returns under square root. |
| Rate | `archive.zip size / 37,545,489` | 25x | Linear but small range. Going from 3% to 1.5% saves only 0.375 points. |

### Current Leader Breakdown (PR #17, Score 2.90)

| Component | Raw Value | Score Contribution |
|-----------|-----------|--------------------|
| SegNet distortion | 0.00522 | 100 * 0.00522 = **0.522** |
| PoseNet distortion | 0.332 | sqrt(10 * 0.332) = sqrt(3.32) = **1.822** |
| Rate | 835,000 / 37,545,489 = 0.02224 | 25 * 0.02224 = **0.556** |
| **Total** | | **2.90** |

### Where Points Can Be Saved

To get from 2.90 to 2.50 we need to save ~0.40 points. Possible sources:

1. **SegNet (0.522)**: Reducing from 0.00522 to 0.00400 saves 0.122 points. This is the highest-leverage term.
2. **Rate (0.556)**: Reducing archive from 835 KB to 500 KB saves 25 * (335000/37545489) = 0.223 points.
3. **PoseNet (1.822)**: Reducing from 0.332 to 0.250 saves sqrt(3.32) - sqrt(2.50) = 1.822 - 1.581 = 0.241 points.

All three are needed. The strategy is: use a better codec (VVC or better-tuned AV1) to reduce rate while maintaining or improving distortion, and use preprocessing to reduce entropy in SegNet-unimportant regions.

### What the Evaluation Pipeline Does

The full path from compression to score:

```
compress.sh
  -> encodes videos/0.mkv into archive/ directory
  -> zips archive/ into archive.zip

evaluate.sh
  -> unzips archive.zip into archive/
  -> runs inflate.sh <archive_dir> <inflated_dir> <video_names_file>
  -> inflate.sh calls inflate.py per video
  -> inflate.py decodes compressed video, upscales to 1164x874, writes .raw file
  -> evaluate.py loads .raw via TensorVideoDataset
  -> pairs frames into (B, 2, 874, 1164, 3) batches
  -> DistortionNet.preprocess_input:
       - Rearranges to (B, 2, 3, 874, 1164) float
       - PoseNet path: bilinear resize both frames to 512x384, rgb_to_yuv6, stack to (B, 12, 192, 256)
       - SegNet path: take LAST frame only (index -1 = odd frame), bilinear resize to (B, 3, 384, 512)
  -> Runs both networks, computes distortion
  -> Averages over all 600 samples
  -> score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
```

### Encode Resolution Analysis

Both networks evaluate at `segnet_model_input_size = (512, 384)` (width x height). The preprocessing chain is:

```
Encoded frame (e.g., 522x392)
  -> inflate.py: bicubic upscale to 1164x874
  -> evaluate.py: bilinear downscale to 512x384
```

The baseline encodes at 522x392 (45% of 1164x874). The current leader (PR #17) also uses 45% scale. PR #10 uses 35% scale.

Key insight: encoding at exactly 512x384 might minimize round-trip resampling error, since the upscale to 1164x874 followed by bilinear downscale to 512x384 is not lossless. However, encoding at a smaller resolution reduces bitrate, so there is a tradeoff.

### Available Codecs

| Codec | ffmpeg encoder | Expected quality | Notes |
|-------|---------------|-----------------|-------|
| H.265/HEVC | libx265 | Good | Used by baseline. All-intra mode is wasteful. |
| AV1 | libsvtav1 | Better | Used by leader. 20-30% better than HEVC at same quality. |
| VVC/H.266 | libvvenc | Best | 20-30% better than AV1 at low bitrates. May not be in default ffmpeg. |
| AV1 | libaom-av1 | Same as SVT | Reference encoder. Much slower than SVT. |

### Critical Observations from Baseline vs. Leader

The baseline (score 4.39) uses `libx265 -preset ultrafast -crf 30 -g 1` (all-intra, every frame is a keyframe). This wastes massive bitrate because it cannot exploit temporal redundancy between frames.

The leader (score 2.90) uses `libsvtav1 -preset 0 -crf 32 -g 180`. By using inter-prediction with a GOP of 180 frames, it dramatically reduces rate while preserving quality.

The key improvements from baseline to leader:
- Codec: x265 -> SVT-AV1 (better compression efficiency)
- Preset: ultrafast -> 0 (much better rate-distortion optimization)
- GOP: 1 (all-intra) -> 180 (inter-prediction)
- CRF: 30 -> 32 (slightly more aggressive quantization, acceptable because inter-prediction compensates)

---

## Task 1: Encode Resolution Optimization

### Objective

Determine the optimal encoding resolution that minimizes the final score, considering the full resample chain: `encode_res -> bicubic up to 1164x874 -> bilinear down to 512x384`.

### Test Resolutions

| Name | Resolution | Scale Factor | Notes |
|------|-----------|-------------|-------|
| exact_model | 512x384 | ~44% x ~44% | Exact model input size. May minimize round-trip error. |
| scale_45 | 522x392 | 45% | Baseline/leader setting. |
| scale_40 | 464x348 | ~40% | Smaller, lower bitrate. |
| scale_35 | 406x306 | ~35% | Used by PR #10 (score 2.99). |
| scale_30 | 348x262 | ~30% | Very aggressive downscale. |
| scale_50 | 582x436 | ~50% | Slightly larger than baseline. |

Note: all resolutions must be even (required for YUV420). The ffmpeg scale filter `trunc(dim/2)*2` ensures this.

### Procedure

For each resolution, using the leader's codec settings (SVT-AV1, preset 0, CRF 32, GOP 180) as a baseline:

1. Compress with `ffmpeg`:
```bash
SCALE=0.45  # vary this
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i videos/0.mkv \
  -vf "scale=trunc(iw*${SCALE}/2)*2:trunc(ih*${SCALE}/2)*2:flags=lanczos" \
  -c:v libsvtav1 -preset 0 -crf 32 \
  -g 180 -svtav1-params "enable-overlays=1" \
  -r 20 output.mkv
```

For the exact 512x384 resolution:
```bash
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i videos/0.mkv \
  -vf "scale=512:384:flags=lanczos" \
  -c:v libsvtav1 -preset 0 -crf 32 \
  -g 180 -svtav1-params "enable-overlays=1" \
  -r 20 output.mkv
```

2. Inflate (decode + upscale to 1164x874):
```bash
cd C:\Users\modce\Documents\comma_video_compression_challenge
python -m submissions.phase1.inflate output.mkv output.raw
```

3. Run evaluation:
```bash
bash evaluate.sh --submission-dir ./submissions/phase1 --device cpu
```

4. Record: archive size, SegNet distortion, PoseNet distortion, total score.

### What to Watch For

- Encoding at 512x384 avoids one resampling step in the upscale->downscale chain, but the bicubic upscale to 1164x874 introduces its own artifacts that the bilinear downscale may not undo cleanly.
- Smaller resolutions reduce rate dramatically but increase distortion. The question is whether the rate savings outweigh the distortion increase.
- The SegNet argmax is very stable for large uniform regions (sky, road), so moderate quality loss in those regions is free. But boundaries between classes (road edge, lane markings, vehicles) are extremely sensitive.

---

## Task 2: Codec and Parameter Grid Search

### Objective

Systematically search codec parameters to find the configuration that minimizes the total score. The search must use the FULL evaluation pipeline (SegNet + PoseNet + rate), not PSNR or SSIM, because those metrics do not correlate well with the task-specific scoring.

### Parameters to Search

#### SVT-AV1 (libsvtav1)

| Parameter | Values to Test | Notes |
|-----------|---------------|-------|
| CRF | 25, 27, 29, 30, 31, 32, 33, 35, 38, 40 | Quality vs. rate tradeoff |
| Preset | 0, 1, 2, 3, 4, 6 | Speed vs. efficiency. 0 is best quality. |
| GOP (keyint) | 60, 120, 180, 300, 600, 1200 | Longer GOP = better compression, but more error propagation |
| Scale | 0.35, 0.40, 0.44, 0.45, 0.50, 512x384 | Resolution (from Task 1) |

SVT-AV1 ffmpeg command template:
```bash
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i "$INPUT" \
  -vf "scale=${WIDTH}:${HEIGHT}:flags=lanczos" \
  -c:v libsvtav1 -preset ${PRESET} -crf ${CRF} \
  -g ${GOP} \
  -svtav1-params "enable-overlays=1:film-grain=0:fast-decode=0" \
  -r 20 "$OUTPUT"
```

Important SVT-AV1 parameters:
- `enable-overlays=1`: Enables overlay frames for better compression (requires preset <= 4)
- `film-grain=0`: No film grain synthesis (we want clean output)
- `fast-decode=0`: Allow all decoder optimizations
- `tile-columns` / `tile-rows`: Can be set for parallelism but may hurt efficiency

#### VVenC/VVC (libvvenc)

VVC (H.266) is the successor to HEVC and AV1, offering 20-30% better compression at low bitrates. If available in ffmpeg:

```bash
# Check if libvvenc is available
ffmpeg -encoders 2>/dev/null | grep vvenc

# If available:
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i "$INPUT" \
  -vf "scale=${WIDTH}:${HEIGHT}:flags=lanczos" \
  -c:v libvvenc -qp ${QP} -preset ${PRESET} \
  -g ${GOP} -period ${GOP} \
  -r 20 "$OUTPUT"
```

VVC parameters to test:
| Parameter | Values |
|-----------|--------|
| QP | 25, 28, 30, 32, 35, 38 |
| Preset | slow, slower, veryslow (or 0-3) |
| GOP | 60, 120, 180, 600 |

Note: VVC uses QP (not CRF) in most implementations. QP 32 in VVC roughly corresponds to CRF 30 in AV1.

If libvvenc is not available in the system ffmpeg, install it:
```bash
# On Ubuntu:
sudo apt-get install -y vvenc
# Or build ffmpeg with --enable-libvvenc

# Alternative: use vvencapp directly
vvencapp --input input.yuv --size ${WIDTH}x${HEIGHT} --fps 20 \
  --qp ${QP} --preset slower --threads 4 \
  --output output.266

# Then mux into MP4:
ffmpeg -i output.266 -c copy output.mp4
```

For VVC decoding at inflate time, use vvdec or ffmpeg with libvvdec:
```bash
# Check decoder availability
ffmpeg -decoders 2>/dev/null | grep vvc
```

#### x265 Comparison (as reference)

```bash
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i "$INPUT" \
  -vf "scale=${WIDTH}:${HEIGHT}:flags=lanczos" \
  -c:v libx265 -preset veryslow -crf ${CRF} \
  -g ${GOP} -bf 3 \
  -x265-params "keyint=${GOP}:min-keyint=${GOP}:scenecut=0:bframes=3:b-adapt=2:ref=5:me=umh:subme=7:rd=6:aq-mode=3:log-level=warning" \
  -r 20 "$OUTPUT"
```

x265 parameters to test:
| Parameter | Values |
|-----------|--------|
| CRF | 25, 28, 30, 32, 35 |
| Preset | slow, slower, veryslow |
| GOP | 60, 120, 180, 600 |

### Automated Grid Search Script

Create `grid_search.py` at the repo root:

```python
#!/usr/bin/env python
"""
Automated grid search for optimal codec parameters.
Runs full evaluation pipeline (compress -> inflate -> evaluate) for each configuration.
Outputs results to grid_search_results.csv.

Usage:
    python grid_search.py [--device cpu|cuda|mps] [--quick]
"""

import os
import sys
import csv
import time
import shutil
import subprocess
import argparse
import zipfile
from pathlib import Path
from itertools import product

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
VIDEO_NAMES_FILE = ROOT / "public_test_video_names.txt"
INPUT_VIDEO = VIDEOS_DIR / "0.mkv"
SUBMISSION_DIR = ROOT / "submissions" / "phase1"
ARCHIVE_DIR = SUBMISSION_DIR / "archive"
INFLATED_DIR = SUBMISSION_DIR / "inflated"
RESULTS_CSV = ROOT / "grid_search_results.csv"

# Original file size for rate calculation
ORIGINAL_SIZE = 37_545_489


def run_cmd(cmd, timeout=1800):
    """Run a shell command and return (returncode, stdout, stderr)."""
    print(f"  CMD: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
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
        "-svtav1-params", "enable-overlays=1:film-grain=0",
        "-r", "20", str(output_path),
    ]
    return run_cmd(cmd)


def compress_x265(input_path, output_path, crf, preset, gop, width, height):
    """Compress with x265."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-r", "20", "-fflags", "+genpts", "-i", str(input_path),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libx265", "-preset", preset, "-crf", str(crf),
        "-g", str(gop), "-bf", "3",
        "-x265-params",
        f"keyint={gop}:min-keyint={min(gop, 60)}:scenecut=40:bframes=3:"
        f"b-adapt=2:ref=5:me=umh:subme=7:rd=6:aq-mode=3:log-level=warning",
        "-r", "20", str(output_path),
    ]
    return run_cmd(cmd)


def compress_vvenc(input_path, output_path, qp, preset, gop, width, height):
    """Compress with VVenC (if available)."""
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "warning",
        "-r", "20", "-fflags", "+genpts", "-i", str(input_path),
        "-vf", f"scale={width}:{height}:flags=lanczos",
        "-c:v", "libvvenc", "-qp", str(qp), "-preset", str(preset),
        "-g", str(gop), "-period", str(gop),
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
    # Find the compressed video in archive_dir
    compressed_files = list(archive_dir.glob("0.*"))
    if not compressed_files:
        return False, "No compressed file found in archive"

    src = compressed_files[0]
    dst = inflated_dir / "0.raw"

    cmd = [
        sys.executable, "-m", "submissions.phase1.inflate",
        str(src), str(dst),
    ]
    rc, stdout, stderr = run_cmd(cmd)
    return rc == 0, stderr if rc != 0 else stdout


def run_evaluation(submission_dir, device="cpu"):
    """Run evaluate.py and parse results."""
    cmd = [
        sys.executable, str(ROOT / "evaluate.py"),
        "--submission-dir", str(submission_dir),
        "--uncompressed-dir", str(VIDEOS_DIR),
        "--report", str(submission_dir / "report.txt"),
        "--video-names-file", str(VIDEO_NAMES_FILE),
        "--device", device,
    ]
    rc, stdout, stderr = run_cmd(cmd, timeout=1800)
    if rc != 0:
        return None

    # Parse report.txt
    report_path = submission_dir / "report.txt"
    if not report_path.exists():
        return None

    results = {}
    with open(report_path) as f:
        for line in f:
            line = line.strip()
            if "PoseNet Distortion" in line:
                results["posenet_dist"] = float(line.split(":")[-1].strip())
            elif "SegNet Distortion" in line:
                results["segnet_dist"] = float(line.split(":")[-1].strip())
            elif "Submission file size" in line:
                results["archive_size"] = int(line.split(":")[-1].strip().replace(",", "").split()[0])
            elif "Compression Rate" in line:
                results["rate"] = float(line.split(":")[-1].strip())
            elif "Final score" in line:
                # Extract the numeric score at the end
                parts = line.split("=")
                results["score"] = float(parts[-1].strip())
    return results


def get_resolution(scale):
    """Compute even-aligned resolution from scale factor."""
    if isinstance(scale, tuple):
        return scale  # explicit (width, height)
    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2
    return w, h


def check_encoder_available(encoder_name):
    """Check if an ffmpeg encoder is available."""
    result = subprocess.run(
        ["ffmpeg", "-encoders"],
        capture_output=True, text=True, timeout=10
    )
    return encoder_name in result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true", help="Run reduced grid for quick testing")
    parser.add_argument("--codec", default="all", choices=["all", "svtav1", "x265", "vvenc"])
    args = parser.parse_args()

    # Ensure submission directory structure exists
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    # Check available encoders
    has_svtav1 = check_encoder_available("libsvtav1")
    has_vvenc = check_encoder_available("libvvenc")
    has_x265 = check_encoder_available("libx265")

    print(f"Available encoders: svtav1={has_svtav1}, vvenc={has_vvenc}, x265={has_x265}")

    # Define grid
    if args.quick:
        scales = [0.44, 0.40, (512, 384)]
        svtav1_crfs = [30, 32, 35]
        svtav1_presets = [0, 4]
        gops = [180, 600]
        x265_crfs = [28, 32]
        x265_presets = ["veryslow"]
        vvenc_qps = [30, 35]
        vvenc_presets = [0]
    else:
        scales = [0.30, 0.35, 0.40, 0.44, 0.45, 0.50, (512, 384)]
        svtav1_crfs = [25, 27, 29, 30, 31, 32, 33, 35, 38, 40]
        svtav1_presets = [0, 1, 2, 3, 4, 6]
        gops = [60, 120, 180, 300, 600, 1200]
        x265_crfs = [25, 28, 30, 32, 35]
        x265_presets = ["slow", "slower", "veryslow"]
        vvenc_qps = [25, 28, 30, 32, 35, 38]
        vvenc_presets = [0, 1, 2]

    # Build configurations
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
                "scale": f"{scale}" if isinstance(scale, (int, float)) else f"{w}x{h}",
            })

    if has_x265 and args.codec in ("all", "x265"):
        for scale, crf, preset, gop in product(scales, x265_crfs, x265_presets, gops):
            w, h = get_resolution(scale)
            configs.append({
                "codec": "x265",
                "crf": crf,
                "preset": preset,
                "gop": gop,
                "width": w,
                "height": h,
                "scale": f"{scale}" if isinstance(scale, (int, float)) else f"{w}x{h}",
            })

    if has_vvenc and args.codec in ("all", "vvenc"):
        for scale, qp, preset, gop in product(scales, vvenc_qps, vvenc_presets, gops):
            w, h = get_resolution(scale)
            configs.append({
                "codec": "vvenc",
                "crf": qp,  # stored as 'crf' column but is QP for VVC
                "preset": preset,
                "gop": gop,
                "width": w,
                "height": h,
                "scale": f"{scale}" if isinstance(scale, (int, float)) else f"{w}x{h}",
            })

    print(f"\nTotal configurations to test: {len(configs)}")
    print(f"Results will be written to: {RESULTS_CSV}")

    # CSV header
    fieldnames = [
        "codec", "scale", "width", "height", "crf", "preset", "gop",
        "archive_size_bytes", "rate", "segnet_dist", "posenet_dist",
        "segnet_score", "posenet_score", "rate_score", "total_score",
        "elapsed_seconds", "status",
    ]

    # Resume support: load existing results
    existing = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["codec"], row["scale"], row["crf"], row["preset"], row["gop"])
                existing.add(key)
        print(f"Resuming: {len(existing)} configurations already completed")
        csv_mode = "a"
    else:
        csv_mode = "w"

    with open(RESULTS_CSV, csv_mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csv_mode == "w":
            writer.writeheader()

        for i, cfg in enumerate(configs):
            key = (cfg["codec"], cfg["scale"], str(cfg["crf"]), str(cfg["preset"]), str(cfg["gop"]))
            if key in existing:
                continue

            print(f"\n[{i+1}/{len(configs)}] {cfg['codec']} scale={cfg['scale']} "
                  f"crf={cfg['crf']} preset={cfg['preset']} gop={cfg['gop']}")

            t0 = time.time()
            row = {k: cfg.get(k, "") for k in fieldnames}

            # Clean up
            if ARCHIVE_DIR.exists():
                shutil.rmtree(ARCHIVE_DIR)
            ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
            if INFLATED_DIR.exists():
                shutil.rmtree(INFLATED_DIR)

            # Determine output extension based on codec
            ext = ".mkv" if cfg["codec"] in ("svtav1", "x265") else ".mp4"
            output_path = ARCHIVE_DIR / f"0{ext}"

            # Compress
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
            elif cfg["codec"] == "vvenc":
                rc, _, _ = compress_vvenc(
                    INPUT_VIDEO, output_path,
                    cfg["crf"], cfg["preset"], cfg["gop"],
                    cfg["width"], cfg["height"],
                )
            else:
                row["status"] = "unknown_codec"
                writer.writerow(row)
                continue

            if rc != 0:
                row["status"] = "compress_failed"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            # Create archive.zip
            zip_path = SUBMISSION_DIR / "archive.zip"
            create_archive_zip(ARCHIVE_DIR, zip_path)
            archive_size = zip_path.stat().st_size

            # Inflate
            ok, msg = inflate_video(ARCHIVE_DIR, INFLATED_DIR)
            if not ok:
                row["status"] = f"inflate_failed: {msg[:100]}"
                row["elapsed_seconds"] = f"{time.time() - t0:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            # Evaluate
            results = run_evaluation(SUBMISSION_DIR, device=args.device)
            elapsed = time.time() - t0

            if results is None:
                row["status"] = "eval_failed"
                row["elapsed_seconds"] = f"{elapsed:.1f}"
                writer.writerow(row)
                csvfile.flush()
                continue

            # Fill in results
            import math
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

            print(f"  Score: {results['score']:.4f} "
                  f"(seg={100*results['segnet_dist']:.3f} "
                  f"pose={math.sqrt(10*results['posenet_dist']):.3f} "
                  f"rate={25*results['rate']:.3f}) "
                  f"size={archive_size:,} bytes  [{elapsed:.0f}s]")

    print(f"\nGrid search complete. Results in {RESULTS_CSV}")

    # Print top 10 results
    if RESULTS_CSV.exists():
        rows = []
        with open(RESULTS_CSV) as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    rows.append(row)
        rows.sort(key=lambda r: float(r["total_score"]))
        print("\n=== Top 10 Configurations ===")
        for i, r in enumerate(rows[:10]):
            print(f"  {i+1}. score={r['total_score']}  codec={r['codec']} "
                  f"scale={r['scale']} crf={r['crf']} preset={r['preset']} "
                  f"gop={r['gop']} size={r['archive_size_bytes']}")


if __name__ == "__main__":
    main()
```

Save this as `C:\Users\modce\Documents\comma_video_compression_challenge\grid_search.py`.

### Running the Grid Search

Full grid (will take many hours):
```bash
cd C:\Users\modce\Documents\comma_video_compression_challenge
python grid_search.py --device cpu
```

Quick grid (reduced parameter space, faster iteration):
```bash
python grid_search.py --device cpu --quick
```

Single codec:
```bash
python grid_search.py --device cpu --codec svtav1 --quick
```

### Expected Grid Size

- Full SVT-AV1: 7 scales x 10 CRFs x 6 presets x 6 GOPs = 2,520 configurations
- Full x265: 7 x 5 x 3 x 6 = 630 configurations
- Full VVenC: 7 x 6 x 3 x 6 = 756 configurations
- Total full grid: ~3,906 configurations

With quick mode: 3 x 3 x 2 x 2 = 36 (SVT-AV1) + 3 x 2 x 1 x 2 = 12 (x265) + 3 x 2 x 1 x 2 = 12 (VVenC) = ~60 configurations.

Each evaluation takes 2-5 minutes on CPU (mostly the neural network inference), so the quick grid takes about 2-5 hours and the full grid would take days. Strategy: run quick grid first, identify promising regions, then do a focused search.

---

## Task 3: SegNet-Aware Preprocessing

### Objective

Before encoding, identify regions of each frame where the SegNet prediction is stable/uniform, and apply entropy-reducing transformations (blur, flat-color fill) to those regions. This gives the codec more bits for critical boundary regions, reducing overall SegNet distortion at the same or lower bitrate.

### Why This Works

The SegNet computes `argmax(logits)` per pixel over 5 classes. In large uniform regions (sky, road surface), the dominant class has a large margin over alternatives. Blurring or simplifying those regions does not change the argmax, but it reduces pixel-level entropy, which means the codec spends fewer bits on them. The saved bits go to boundary regions where small pixel changes can flip the argmax and cause distortion.

### Implementation

Create `preprocess.py` at `C:\Users\modce\Documents\comma_video_compression_challenge\preprocess.py`:

```python
#!/usr/bin/env python
"""
SegNet-aware preprocessing: identify low-importance regions and reduce their entropy
before encoding, giving the codec more bits for critical boundary areas.

Usage:
    python preprocess.py --input videos/0.mkv --output preprocessed/0.mkv \
        --blur-sigma 3.0 --margin-threshold 2.0

The output is a raw YUV/MKV video ready for encoding with ffmpeg.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, segnet_model_input_size, yuv420_to_rgb
from modules import SegNet, segnet_sd_path
from safetensors.torch import load_file


def load_segnet(device):
    """Load the SegNet model."""
    segnet = SegNet().eval().to(device)
    sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(sd)
    return segnet


def get_segnet_importance_map(segnet, frame_rgb, device):
    """
    Run SegNet on a frame and compute per-pixel importance.

    Returns:
        importance: (H, W) float tensor. High values = boundary/important region.
        argmax_map: (H, W) int tensor. Per-pixel class predictions.
        margin: (H, W) float tensor. Difference between top-1 and top-2 logit.
    """
    # frame_rgb: (H, W, 3) uint8 tensor
    H, W, C = frame_rgb.shape
    model_w, model_h = segnet_model_input_size  # 512, 384

    # Prepare input: (1, 3, 384, 512)
    x = frame_rgb.permute(2, 0, 1).unsqueeze(0).float().to(device)
    x = F.interpolate(x, size=(model_h, model_w), mode='bilinear', align_corners=False)

    with torch.inference_mode():
        logits = segnet(x)  # (1, 5, 384, 512)

    # Compute argmax and margin
    sorted_logits, _ = logits.sort(dim=1, descending=True)
    top1 = sorted_logits[:, 0, :, :]  # (1, 384, 512)
    top2 = sorted_logits[:, 1, :, :]
    margin = (top1 - top2).squeeze(0)  # (384, 512) - large margin = stable prediction

    argmax_map = logits.argmax(dim=1).squeeze(0)  # (384, 512)

    # Upscale importance/margin back to original resolution
    margin_full = F.interpolate(
        margin.unsqueeze(0).unsqueeze(0),
        size=(H, W), mode='bilinear', align_corners=False
    ).squeeze()

    argmax_full = F.interpolate(
        argmax_map.unsqueeze(0).unsqueeze(0).float(),
        size=(H, W), mode='nearest'
    ).squeeze().long()

    return margin_full, argmax_full


def preprocess_frame(frame_rgb, margin, argmax_map, blur_sigma, margin_threshold,
                     blur_mode="gaussian"):
    """
    Apply entropy-reducing preprocessing to low-importance regions.

    Args:
        frame_rgb: (H, W, 3) uint8 tensor
        margin: (H, W) float tensor - segnet class margin
        argmax_map: (H, W) int tensor - segnet argmax per pixel
        blur_sigma: Gaussian blur sigma for low-importance regions
        margin_threshold: Pixels with margin > threshold are considered stable
        blur_mode: "gaussian" or "flat" (flat replaces with mean color per class)

    Returns:
        processed: (H, W, 3) uint8 tensor
    """
    H, W, C = frame_rgb.shape
    frame_float = frame_rgb.float()

    # Mask: True = low importance (stable prediction, high margin)
    stable_mask = margin > margin_threshold  # (H, W)

    if blur_mode == "gaussian":
        # Apply Gaussian blur to the whole frame, then blend
        kernel_size = max(int(blur_sigma * 6) | 1, 3)  # Ensure odd
        frame_chw = frame_float.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        # Create Gaussian kernel
        x = torch.arange(kernel_size, dtype=torch.float32, device=frame_rgb.device) - kernel_size // 2
        gauss_1d = torch.exp(-x.pow(2) / (2 * blur_sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        gauss_2d = gauss_1d.unsqueeze(1) * gauss_1d.unsqueeze(0)
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

        pad = kernel_size // 2
        blurred = F.conv2d(
            F.pad(frame_chw, [pad, pad, pad, pad], mode='reflect'),
            gauss_2d, groups=3
        ).squeeze(0).permute(1, 2, 0)  # (H, W, 3)

        # Blend: use blurred in stable regions, original in boundary regions
        mask_3d = stable_mask.unsqueeze(-1).expand_as(frame_float)
        result = torch.where(mask_3d, blurred, frame_float)

    elif blur_mode == "flat":
        # Replace stable regions with mean color per class
        result = frame_float.clone()
        for cls in range(5):
            cls_mask = (argmax_map == cls) & stable_mask
            if cls_mask.any():
                mean_color = frame_float[cls_mask].mean(dim=0)  # (3,)
                result[cls_mask] = mean_color

    else:
        result = frame_float

    return result.clamp(0, 255).round().to(torch.uint8)


def verify_no_argmax_flip(segnet, original_frame, processed_frame, device):
    """
    Verify that preprocessing did not flip any SegNet argmax decisions.
    Returns the fraction of pixels that flipped.
    """
    model_w, model_h = segnet_model_input_size

    orig_x = original_frame.permute(2, 0, 1).unsqueeze(0).float().to(device)
    proc_x = processed_frame.permute(2, 0, 1).unsqueeze(0).float().to(device)

    orig_x = F.interpolate(orig_x, size=(model_h, model_w), mode='bilinear', align_corners=False)
    proc_x = F.interpolate(proc_x, size=(model_h, model_w), mode='bilinear', align_corners=False)

    with torch.inference_mode():
        orig_logits = segnet(orig_x)
        proc_logits = segnet(proc_x)

    orig_argmax = orig_logits.argmax(dim=1)
    proc_argmax = proc_logits.argmax(dim=1)

    flip_rate = (orig_argmax != proc_argmax).float().mean().item()
    return flip_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory for preprocessed frames")
    parser.add_argument("--blur-sigma", type=float, default=3.0, help="Gaussian blur sigma")
    parser.add_argument("--margin-threshold", type=float, default=2.0,
                        help="Margin threshold for stability (higher = more conservative)")
    parser.add_argument("--blur-mode", default="gaussian", choices=["gaussian", "flat"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify", action="store_true", help="Verify no argmax flips")
    args = parser.parse_args()

    device = torch.device(args.device)
    segnet = load_segnet(device)

    # Decode input video
    import av
    container = av.open(args.input)
    stream = container.streams.video[0]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for frame in container.decode(stream):
        t = yuv420_to_rgb(frame)  # (H, W, 3) uint8 tensor
        frames.append(t)
    container.close()

    print(f"Loaded {len(frames)} frames from {args.input}")
    print(f"Frame size: {frames[0].shape}")
    print(f"Blur sigma: {args.blur_sigma}, margin threshold: {args.margin_threshold}")

    total_flip_rate = 0.0
    processed_frames = []

    for i, frame in enumerate(frames):
        if i % 100 == 0:
            print(f"Processing frame {i}/{len(frames)}...")

        margin, argmax_map = get_segnet_importance_map(segnet, frame, device)
        processed = preprocess_frame(
            frame, margin, argmax_map,
            args.blur_sigma, args.margin_threshold, args.blur_mode
        )

        if args.verify:
            flip_rate = verify_no_argmax_flip(segnet, frame, processed, device)
            total_flip_rate += flip_rate
            if flip_rate > 0:
                print(f"  Frame {i}: flip rate = {flip_rate:.6f}")

        processed_frames.append(processed)

    if args.verify:
        avg_flip = total_flip_rate / len(frames)
        print(f"\nAverage flip rate from preprocessing: {avg_flip:.8f}")
        print(f"This would add {100 * avg_flip:.4f} to the SegNet score component")

    # Write preprocessed frames as raw file for ffmpeg input
    raw_path = output_dir / "preprocessed.raw"
    H, W, C = frames[0].shape
    with open(raw_path, 'wb') as f:
        for frame in processed_frames:
            f.write(frame.contiguous().numpy().tobytes())

    print(f"\nWrote {len(processed_frames)} preprocessed frames to {raw_path}")
    print(f"Frame dimensions: {W}x{H}")
    print(f"\nTo encode with ffmpeg:")
    print(f"  ffmpeg -f rawvideo -pix_fmt rgb24 -s {W}x{H} -r 20 -i {raw_path} \\")
    print(f"    -vf 'scale=512:384:flags=lanczos' \\")
    print(f"    -c:v libsvtav1 -preset 0 -crf 32 -g 180 -r 20 output.mkv")


if __name__ == "__main__":
    main()
```

### Preprocessing Parameters to Test

| blur_sigma | margin_threshold | Expected Effect |
|-----------|-----------------|----------------|
| 1.0 | 3.0 | Very conservative: minimal blur, only very stable regions |
| 2.0 | 2.0 | Moderate: blur stable interiors |
| 3.0 | 1.5 | Aggressive: blur larger regions with moderate stability |
| 5.0 | 1.0 | Very aggressive: risk of argmax flips |
| 0 (flat) | 2.0 | Replace stable regions with class mean color |

### Integration with Compression Pipeline

The preprocessing adds a step before encoding:

```
videos/0.mkv
  -> preprocess.py (SegNet-aware blur)
  -> preprocessed/preprocessed.raw
  -> ffmpeg encode (using rawvideo input)
  -> archive/0.mkv
  -> zip -> archive.zip
```

Modified `compress.sh` for preprocessing:

```bash
#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

# Step 1: Preprocess
cd "$PD"
python preprocess.py \
  --input videos/0.mkv \
  --output "${HERE}/preprocessed" \
  --blur-sigma 3.0 \
  --margin-threshold 2.0 \
  --device cpu

# Step 2: Encode preprocessed video
W=1164; H=874  # original frame dimensions
ARCHIVE_DIR="${HERE}/archive"
rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -f rawvideo -pix_fmt rgb24 -s "${W}x${H}" -r 20 \
  -i "${HERE}/preprocessed/preprocessed.raw" \
  -vf "scale=512:384:flags=lanczos" \
  -c:v libsvtav1 -preset 0 -crf 32 \
  -g 180 -svtav1-params "enable-overlays=1:film-grain=0" \
  -r 20 "${ARCHIVE_DIR}/0.mkv"

# Step 3: Create archive
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
echo "Compressed to ${HERE}/archive.zip"
```

### Safety Check

Always run the verification step to ensure preprocessing does not flip argmax decisions:

```bash
python preprocess.py --input videos/0.mkv --output preprocessed/ \
  --blur-sigma 3.0 --margin-threshold 2.0 --verify --device cpu
```

If the average flip rate is nonzero, increase `margin_threshold` or decrease `blur_sigma` until it reaches zero. The preprocessing must be **lossless** from the SegNet perspective -- it should reduce entropy without changing any predictions.

---

## Task 4: Odd/Even Frame Asymmetry Exploitation

### Objective

The SegNet evaluates only the **last frame** of each 2-frame pair (frames 1, 3, 5, ..., 1199). The PoseNet evaluates both frames but its contribution is under a square root, making it less sensitive. This means odd frames are more important than even frames. Allocating more bits to odd frames and fewer to even frames should improve the overall score.

### Frame Pairs in Evaluation

```
Pair 0: frames 0 (even), 1 (odd) -> SegNet uses frame 1, PoseNet uses both
Pair 1: frames 2 (even), 3 (odd) -> SegNet uses frame 3, PoseNet uses both
...
Pair 599: frames 1198 (even), 1199 (odd) -> SegNet uses frame 1199, PoseNet uses both
```

600 pairs total. SegNet sees 600 frames (odd-indexed). PoseNet sees all 1200.

### Approach 1: Per-Frame QP Offsets via x265

x265 supports per-frame QP offsets via a zones file or `--qpfile`:

```
# qpfile format: <frame_number> <slice_type> <qp_offset>
# Even frames get higher QP (lower quality), odd frames get lower QP (higher quality)
0 I 4
1 I -2
2 P 4
3 P -2
4 P 4
5 P -2
...
```

Generate the qpfile:
```python
# generate_qpfile.py
with open("qpfile.txt", "w") as f:
    for i in range(1200):
        frame_type = "I" if i == 0 else "P"
        qp_offset = 4 if i % 2 == 0 else -2  # even frames: +4 QP, odd frames: -2 QP
        f.write(f"{i} {frame_type} {qp_offset}\n")
```

Usage with x265:
```bash
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i input.mkv \
  -vf "scale=512:384:flags=lanczos" \
  -c:v libx265 -preset veryslow -crf 30 \
  -g 180 \
  -x265-params "keyint=180:qpfile=qpfile.txt:log-level=warning" \
  -r 20 output.mkv
```

### Approach 2: Adaptive Quantization with SVT-AV1

SVT-AV1 does not directly support per-frame QP files, but it has temporal layer-based quality control. Alternative strategies:

1. **Two-pass encoding with different qualities for odd/even frames**: Encode odd frames at lower CRF, even frames at higher CRF, then interleave. This requires a custom pipeline:

```bash
# Extract odd and even frames
ffmpeg -i input.mkv -vf "select='not(mod(n\,2))',setpts=N/20/TB" -r 20 even_frames.mkv
ffmpeg -i input.mkv -vf "select='mod(n\,2)',setpts=N/20/TB" -r 20 odd_frames.mkv

# Encode separately
ffmpeg -i even_frames.mkv -c:v libsvtav1 -preset 0 -crf 38 -g 90 even_encoded.mkv
ffmpeg -i odd_frames.mkv -c:v libsvtav1 -preset 0 -crf 28 -g 90 odd_encoded.mkv

# Interleave back (custom Python script needed)
```

This approach is complex and loses temporal prediction between odd and even frames. It is likely not worth the complexity unless the per-frame QP approach with x265 shows significant improvement.

2. **Temporal layer QP offset**: SVT-AV1's hierarchical GOP structure already allocates fewer bits to non-reference frames. With a long GOP, the codec naturally makes some frames higher quality (reference frames) and others lower quality (non-reference). The question is whether the natural allocation aligns with the odd/even pattern we want.

3. **Film grain table manipulation**: Not applicable here.

### Approach 3: Frame Duplication for Odd Frames

A simpler idea: duplicate each odd frame so the codec encodes it twice, giving it more bits. Then at decode time, drop the duplicate. This doubles the frame count but could improve odd-frame quality at the cost of even-frame quality.

This is hacky and probably not effective with modern codecs that would just P-frame predict the duplicate and spend almost zero bits on it.

### Recommended Approach

Start with Approach 1 (x265 with qpfile) as a test. If it shows improvement, investigate whether SVT-AV1 has any similar capability. The QP offsets to test:

| Even QP offset | Odd QP offset | Rationale |
|----------------|--------------|-----------|
| +2 | -1 | Conservative |
| +4 | -2 | Moderate |
| +6 | -3 | Aggressive |
| +8 | -4 | Very aggressive |

The total bit budget should remain roughly constant (the positive offset on even frames saves bits, the negative offset on odd frames spends extra bits).

---

## Task 5: Post-Decode Refinement

### Objective

After decoding and upscaling to 1164x874, apply lightweight refinements that improve SegNet/PoseNet consistency. These must be fast enough to run within the 30-minute evaluation time limit.

### Approach 1: Edge-Aware Sharpening

Compression and upscaling blur edges. Since SegNet distortion is driven by class boundary disagreements, sharpening edges can help:

```python
import torch
import torch.nn.functional as F

def sharpen_frame(frame_rgb, strength=0.3):
    """
    Apply unsharp mask to sharpen edges.
    frame_rgb: (H, W, 3) uint8 tensor
    """
    x = frame_rgb.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)

    # Gaussian blur
    kernel_size = 5
    sigma = 1.0
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss = torch.exp(-coords.pow(2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    gauss_2d = gauss.unsqueeze(1) * gauss.unsqueeze(0)
    gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    pad = kernel_size // 2
    blurred = F.conv2d(F.pad(x, [pad]*4, mode='reflect'), gauss_2d, groups=3)

    # Unsharp mask: original + strength * (original - blurred)
    sharpened = x + strength * (x - blurred)
    return sharpened.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
```

### Approach 2: SegNet-Guided Color Correction

Run SegNet on the decoded frame and on the original frame, compute the argmax maps, and for pixels where they disagree, adjust the decoded pixel toward the original class centroid. This requires access to the original video at inflate time, which IS allowed by the rules.

However, this is essentially a neural refinement approach and may be better suited for Phase 2. For Phase 1, keep it simple.

### Approach 3: Bilateral Filter

A bilateral filter preserves edges while smoothing flat regions, which can help the SegNet by reducing noise in class interiors while keeping boundaries sharp:

```python
# Using OpenCV (if available):
import cv2
refined = cv2.bilateralFilter(frame_np, d=9, sigmaColor=75, sigmaSpace=75)
```

Note: OpenCV is not in the default dependencies. Check if it is available or add it.

### Integration

Add post-processing to `inflate.py`:

```python
def decode_and_resize_to_file(video_path: str, dst: str):
    target_w, target_h = camera_size
    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)

            # Optional: apply sharpening
            # t = sharpen_frame(t, strength=0.2)

            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n
```

### Time Budget

The inflate step processes 1200 frames. Per-frame operations:
- Decode: ~2ms/frame
- Bicubic upscale: ~5ms/frame
- Sharpening: ~3ms/frame
- SegNet inference (if needed): ~50ms/frame on CPU

Total for 1200 frames:
- Without SegNet: ~12 seconds (well within budget)
- With SegNet per-frame: ~60 seconds (still within budget)

The evaluation itself (running both networks on 600 pairs) takes the bulk of the 30-minute time limit. Keep post-processing fast.

---

## Deliverables

### File Structure

```
submissions/phase1/
    __init__.py         # Empty file (required for Python module imports)
    compress.sh         # Compression script
    inflate.py          # Inflate script (decode + upscale + optional post-processing)
    inflate.sh          # Shell wrapper for inflate.py
    archive/            # (generated) Compressed video files
    archive.zip         # (generated) Zipped archive

grid_search.py          # Automated parameter search (at repo root)
preprocess.py           # SegNet-aware preprocessing (at repo root)
```

### submissions/phase1/compress.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PD="$(cd "${HERE}/../.." && pwd)"

IN_DIR="${PD}/videos"
VIDEO_NAMES_FILE="${PD}/public_test_video_names.txt"
ARCHIVE_DIR="${HERE}/archive"
JOBS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir|--in_dir)
      IN_DIR="${2%/}"; shift 2 ;;
    --jobs)
      JOBS="$2"; shift 2 ;;
    --video-names-file|--video_names_file)
      VIDEO_NAMES_FILE="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--in-dir <dir>] [--jobs <n>] [--video-names-file <file>]" >&2
      exit 2 ;;
  esac
done

rm -rf "$ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

export IN_DIR ARCHIVE_DIR

# IMPORTANT: Update these parameters based on grid search results.
# These are initial values based on the leader's approach with improvements.
CODEC="libsvtav1"
CRF=32
PRESET=0
GOP=180
SCALE_W=512
SCALE_H=384

head -n "$(wc -l < "$VIDEO_NAMES_FILE")" "$VIDEO_NAMES_FILE" | while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue

  IN="${IN_DIR}/${rel}"
  BASE="${rel%.*}"
  OUT="${ARCHIVE_DIR}/${BASE}.mkv"

  echo "Encoding ${IN} -> ${OUT}"
  echo "  Codec: ${CODEC}, CRF: ${CRF}, Preset: ${PRESET}, GOP: ${GOP}, Scale: ${SCALE_W}x${SCALE_H}"

  ffmpeg -nostdin -y -hide_banner -loglevel warning \
    -r 20 -fflags +genpts -i "$IN" \
    -vf "scale=${SCALE_W}:${SCALE_H}:flags=lanczos" \
    -c:v ${CODEC} -preset ${PRESET} -crf ${CRF} \
    -g ${GOP} \
    -svtav1-params "enable-overlays=1:film-grain=0" \
    -r 20 "$OUT"

  echo "  Output size: $(stat -f%z "$OUT" 2>/dev/null || stat --printf='%s' "$OUT" 2>/dev/null || echo 'unknown') bytes"
done

# Create archive.zip
cd "$ARCHIVE_DIR"
zip -r "${HERE}/archive.zip" .
ZIPSIZE=$(stat -f%z "${HERE}/archive.zip" 2>/dev/null || stat --printf='%s' "${HERE}/archive.zip" 2>/dev/null || echo 'unknown')
echo "Compressed to ${HERE}/archive.zip (${ZIPSIZE} bytes)"
```

### submissions/phase1/inflate.py

```python
#!/usr/bin/env python
"""
Phase 1 inflate: decode compressed video, upscale to camera resolution, write .raw file.
Identical to baseline but may include post-decode refinements.
"""
import av
import torch
import torch.nn.functional as F
from frame_utils import camera_size, yuv420_to_rgb


def decode_and_resize_to_file(video_path: str, dst: str):
    target_w, target_h = camera_size  # (1164, 874)
    fmt = 'hevc' if video_path.endswith('.hevc') else None
    container = av.open(video_path, format=fmt)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3) uint8
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                x = t.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
                x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == "__main__":
    import sys
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize_to_file(src, dst)
    print(f"saved {n} frames")
```

### submissions/phase1/inflate.sh

```bash
#!/usr/bin/env bash
# Must produce a raw video file at `<output_dir>/<base_name>.raw`.
# A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)`
# where N is the number of frames, H and W match the original video dimensions, no header.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
  [ -z "$line" ] && continue
  BASE="${line%.*}"
  SRC="${DATA_DIR}/${BASE}.mkv"
  DST="${OUTPUT_DIR}/${BASE}.raw"

  [ ! -f "$SRC" ] && echo "ERROR: ${SRC} not found" >&2 && exit 1

  printf "Decoding + resizing %s ... " "$line"
  cd "$ROOT"
  python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"
```

### submissions/phase1/__init__.py

```python
# Empty - required for Python module imports
```

---

## Implementation Notes

### Working Directory

All paths are relative to `C:\Users\modce\Documents\comma_video_compression_challenge`. On the CI evaluation runner, the working directory is the repo root.

### Python Environment

- Python 3.11 (`.python-version`)
- Dependencies managed by `uv` with `pyproject.toml`
- Key packages: `torch`, `av`, `timm`, `safetensors`, `segmentation-models-pytorch`, `einops`, `tqdm`, `pillow`, `numpy`

### How to Run Full Evaluation Locally

```bash
cd C:\Users\modce\Documents\comma_video_compression_challenge

# Step 1: Compress
bash submissions/phase1/compress.sh

# Step 2: Evaluate (unzip + inflate + score)
bash evaluate.sh --submission-dir ./submissions/phase1 --device cpu
```

Or run each step individually for debugging:

```bash
# Unzip
rm -rf submissions/phase1/archive
mkdir -p submissions/phase1/archive
unzip -o submissions/phase1/archive.zip -d submissions/phase1/archive

# Inflate
bash submissions/phase1/inflate.sh \
  submissions/phase1/archive \
  submissions/phase1/inflated \
  public_test_video_names.txt

# Verify .raw file exists and has correct size
ls -la submissions/phase1/inflated/0.raw
# Expected size: 3,661,113,600 bytes (1200 frames * 874 * 1164 * 3)

# Evaluate
python evaluate.py \
  --submission-dir ./submissions/phase1 \
  --uncompressed-dir ./videos \
  --report ./submissions/phase1/report.txt \
  --video-names-file ./public_test_video_names.txt \
  --device cpu
```

### How evaluate.sh Connects Everything

```
evaluate.sh receives: --submission-dir ./submissions/phase1 --device cpu

evaluate.sh does:
  1. Checks ./submissions/phase1/archive.zip exists
  2. Checks ./submissions/phase1/inflate.sh exists
  3. Unzips archive.zip -> ./submissions/phase1/archive/
  4. Runs: bash ./submissions/phase1/inflate.sh \
       ./submissions/phase1/archive \
       ./submissions/phase1/inflated \
       ./public_test_video_names.txt
  5. Checks ./submissions/phase1/inflated/0.raw exists
  6. Runs: python evaluate.py \
       --submission-dir ./submissions/phase1 \
       --uncompressed-dir ./videos \
       --report ./submissions/phase1/report.txt \
       --video-names-file ./public_test_video_names.txt \
       --device cpu
```

### How inflate.sh Works

```
inflate.sh receives: $1=archive_dir $2=output_dir $3=file_list

For each video name in file_list (e.g., "0.mkv"):
  1. SRC = archive_dir/0.mkv
  2. DST = output_dir/0.raw
  3. cd to repo root
  4. python -m submissions.phase1.inflate SRC DST
```

This means inflate.py is run as a Python module from the repo root, which lets it import from `frame_utils`, `modules`, etc.

### Installing SVT-AV1

On the CI runner (Ubuntu):
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
# ffmpeg on Ubuntu usually includes libsvtav1
ffmpeg -encoders 2>/dev/null | grep svtav1
```

On Windows (local development):
```bash
# Check if your ffmpeg build includes libsvtav1:
ffmpeg -encoders 2>/dev/null | grep svtav1

# If not, download a full ffmpeg build:
# https://www.gyan.dev/ffmpeg/builds/ (full build includes libsvtav1)
```

### Installing VVenC (if needed)

```bash
# On Ubuntu:
sudo apt-get install -y libvvenc-dev vvenc
# Or build from source:
git clone https://github.com/fraunhoferhhi/vvenc.git
cd vvenc && mkdir build && cd build && cmake .. && make -j$(nproc) && sudo make install

# Build ffmpeg with --enable-libvvenc (if not already included)
```

### Key Constants Reference

| Constant | Value | Source |
|----------|-------|--------|
| `camera_size` | (1164, 874) W x H | `frame_utils.py:11` |
| `segnet_model_input_size` | (512, 384) W x H | `frame_utils.py:13` |
| `seq_len` | 2 | `frame_utils.py:10` |
| Original video size | 37,545,489 bytes | `videos/0.mkv` |
| Frame count | 1200 | 60s * 20fps |
| Sample count | 600 | 1200 / 2 |
| Raw frame size | 3,050,928 bytes | 874 * 1164 * 3 |
| Total .raw size | 3,661,113,600 bytes | 1200 * 3,050,928 |
| SegNet classes | 5 | `modules.py:105` |
| PoseNet output dims | 12 (6 used for distortion) | `modules.py:26,84` |

### Score Sensitivity Quick Reference

| Change | Score Impact |
|--------|-------------|
| SegNet distortion +0.001 | +0.100 |
| SegNet distortion -0.001 | -0.100 |
| PoseNet distortion from 0.33 to 0.30 | sqrt(3.3)-sqrt(3.0) = 1.817-1.732 = -0.085 |
| PoseNet distortion from 0.33 to 0.25 | sqrt(3.3)-sqrt(2.5) = 1.817-1.581 = -0.236 |
| Archive size from 835KB to 600KB | 25*(235000/37545489) = -0.157 |
| Archive size from 835KB to 400KB | 25*(435000/37545489) = -0.290 |
| Archive size from 835KB to 200KB | 25*(635000/37545489) = -0.423 |

### Priority Order

1. **Grid search** (Task 2) -- highest ROI, determines the best codec/parameters
2. **Encode resolution** (Task 1) -- quick to test, high impact
3. **SegNet preprocessing** (Task 3) -- moderate complexity, potentially high impact on SegNet term
4. **Odd/even asymmetry** (Task 4) -- complex, moderate impact
5. **Post-decode refinement** (Task 5) -- low complexity, low-moderate impact

### Strategy for Getting Below 2.90

Based on the score breakdown analysis, the most promising path is:

1. **Use VVC if available**: VVC should provide 20-30% better compression than AV1 at low bitrates. If we can reduce archive size from 835KB to 500KB while maintaining the same distortion, that saves 0.22 points on rate alone.

2. **Optimize encoding resolution**: Test 512x384 (exact model input) vs. 45% scale. If 512x384 reduces SegNet distortion by even 0.001 (from 0.00522 to 0.00422), that saves 0.10 points.

3. **Increase CRF slightly if using VVC**: Better codec efficiency means we can push CRF a bit higher, trading minimal distortion increase for significant rate reduction.

4. **Apply conservative SegNet preprocessing**: Even a 5% entropy reduction in sky/road regions gives the codec more bits for boundaries.

Combined, these optimizations should achieve a score in the 2.3-2.7 range.

---

## Appendix: Score Formula Derivation

From `evaluate.py:92`:

```python
score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate
```

Where:
- `segnet_dist` = average fraction of pixels with argmax disagreement, averaged over 600 samples (odd frames only)
- `posenet_dist` = average MSE of first 6 pose dims, averaged over 600 samples (both frames)
- `rate` = `archive.zip size` / `sum of all files in videos/ directory` = `archive.zip size` / 37,545,489

The three terms operate on very different scales:
- SegNet: a distortion of 0.005 (0.5% pixel disagreement) = 0.50 score
- PoseNet: a distortion of 0.30 (MSE) = sqrt(3.0) = 1.73 score
- Rate: a compression to 2% = 25 * 0.02 = 0.50 score

The PoseNet term is inherently the largest contributor at reasonable compression levels because even the uncompressed video (rate=0) would have PoseNet=0, but the sqrt function means reducing PoseNet from 0.30 to 0.15 only saves about 0.46 points.

The most effective strategy is to minimize SegNet distortion (100x multiplier, linear) and rate (25x multiplier, linear), while accepting moderate PoseNet distortion (under square root, diminishing returns).

---

## Appendix: Understanding the Resample Chain

The full chain from encode to evaluation:

```
Original: 1164x874 (camera_size)
  -> compress.sh: Lanczos downscale to encode resolution (e.g., 512x384)
  -> codec: encode at encode resolution
  -> inflate.py: decode at encode resolution
  -> inflate.py: bicubic upscale to 1164x874
  -> evaluate.py: loads 1164x874 .raw file
  -> DistortionNet.preprocess_input:
       -> PoseNet: bilinear downscale to 512x384, then rgb_to_yuv6 -> (192, 256)
       -> SegNet: bilinear downscale to 512x384
```

The round-trip is: 1164x874 -> encode_res -> 1164x874 -> 512x384.

If we encode at exactly 512x384, the chain becomes: 1164x874 -> 512x384 -> 1164x874 -> 512x384. The question is whether `bicubic_up(512x384, 1164x874)` followed by `bilinear_down(1164x874, 512x384)` is an identity operation. It is NOT -- the bicubic upscale introduces ringing artifacts that the bilinear downscale does not perfectly remove.

However, encoding at 512x384 still avoids the additional quantization error introduced by encoding at a non-model resolution. The grid search in Task 1 will determine empirically which resolution minimizes the total score.

An alternative: modify `inflate.py` to NOT upscale to 1164x874, and instead write frames at 512x384 directly. But this would violate the .raw format requirement (must be `(N, 874, 1164, 3)`). The evaluation asserts the shape matches `camera_size`.

Another alternative worth testing: use **bilinear** upscale in inflate.py instead of bicubic, since the evaluation will bilinear-downscale anyway. Bilinear up followed by bilinear down might be more consistent than bicubic up followed by bilinear down.

```python
# In inflate.py, change interpolation mode:
x = F.interpolate(x, size=(target_h, target_w), mode='bilinear', align_corners=False)
# instead of:
x = F.interpolate(x, size=(target_h, target_w), mode='bicubic', align_corners=False)
```

Test both and compare scores.
