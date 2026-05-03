#!/usr/bin/env python
"""
Probe: per-block brightness correction grid for odd frames (SegNet).

Only processes blocks that actually have mismatches (skips ~85% of blocks).
For each error block, sweeps brightness deltas to find the one that fixes
the most boundary errors without breaking other blocks.

Usage:
  python probe_odd_frame_blockgrid.py [--block-size 16] [--deltas "-8,-4,-2,0,2,4,8"]
  python probe_odd_frame_blockgrid.py --benchmark  # time estimate only
"""
import argparse, bz2, struct, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules import SegNet, segnet_sd_path

W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--deltas", type=str, default="-10,-6,-3,0,3,6,10")
    ap.add_argument("--raw-in", type=str, default=str(SUB / "inflated" / "0.raw"))
    ap.add_argument("--gt-cache", type=str, default=str(SUB / "_cache" / "gt.pt"))
    ap.add_argument("--benchmark", action="store_true", help="Run on 5 frames and estimate total time")
    ap.add_argument("--max-pairs", type=int, default=0, help="Limit pairs (0=all)")
    args = ap.parse_args()

    BS = args.block_size
    deltas = [int(x) for x in args.deltas.split(",")]
    nonzero_deltas = [d for d in deltas if d != 0]
    print(f"Device: {DEVICE}", flush=True)
    print(f"Block size: {BS}x{BS}, deltas: {nonzero_deltas}", flush=True)

    grid_h = MODEL_H // BS
    grid_w = MODEL_W // BS
    print(f"Grid: {grid_h}x{grid_w} = {grid_h * grid_w} blocks per frame")

    # Load SegNet
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)

    # Load GT
    gt = torch.load(args.gt_cache, weights_only=True)
    gt_seg = gt["seg"].to(DEVICE)
    n_pairs = gt_seg.shape[0]

    # Load inflated frames
    raw = np.fromfile(args.raw_in, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)

    # Extract odd frames at model resolution
    print("Resizing odd frames to model resolution...", flush=True)
    odd_model = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        odd_model[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    if args.benchmark:
        n_test = 5
    elif args.max_pairs > 0:
        n_test = min(args.max_pairs, n_pairs)
    else:
        n_test = n_pairs

    # Pass 1: identify error blocks per frame (one inference each)
    print(f"\nPass 1: identifying error blocks ({n_test} frames)...", flush=True)
    t_pass1 = time.time()

    frame_error_blocks = []  # list of lists of (by, bx, n_errors)
    baseline_errors = np.zeros(n_test, dtype=np.int32)

    with torch.inference_mode():
        for i in range(0, n_test, 16):
            end = min(i + 16, n_test)
            x = torch.from_numpy(odd_model[i:end].copy()).to(DEVICE).float().permute(0, 3, 1, 2)
            preds = segnet(x).argmax(dim=1)

            for j in range(end - i):
                mismatch = (preds[j] != gt_seg[i + j])
                baseline_errors[i + j] = mismatch.sum().item()

                # Find which blocks have errors
                blocks = []
                for by in range(grid_h):
                    for bx in range(grid_w):
                        y0, y1 = by * BS, (by + 1) * BS
                        x0, x1 = bx * BS, (bx + 1) * BS
                        block_err = mismatch[y0:y1, x0:x1].sum().item()
                        if block_err > 0:
                            blocks.append((by, bx, block_err))
                frame_error_blocks.append(blocks)

    pass1_time = time.time() - t_pass1
    total_error_blocks = sum(len(b) for b in frame_error_blocks)
    avg_error_blocks = total_error_blocks / n_test
    print(f"  Pass 1 done in {pass1_time:.1f}s")
    print(f"  Avg error blocks per frame: {avg_error_blocks:.0f} / {grid_h * grid_w} "
          f"({100 * avg_error_blocks / (grid_h * grid_w):.0f}% need processing)")
    print(f"  Total SegNet inferences needed for pass 2: "
          f"{total_error_blocks * len(nonzero_deltas):,}")
    print(f"  Baseline errors: {baseline_errors.sum():,} total, {baseline_errors.mean():.0f}/frame")

    # Warmup + benchmark single inference
    with torch.inference_mode():
        x = torch.randn(1, 3, MODEL_H, MODEL_W, device=DEVICE)
        for _ in range(3):
            segnet(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            segnet(x)
        torch.cuda.synchronize()
        ms_per_inference = (time.perf_counter() - t0) / 20 * 1000

    print(f"  SegNet inference: {ms_per_inference:.1f}ms")
    total_inferences = total_error_blocks * len(nonzero_deltas)
    if args.benchmark:
        # Extrapolate to full 600 frames
        full_inferences = int(total_inferences / n_test * n_pairs)
        full_time = full_inferences * ms_per_inference / 1000
        print(f"\n=== BENCHMARK ESTIMATE ===")
        print(f"  Frames to process: {n_pairs}")
        print(f"  Estimated error blocks: {int(avg_error_blocks * n_pairs):,}")
        print(f"  Estimated inferences: {full_inferences:,}")
        print(f"  Estimated time: {full_time:.0f}s ({full_time/60:.1f} min)")
        print(f"  (Pass 1 adds ~{pass1_time / n_test * n_pairs:.0f}s)")
        return

    # Pass 2: sweep deltas only on error blocks
    print(f"\nPass 2: optimizing {total_error_blocks} error blocks...", flush=True)
    t_pass2 = time.time()

    all_grids = np.zeros((n_test, grid_h, grid_w), dtype=np.int8)
    total_fixed = 0
    inferences_done = 0

    for i in range(n_test):
        error_blocks = frame_error_blocks[i]
        if not error_blocks:
            continue

        frame = odd_model[i].copy().astype(np.float32)
        gt_map = gt_seg[i]

        # Get baseline prediction
        with torch.inference_mode():
            x = torch.from_numpy(frame).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
            base_pred = segnet(x).argmax(dim=1)[0]
            base_total = (base_pred != gt_map).sum().item()

        best_grid = np.zeros((grid_h, grid_w), dtype=np.int8)

        for by, bx, block_errs in error_blocks:
            y0, y1 = by * BS, (by + 1) * BS
            x0, x1 = bx * BS, (bx + 1) * BS

            best_delta = 0
            best_total = base_total

            for delta in nonzero_deltas:
                modified = frame.copy()
                modified[y0:y1, x0:x1] = np.clip(modified[y0:y1, x0:x1] + delta, 0, 255)

                with torch.inference_mode():
                    x_mod = torch.from_numpy(modified).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
                    mod_pred = segnet(x_mod).argmax(dim=1)[0]
                inferences_done += 1

                new_total = (mod_pred != gt_map).sum().item()
                if new_total < best_total:
                    best_delta = delta
                    best_total = new_total

            best_grid[by, bx] = best_delta

        # Apply all corrections and measure actual improvement
        corrected = frame.copy()
        for by in range(grid_h):
            for bx in range(grid_w):
                if best_grid[by, bx] != 0:
                    y0, y1 = by * BS, (by + 1) * BS
                    x0, x1 = bx * BS, (bx + 1) * BS
                    corrected[y0:y1, x0:x1] = np.clip(
                        corrected[y0:y1, x0:x1] + float(best_grid[by, bx]), 0, 255
                    )

        with torch.inference_mode():
            x_corr = torch.from_numpy(corrected).to(DEVICE).permute(2, 0, 1).unsqueeze(0)
            corr_pred = segnet(x_corr).argmax(dim=1)[0]
            new_errors = (corr_pred != gt_map).sum().item()
            fixed = baseline_errors[i] - new_errors
            total_fixed += max(0, fixed)

        all_grids[i] = best_grid
        nonzero = np.count_nonzero(best_grid)

        if i % 20 == 0 or i == n_test - 1:
            elapsed = time.time() - t_pass2
            rate = inferences_done / elapsed if elapsed > 0 else 0
            remaining = total_error_blocks * len(nonzero_deltas) - inferences_done
            eta = remaining / rate if rate > 0 else 0
            print(f"  Pair {i:3d}/{n_test}: {baseline_errors[i]} errs, "
                  f"fixed {max(0,fixed)}, {nonzero} blocks, "
                  f"{rate:.0f} inf/s, ETA {eta:.0f}s", flush=True)

    # Results
    new_total = baseline_errors[:n_test].sum() - total_fixed
    old_seg = baseline_errors[:n_test].sum() / (n_test * MODEL_H * MODEL_W)
    new_seg = new_total / (n_test * MODEL_H * MODEL_W)

    print(f"\n{'='*60}")
    print(f"RESULTS ({n_test} frames)")
    print(f"{'='*60}")
    print(f"Baseline errors: {baseline_errors[:n_test].sum():,} ({old_seg:.6f} seg_dist)")
    print(f"Fixed errors:    {total_fixed:,}")
    print(f"New errors:      {new_total:,} ({new_seg:.6f} seg_dist)")
    print(f"100*seg improvement: {100*(old_seg - new_seg):.4f}")

    # Compress and save
    raw_bytes = all_grids.tobytes()
    header = struct.pack("<III", grid_h, grid_w, n_test)
    payload = bz2.compress(header + raw_bytes, compresslevel=9)
    out_path = SUB / "archive" / "frame1_blockgrid_q.bin"
    out_path.write_bytes(payload)

    print(f"\nSidecar: {out_path.name}")
    print(f"  Raw: {len(raw_bytes):,} bytes ({len(raw_bytes)/1024:.1f} KB)")
    print(f"  Compressed: {len(payload):,} bytes ({len(payload)/1024:.1f} KB)")
    rate_cost = 25 * len(payload) / 37_545_489
    seg_gain = 100 * (old_seg - new_seg)
    print(f"  Rate cost:  +{rate_cost:.4f}")
    print(f"  Seg gain:   -{seg_gain:.4f}")
    print(f"  Net score:  {-seg_gain + rate_cost:+.4f} ({'WINS' if seg_gain > rate_cost else 'LOSES'})")
    print(f"  Time: {time.time() - t_pass2:.0f}s")


if __name__ == "__main__":
    main()
