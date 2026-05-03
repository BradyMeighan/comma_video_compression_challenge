#!/usr/bin/env python
"""
Gradient-based per-pixel correction for odd frames (SegNet).

For each mismatched pixel, backprops through SegNet to find the minimal
RGB delta that flips the prediction to the correct class. Stores sparse
corrections as (x, y, dr, dg, db) tuples, bz2 compressed.

Usage:
  python probe_odd_frame_gradient.py [--max-pairs 30] [--steps 20] [--lr 5.0]
"""
import argparse, bz2, struct, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
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
    ap.add_argument("--max-pairs", type=int, default=0, help="0=all 600")
    ap.add_argument("--steps", type=int, default=30, help="gradient steps per frame")
    ap.add_argument("--lr", type=float, default=8.0, help="pixel-space learning rate")
    ap.add_argument("--raw-in", type=str, default=str(SUB / "inflated" / "0.raw"))
    ap.add_argument("--gt-cache", type=str, default=str(SUB / "_cache" / "gt.pt"))
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    print(f"Steps: {args.steps}, LR: {args.lr}", flush=True)

    # Load SegNet (need gradients, so don't use inference_mode)
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)

    # Load GT
    gt = torch.load(args.gt_cache, weights_only=True)
    gt_seg = gt["seg"].to(DEVICE)  # (600, MODEL_H, MODEL_W)
    n_pairs = gt_seg.shape[0]

    # Load inflated frames
    raw = np.fromfile(args.raw_in, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)

    # Extract odd frames at model resolution
    print("Resizing odd frames to model resolution...", flush=True)
    odd_model = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        odd_model[i] = cv2.resize(raw[2 * i + 1], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    n_test = args.max_pairs if args.max_pairs > 0 else n_pairs

    # Pass 1: find baseline errors
    print(f"\nPass 1: baseline errors ({n_test} frames)...", flush=True)
    baseline_errors = np.zeros(n_test, dtype=np.int32)
    with torch.inference_mode():
        for i in range(0, n_test, 16):
            end = min(i + 16, n_test)
            x = torch.from_numpy(odd_model[i:end].copy()).to(DEVICE).float().permute(0, 3, 1, 2)
            preds = segnet(x).argmax(dim=1)
            for j in range(end - i):
                baseline_errors[i + j] = (preds[j] != gt_seg[i + j]).sum().item()
    print(f"  Baseline: {baseline_errors.sum():,} errors, {baseline_errors.mean():.0f}/frame")

    # Pass 2: gradient-based correction per frame
    print(f"\nPass 2: gradient optimization ({n_test} frames, {args.steps} steps each)...", flush=True)
    t0 = time.time()

    total_fixed = 0
    all_deltas = []  # per-frame list of (y, x, dr, dg, db) corrections

    for i in range(n_test):
        frame = torch.from_numpy(odd_model[i].copy()).to(DEVICE).float()
        frame_chw = frame.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        target = gt_seg[i]  # (H, W)

        # Create learnable delta (starts at zero = no correction)
        delta = torch.zeros_like(frame_chw, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=args.lr)

        for step in range(args.steps):
            optimizer.zero_grad()
            corrected = (frame_chw + delta).clamp(0, 255)
            logits = segnet(corrected)  # (1, 5, H, W)

            # Loss: cross-entropy against ground truth classes
            # This pushes each pixel's prediction toward the correct class
            loss = F.cross_entropy(logits, target.unsqueeze(0))

            # Regularization: keep delta small (L2 penalty)
            loss = loss + 0.001 * (delta ** 2).mean()

            loss.backward()
            optimizer.step()

            # Clamp delta to reasonable range
            with torch.no_grad():
                delta.data.clamp_(-30, 30)

        # Evaluate result
        with torch.inference_mode():
            corrected = (frame_chw + delta.detach()).clamp(0, 255)
            new_pred = segnet(corrected).argmax(dim=1)[0]
            new_errors = (new_pred != target).sum().item()
            fixed = baseline_errors[i] - new_errors
            total_fixed += max(0, fixed)

        # Extract sparse corrections: only where prediction actually changed
        with torch.inference_mode():
            orig_pred = segnet(frame_chw).argmax(dim=1)[0]
            flipped = (orig_pred != new_pred)  # pixels where prediction changed
            # Keep only corrections where prediction improved (flipped to correct)
            improved = flipped & (new_pred == target)
            n_improved = improved.sum().item()

        # Store the delta values at improved pixels
        delta_np = delta.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        frame_corrections = []
        if n_improved > 0:
            coords = improved.nonzero()  # (N, 2)
            for c in coords:
                y, x = c[0].item(), c[1].item()
                dr, dg, db = delta_np[y, x]
                frame_corrections.append((y, x, int(round(dr)), int(round(dg)), int(round(db))))

        all_deltas.append(frame_corrections)

        if i % 10 == 0 or i == n_test - 1:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_test - i - 1)
            print(f"  Pair {i:3d}/{n_test}: {baseline_errors[i]} errs, "
                  f"fixed {max(0,fixed)}, {n_improved} px improved, "
                  f"delta_mag={delta.detach().abs().mean().item():.1f}, "
                  f"ETA {eta:.0f}s", flush=True)

    # Results
    old_total = baseline_errors[:n_test].sum()
    new_total = old_total - total_fixed
    old_seg = old_total / (n_test * MODEL_H * MODEL_W)
    new_seg = new_total / (n_test * MODEL_H * MODEL_W)

    print(f"\n{'='*60}")
    print(f"RESULTS ({n_test} frames)")
    print(f"{'='*60}")
    print(f"Baseline errors: {old_total:,} ({old_seg:.6f} seg_dist, 100*seg={100*old_seg:.4f})")
    print(f"Fixed errors:    {total_fixed:,}")
    print(f"New errors:      {new_total:,} ({new_seg:.6f} seg_dist, 100*seg={100*new_seg:.4f})")
    print(f"100*seg improvement: {100*(old_seg - new_seg):.4f}")

    # Measure storage cost
    total_corrections = sum(len(f) for f in all_deltas)
    avg_corrections = total_corrections / n_test
    print(f"\nCorrections: {total_corrections} total, {avg_corrections:.0f}/frame avg")

    # Encode: per-frame sparse corrections
    # Format: for each frame, store count + (y_uint16, x_uint16, dr_int8, dg_int8, db_int8) per correction
    buf = bytearray()
    buf.extend(struct.pack("<I", n_test))
    for corrections in all_deltas:
        buf.extend(struct.pack("<H", len(corrections)))
        for y, x, dr, dg, db in corrections:
            buf.extend(struct.pack("<HH", y, x))
            buf.extend(struct.pack("<bbb",
                max(-127, min(127, dr)),
                max(-127, min(127, dg)),
                max(-127, min(127, db))))

    compressed = bz2.compress(bytes(buf), compresslevel=9)
    out_path = SUB / "archive" / "frame1_gradient_q.bin"
    out_path.write_bytes(compressed)

    # Also save the full dense delta maps for potential reuse
    cache_path = SUB / "_cache" / "gradient_deltas.npz"
    cache_path.parent.mkdir(exist_ok=True)
    # Store as list of sparse corrections
    np.savez_compressed(str(cache_path),
                        corrections=[np.array(c) for c in all_deltas],
                        baseline_errors=baseline_errors[:n_test])
    print(f"Cached to {cache_path}")

    print(f"\nSidecar: {out_path.name}")
    print(f"  Raw: {len(buf):,} bytes ({len(buf)/1024:.1f} KB)")
    print(f"  Compressed: {len(compressed):,} bytes ({len(compressed)/1024:.1f} KB)")

    rate_cost = 25 * len(compressed) / 37_545_489
    seg_gain = 100 * (old_seg - new_seg)
    print(f"  Rate cost:  +{rate_cost:.4f}")
    print(f"  Seg gain:   -{seg_gain:.4f}")
    print(f"  Net score:  {-seg_gain + rate_cost:+.4f} ({'WINS' if seg_gain > rate_cost else 'LOSES'})")

    # Extrapolate to full 600 frames
    if n_test < n_pairs:
        full_compressed_est = len(compressed) * n_pairs / n_test
        full_rate = 25 * full_compressed_est / 37_545_489
        full_seg_gain = seg_gain  # assume similar distribution
        print(f"\n  Extrapolated to {n_pairs} frames:")
        print(f"    Sidecar: ~{full_compressed_est/1024:.1f} KB")
        print(f"    Rate cost: +{full_rate:.4f}")
        print(f"    Net score: {-full_seg_gain + full_rate:+.4f}")

    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
