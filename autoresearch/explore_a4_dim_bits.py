#!/usr/bin/env python
"""
A4: Pose-dim-aware bit allocation.

Idea: instead of int8 deltas (8 bits each), use variable bit width per delta based on
the dim distribution. From sidecar_deep_analyze:
- After RGB patches, residual RMS per dim: dim0=0.014, dim1=0.037, dim2=0.029, dim3=0.011, dim4=0.008, dim5=0.027

For OUR sidecar this doesn't apply directly (we encode pixel deltas, not pose dims).
But: we can analyze our actual delta distribution and use VARIABLE-LENGTH int encoding.

This script analyzes the delta distribution of our best sidecar and tests:
- Truncating int8 → int6 (saves 25% per delta byte) — drops range from ±127 to ±31
- Truncating int8 → int7 — range ±63
- Pure entropy estimate (lower bound)
"""
import sys, os, pickle, struct, bz2
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def main():
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    rgb_patches = bp['rgb_patches']
    score_bl = bp['score']
    sb_total_bl = bp['sb_total_bz2']

    # Analyze delta distribution
    all_deltas = []
    for ps in rgb_patches.values():
        for (x, y, c, d) in ps:
            all_deltas.append(d)
    all_deltas = np.array(all_deltas)

    print(f"=== A4: delta distribution analysis ===")
    print(f"N deltas: {len(all_deltas)}")
    print(f"Mean: {all_deltas.mean():.2f}, std: {all_deltas.std():.2f}")
    print(f"Min: {all_deltas.min()}, max: {all_deltas.max()}")
    print(f"Quantiles: q25={np.percentile(all_deltas,25)} q50={np.percentile(np.abs(all_deltas),50)} q90={np.percentile(np.abs(all_deltas),90)} q99={np.percentile(np.abs(all_deltas),99)}")
    print(f"Frac with |d|<=31 (int6 fits): {(np.abs(all_deltas)<=31).mean():.3f}")
    print(f"Frac with |d|<=63 (int7 fits): {(np.abs(all_deltas)<=63).mean():.3f}")
    print(f"Frac with |d|<=15 (int5 fits): {(np.abs(all_deltas)<=15).mean():.3f}")
    print(f"Frac with |d|<=7 (int4 fits): {(np.abs(all_deltas)<=7).mean():.3f}")

    # Empirical Shannon entropy of int8 deltas
    counts = np.bincount((all_deltas + 128).astype(np.int32), minlength=256)
    p = counts / counts.sum()
    p_nz = p[p > 0]
    H = -(p_nz * np.log2(p_nz)).sum()
    print(f"\nShannon entropy of deltas: {H:.3f} bits/symbol")
    print(f"Optimal coding would give: {len(all_deltas) * H / 8:.0f}B for {len(all_deltas)} deltas")
    print(f"Current bz2 cost per delta: {bp['sb_rgb_bz2'] / sum(len(v) for v in rgb_patches.values()):.2f}B")

    # Try TRUNCATING to lower bit-widths (lossy delta encoding)
    print("\n=== Test: truncate deltas to lower bit width (lossy quantization) ===")
    for bits in [4, 5, 6, 7]:
        max_val = 2**(bits-1) - 1
        all_d_trunc = np.clip(all_deltas, -max_val, max_val)
        mse_loss = ((all_deltas - all_d_trunc) ** 2).mean()
        print(f"  int{bits} (range ±{max_val}): clipped {(all_deltas != all_d_trunc).sum()} deltas, "
              f"avg MSE per clipped delta = {((all_deltas[all_deltas != all_d_trunc] - np.clip(all_deltas[all_deltas != all_d_trunc], -max_val, max_val))**2).mean() if (all_deltas != all_d_trunc).sum() > 0 else 0:.1f}")

    import csv
    with open(OUTPUT_DIR / "a4_dim_bits_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["n_deltas", len(all_deltas)])
        w.writerow(["mean", all_deltas.mean()])
        w.writerow(["std", all_deltas.std()])
        w.writerow(["entropy_bits_per_sym", H])
        w.writerow(["bz2_actual_per_delta", bp['sb_rgb_bz2'] / sum(len(v) for v in rgb_patches.values())])
        w.writerow(["frac_int6_fits", (np.abs(all_deltas)<=31).mean()])
        w.writerow(["frac_int7_fits", (np.abs(all_deltas)<=63).mean()])


if __name__ == "__main__":
    main()
