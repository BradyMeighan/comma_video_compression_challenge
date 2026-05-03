#!/usr/bin/env python
"""
Extension experiment: stack winner says extending coverage helps.
Test:
  1. Uniform K=7 across ALL 600 (full coverage)
  2. top500_K7 / top600_K7 — push extension limits
  3. Tiered: top300_K10 + next200_K5 + last100_K3 (variable K by error tier)
  4. Tiered with K=8/4/2 (more patches concentrated, fewer in tail)

All use cached DistortionNet + bs=8.
"""
import sys, os, time, csv
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (load_posenet, estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("Loading...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    seg, pose = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg, pose, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    csv_path = OUTPUT_DIR / "extend_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "pairs", "K_total", "sidecar_bytes",
                                 "score", "pose_term", "delta", "elapsed"])

    def run(spec, pair_list_K_pairs):
        """pair_list_K_pairs: list of (pair_indices, K) tuples to merge."""
        all_patches = {}
        t0 = time.time()
        for pairs, K in pair_list_K_pairs:
            patches = find_pose_patches_for_pairs(
                f1_all, f2_all, data["val_poses"], posenet,
                pairs, K=K, n_iter=80, device=device)
            for k, v in patches.items():
                all_patches[k] = v  # last write wins; pairs disjoint here anyway
        elapsed = time.time() - t0
        sb = sparse_sidecar_size(all_patches)
        f1_p = apply_sparse_patches(f1_all, all_patches)
        seg, pose = fast_eval(f1_p, f2_all, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb)
        delta = full['score'] - base['score']
        npairs = len(all_patches)
        K_total = sum(xy.shape[0] for xy, d in all_patches.values())
        print(f"  >> {spec}: pairs={npairs} K_total={K_total} sidecar={sb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, npairs, K_total, sb,
                                     full['score'], full['pose_term'], delta, elapsed])

    # ── Uniform K=7 extensions ──
    print("\n=== Uniform K=7 extensions ===", flush=True)
    all_pairs = [int(x) for x in rank[:600]]
    run("top400_K7",  [([int(x) for x in rank[:400]], 7)])  # repro stack winner
    run("top500_K7",  [([int(x) for x in rank[:500]], 7)])
    run("top600_K7",  [(all_pairs, 7)])  # = uniform K=7 across all 600

    # ── Uniform K=6, K=5 across all 600 (more conservative) ──
    print("\n=== Uniform K variations ===", flush=True)
    run("top600_K6",  [(all_pairs, 6)])
    run("top600_K5",  [(all_pairs, 5)])

    # ── Tiered: more on hard, less on easy ──
    print("\n=== Tiered K ===", flush=True)
    # Tier1: top300 K=10, next200 K=5, last100 K=3
    run("tier_K10_5_3",
        [([int(x) for x in rank[:300]], 10),
         ([int(x) for x in rank[300:500]], 5),
         ([int(x) for x in rank[500:]], 3)])
    # Tier2: top200 K=10, next200 K=7, next200 K=4
    run("tier_K10_7_4",
        [([int(x) for x in rank[:200]], 10),
         ([int(x) for x in rank[200:400]], 7),
         ([int(x) for x in rank[400:]], 4)])
    # Tier3: top150 K=12, next150 K=8, next300 K=4
    run("tier_K12_8_4",
        [([int(x) for x in rank[:150]], 12),
         ([int(x) for x in rank[150:300]], 8),
         ([int(x) for x in rank[300:]], 4)])
    # Tier4: top100 K=15, next200 K=8, next300 K=4
    run("tier_K15_8_4",
        [([int(x) for x in rank[:100]], 15),
         ([int(x) for x in rank[100:300]], 8),
         ([int(x) for x in rank[300:]], 4)])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
