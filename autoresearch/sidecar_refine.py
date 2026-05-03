#!/usr/bin/env python
"""Refine around best-combo winner H (top400_K7 + tail200_K2 = -0.0263).
Tighter sweep on tier boundaries + lower-K tail variants."""
import sys, os, time, csv
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import load_posenet, estimate_model_bytes
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


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
    print(f"Baseline: score={base['score']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    csv_path = OUTPUT_DIR / "refine_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "pairs", "K_total", "sidecar_bytes",
                                 "score", "pose_term", "delta", "elapsed"])

    def run(spec, tiers):
        all_patches = {}
        t0 = time.time()
        for start, end, K in tiers:
            pairs = [int(x) for x in rank[start:end]]
            if not pairs or K <= 0: continue
            ps = find_pose_patches_for_pairs(
                f1_all, f2_all, data["val_poses"], posenet,
                pairs, K=K, n_iter=80, device=device)
            all_patches.update(ps)
        elapsed = time.time() - t0
        sb = sparse_sidecar_size(all_patches)
        f1_p = apply_sparse_patches(f1_all, all_patches)
        seg, pose = fast_eval(f1_p, f2_all, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb)
        delta = full['score'] - base['score']
        K_total = sum(xy.shape[0] for xy, d in all_patches.values())
        npairs = len(all_patches)
        print(f"  >> {spec}: pairs={npairs} K_total={K_total} sidecar={sb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, npairs, K_total, sb,
                                     full['score'], full['pose_term'], delta, elapsed])

    print("\n=== Refinement around H ===", flush=True)
    # Reproduce H (control)
    run("H_repro:400_K7+200_K2",     [(0, 400, 7), (400, 600, 2)])
    # Lower tail
    run("400_K7+200_K1",             [(0, 400, 7), (400, 600, 1)])
    run("400_K7+150_K2+50_K1",       [(0, 400, 7), (400, 550, 2), (550, 600, 1)])
    run("400_K7+100_K2",             [(0, 400, 7), (400, 500, 2)])
    # Different top tier sizes
    run("350_K7+250_K2",             [(0, 350, 7), (350, 600, 2)])
    run("450_K7+150_K2",             [(0, 450, 7), (450, 600, 2)])
    # Mixed: big head, small tail
    run("200_K10+200_K5+200_K2",     [(0, 200, 10), (200, 400, 5), (400, 600, 2)])
    run("100_K12+300_K6+200_K2",     [(0, 100, 12), (100, 400, 6), (400, 600, 2)])
    # K=7 cap, taper tail
    run("400_K7+100_K3+100_K1",      [(0, 400, 7), (400, 500, 3), (500, 600, 1)])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
