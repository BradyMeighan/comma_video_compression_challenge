#!/usr/bin/env python
"""Quick sweep of uniform K over all 600 pairs at high n_iter to find the
score-vs-bytes Pareto front. Compares directly against current best -0.026.
"""
import sys, os, time, csv
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import load_posenet
from train import Generator, load_data_full
import sidecar_explore as se

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
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    base = se.eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=gen)
    base_score = se.compose_score(base["seg_dist"], base["pose_dist"], base["model_bytes"], 0)
    print(f"Baseline: score={base_score['score']:.4f}", flush=True)

    posenet = load_posenet(device)
    csv_path = OUTPUT_DIR / "uniform_sweep.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["K", "n_iter", "sidecar_bytes", "score",
                                 "seg_term", "pose_term", "rate_term", "delta", "elapsed"])

    for K, n_iter in [(3, 100), (5, 100), (7, 100), (10, 100), (15, 100), (20, 100), (30, 80)]:
        print(f"\n=== K={K} n_iter={n_iter} ===", flush=True)
        t0 = time.time()
        pts_xy, pts_d, _ = se.find_pose_patches(
            f1_all, f2_all, data["val_poses"], posenet,
            K=K, n_iter=n_iter, lr=1.0, device=device, target='f1')
        elapsed = time.time() - t0
        f1_p = se.apply_patches_to_frames(f1_all, pts_xy, pts_d, radius=0)
        sb = se.sidecar_size(pts_xy, pts_d, None, 0)
        result = se.eval_with_frames(f1_p, f2_all, data, device, gen_for_bytes=gen)
        full = se.compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        delta = full['score'] - base_score['score']
        print(f"  K={K}: sidecar={sb}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"rate={full['rate_term']:.4f} delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([K, n_iter, sb, full['score'], full['seg_term'],
                                     full['pose_term'], full['rate_term'], delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
