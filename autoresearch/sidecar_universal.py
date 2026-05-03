#!/usr/bin/env python
"""Quick universal-patch test using the fixed memory-efficient backward."""
import sys, os, csv, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

import torch
from prepare import load_posenet, MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE
from train import Generator, load_data_full
import sidecar_explore as se

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))

def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Loading model + data...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)

    print("Generating frames...", flush=True)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    base = se.eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=gen)
    base_score = se.compose_score(base["seg_dist"], base["pose_dist"], base["model_bytes"], 0)
    print(f"Baseline: score={base_score['score']:.4f} pose={base_score['pose_term']:.4f}", flush=True)

    posenet = load_posenet(device)
    csv_path = OUTPUT_DIR / "universal_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["K", "n_iter", "sidecar_bytes", "score", "seg_term", "pose_term", "rate_term", "delta", "elapsed_sec"])

    for K, n_iter in [(20, 50), (50, 50), (100, 50), (200, 50)]:
        print(f"\n=== Universal K={K} n_iter={n_iter} ===", flush=True)
        t0 = time.time()
        pts_xy, pts_d = se.find_universal_patch(f1_all, f2_all, data["val_poses"], posenet, K=K, n_iter=n_iter, device=device)
        elapsed = time.time() - t0
        print(f"  patch search: {elapsed:.1f}s", flush=True)
        f1_p = se.apply_universal_patch_to_frames(f1_all, pts_xy, pts_d)
        sb = se.sidecar_size(pts_xy, pts_d, None, 0)
        result = se.eval_with_frames(f1_p, f2_all, data, device, gen_for_bytes=gen)
        full = se.compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        delta = full['score'] - base_score['score']
        print(f"  K={K}: sidecar={sb}B score={full['score']:.4f} pose={full['pose_term']:.4f} delta={delta:+.4f}", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([K, n_iter, sb, full['score'], full['seg_term'], full['pose_term'], full['rate_term'], delta, elapsed])

    print(f"\nDone. {csv_path}")

if __name__ == "__main__":
    main()
