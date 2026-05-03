#!/usr/bin/env python
"""X3: CMA-ES mask flips extended to top 200 pairs (vs top 100 in O2)."""
import sys, os, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)
from sidecar_mask_verified import (mask_sidecar_size, regenerate_frames_from_masks)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches
from explore_o2_cmaes_mask import cma_es_mask_for_pair

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def main():
    device = torch.device("cuda")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    score_bl = bp['score']

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}")
    print("\n=== X3: CMA-ES mask K=2 on top 200 ===")
    t0 = time.time()
    new_mask_patches = dict(bp['mask_patches'])
    n_added = 0
    for i, pi in enumerate(rank[:200]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        if pi in new_mask_patches:
            for (x, y, c) in new_mask_patches[pi]:
                m[0, y, x] = c
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        flips = cma_es_mask_for_pair(gen, m, p, gt_p, posenet, device, K=2, pop=10, gens=15)
        if flips:
            if pi in new_mask_patches:
                new_mask_patches[pi].extend(flips)
            else:
                new_mask_patches[pi] = flips
            n_added += len(flips)
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/200 (added {n_added}) ({time.time()-t0:.0f}s)", flush=True)

    sb_mask = mask_sidecar_size(new_mask_patches)
    new_masks = masks_cpu.clone()
    for pi, ps in new_mask_patches.items():
        for (x, y, c) in ps:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)

    p_top = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                         [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = channel_sidecar_size(rgb_patches)
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"X3: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} delta={full['score']-score_bl:+.4f} ({time.time()-t0:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "x3_cmaes_top200_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "n_added", "sb_total", "score", "delta"])
        w.writerow(["x3_cmaes_top200_K2", n_added, sb_mask+sb_rgb, full['score'],
                    full['score']-score_bl])


if __name__ == "__main__":
    main()
