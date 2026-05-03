#!/usr/bin/env python
"""X5: X2 (2x2 mask blocks) + CMA-ES additional flips on top.

If both methods individually beat baseline, do their gains stack?
"""
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
from sidecar_mask_verified import (mask_sidecar_size, regenerate_frames_from_masks,
                                     gen_forward_with_oh_mask, pose_loss_for_pair)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches
from explore_x2_mask_blocks import (verified_greedy_block_mask, block_mask_sidecar_size)
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

    # Step 1: X2 — 2x2 mask block flips, K=1, top 600
    print("\n=== X5 step 1: X2 2x2 block mask flips on top 600 ===")
    t0 = time.time()
    block_patches = {}
    for i, pi in enumerate(rank[:600]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_block_mask(gen, m, p, gt_p, posenet, device,
                                                    K=1, n_candidates=10, block=2)
        if accepted:
            block_patches[pi] = accepted
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/600 ({time.time()-t0:.0f}s)", flush=True)
    sb_block = block_mask_sidecar_size(block_patches)
    print(f"X2 blocks: {len(block_patches)} pairs, {sb_block}B")

    # Apply blocks + regen
    new_masks = masks_cpu.clone()
    for pi, ps in block_patches.items():
        for (x, y, c) in ps:
            new_masks[pi, y:y+2, x:x+2] = c

    # Step 2: CMA-ES additional 1-pixel mask flips on top 100 hardest
    print("\n=== X5 step 2: CMA-ES K=2 1-pixel flips on top 100 ===")
    t1 = time.time()
    extra_pixel_flips = {}  # pair_i -> list of (x, y, class) for 1-pixel patches
    for i, pi in enumerate(rank[:100]):
        pi = int(pi)
        m = new_masks[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        flips = cma_es_mask_for_pair(gen, m, p, gt_p, posenet, device, K=2, pop=10, gens=15)
        if flips:
            extra_pixel_flips[pi] = flips
            for (x, y, c) in flips:
                new_masks[pi, y, x] = c
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/100 ({time.time()-t1:.0f}s)", flush=True)

    # Build combined mask sidecar size: blocks (5B each) + 1-pixel (5B each)
    # We need a unified format. Since both are 5B per patch effectively, we can
    # encode them together with a type bit. Or just sum the separately-encoded sizes.
    sb_pixel = mask_sidecar_size(extra_pixel_flips)
    sb_mask_combined = sb_block + sb_pixel
    print(f"Combined mask: blocks={sb_block}B + pixel_extra={sb_pixel}B = {sb_mask_combined}B")

    # Regen with all mask changes
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)

    # Re-find RGB
    p_top = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                         [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = channel_sidecar_size(rgb_patches)
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask_combined + sb_rgb)
    print(f"X5 final: sb_mask={sb_mask_combined}B sb_rgb={sb_rgb}B sb_total={sb_mask_combined+sb_rgb}B "
          f"score={full['score']:.4f} delta={full['score']-score_bl:+.4f} ({time.time()-t0:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "x5_x2_plus_cmaes_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_mask", "sb_rgb", "sb_total", "score", "delta"])
        w.writerow(["x5_blocks_top600+cmaes_K2_top100", sb_mask_combined, sb_rgb,
                    sb_mask_combined+sb_rgb, full['score'], full['score']-score_bl])


if __name__ == "__main__":
    main()
