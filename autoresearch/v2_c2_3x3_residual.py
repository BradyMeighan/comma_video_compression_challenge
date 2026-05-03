"""C2 simplified: 3x3 mask block flips on X2-residual pairs.

For each pair where X2's 2x2 didn't accept any flip (~138 pairs), try 3x3 blocks.
Stack on top of cached X5 (X2 + CMA-ES K=2).

VECTORIZED:
  - Loads X2 + CMA-ES patches from CACHE (saves ~50 min)
  - 3x3 batched candidate eval (5x speedup)
  - Re-runs CMA-ES ONLY for top-100-pairs whose mask state changed (3x3 added)
"""
import sys, os, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import (State, verified_greedy_block_mask_batched,
                         cma_es_mask_for_pair_batched,
                         compose_score, serialize_block_mask_v2)
from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6)
import sidecar_explore as se
from sidecar_stack import fast_eval
from sidecar_mask_verified import (mask_sidecar_size, regenerate_frames_from_masks)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches
from explore_x2_mask_blocks import block_mask_sidecar_size

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"


def main():
    import csv
    s = State()

    # Load cache
    print("\n=== C2: loading cache ===", flush=True)
    if not (CACHE_DIR / "masks_after_x2.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder.py first to populate {CACHE_DIR}")
    masks_after_x2 = torch.load(CACHE_DIR / "masks_after_x2.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", 'rb') as f:
        x2_patches = pickle.load(f)
    with open(CACHE_DIR / "x2_rejected.pkl", 'rb') as f:
        x2_rejected = pickle.load(f)
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", 'rb') as f:
        cmaes_patches_cached = pickle.load(f)
    print(f"loaded: x2={len(x2_patches)} rejected={len(x2_rejected)} "
          f"cmaes={len(cmaes_patches_cached)}", flush=True)

    # Stage 1: 3x3 BATCHED on X2-rejected pairs
    print(f"\n=== C2: 3x3 blocks on {len(x2_rejected)} X2-rejected pairs (BATCHED) ===", flush=True)
    t1 = time.time()
    block3_patches = {}
    for i, pi in enumerate(x2_rejected):
        m = s.masks_cpu[pi:pi+1].to(s.device).long()
        p = s.poses[pi:pi+1].to(s.device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_block_mask_batched(
            s.gen, m, p, gt_p, s.posenet, s.device, K=1, n_candidates=10, block=3)
        if accepted:
            block3_patches[pi] = accepted
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(x2_rejected)} ({time.time()-t1:.0f}s)", flush=True)
    print(f"3x3: {len(block3_patches)} accepted ({time.time()-t1:.0f}s)", flush=True)

    # Apply X2 + 3x3 to mask state
    new_masks = masks_after_x2.clone()
    for pi, ps in block3_patches.items():
        for (x, y, c) in ps:
            new_masks[pi, y:y+3, x:x+3] = c

    # Stage 2: CMA-ES K=2 on top 100 — REUSE cache for pairs whose state didn't change
    print(f"\n=== C2: CMA-ES K=2 top 100 (reusing cache where unchanged) ===", flush=True)
    t2 = time.time()
    top100 = [int(x) for x in s.rank[:100]]
    changed_pairs = set(block3_patches.keys()) & set(top100)
    print(f"  {len(changed_pairs)} pairs changed (3x3 added), {len(top100)-len(changed_pairs)} reused from cache",
          flush=True)
    cmaes_patches = {}
    for pi in top100:
        if pi in changed_pairs:
            m = new_masks[pi:pi+1].to(s.device).long()
            p = s.poses[pi:pi+1].to(s.device).float()
            gt_p = p.clone()
            flips = cma_es_mask_for_pair_batched(s.gen, m, p, gt_p, s.posenet, s.device,
                                                    K=2, pop=12, gens=15)
            if flips:
                cmaes_patches[pi] = flips
        elif pi in cmaes_patches_cached:
            cmaes_patches[pi] = cmaes_patches_cached[pi]
    # Apply CMA-ES flips
    for pi, fs in cmaes_patches.items():
        for (x, y, c) in fs:
            new_masks[pi, y, x] = c
    print(f"CMA-ES: {len(cmaes_patches)} pairs ({time.time()-t2:.0f}s)", flush=True)

    # Re-find RGB on doubly-mask-patched frames
    print(f"\n=== C2: re-find channel-only RGB ===", flush=True)
    f1_new, f2_new = regenerate_frames_from_masks(s.gen, new_masks, s.poses, s.device)
    p_top = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                        [int(x) for x in s.rank[:250]], K=5, n_iter=80, device=s.device)
    p_tail = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                         [int(x) for x in s.rank[250:500]], K=2, n_iter=80, device=s.device)
    rgb_patches = {**p_top, **p_tail}

    sb_x2 = len(serialize_block_mask_v2(x2_patches))
    sb_b3 = len(serialize_block_mask_v2(block3_patches))
    sb_cma = mask_sidecar_size(cmaes_patches)
    sb_rgb = channel_sidecar_size(rgb_patches)
    sb_total = sb_x2 + sb_b3 + sb_cma + sb_rgb

    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    seg, pose = fast_eval(f1_combined, f2_new, s.data["val_rgb"], s.device)
    full = compose_score(seg, pose, s.model_bytes, sb_total)
    delta = full['score'] - s.score_baseline
    print(f"\nC2 final: sb_x2={sb_x2}B sb_3x3={sb_b3}B sb_cma={sb_cma}B sb_rgb={sb_rgb}B "
          f"total={sb_total}B score={full['score']:.4f} delta={delta:+.4f} ({time.time()-t1:.0f}s)")

    out_csv = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_c2_3x3_residual_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_x2", "sb_3x3", "sb_cma", "sb_rgb", "sb_total", "score", "delta"])
        w.writerow(["c2_x2+3x3_residual+cmaes+rgb", sb_x2, sb_b3, sb_cma, sb_rgb,
                    sb_total, full['score'], delta])


if __name__ == "__main__":
    main()
