"""Build the X5 cache (X2 + CMA-ES + frames + RGB) once. All v2 experiments load it.

Saves to OUTPUT_DIR/v2_cache/:
  - x2_patches.pkl              dict: pair_i -> [(x, y, c), ...]   (2x2 blocks)
  - x2_rejected.pkl             list of pair_i where X2 found nothing
  - masks_after_x2.pt           tensor (n_pairs, MH, MW) long
  - cmaes_top100_patches.pkl    dict: pair_i -> [(x, y, c), ...]   (single-pixel)
  - masks_x5.pt                 tensor — masks_after_x2 + CMA-ES applied
  - frames_x5.pt                {'f1', 'f2'} uint8 frames from masks_x5
  - rgb_patches_x5.pkl          dict: pair_i -> [(x, y, c, d), ...]
  - cache_meta.json             timing + counts

This script uses BATCHED versions:
  - X2 batched candidate eval (5x faster than serial)
  - CMA-ES batched population eval (10x faster than serial)
  - RGB find already batches over pairs (bs=8)
"""
import sys, os, pickle, time, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import (State, verified_greedy_block_mask_batched,
                         cma_es_mask_for_pair_batched)
from sidecar_mask_verified import regenerate_frames_from_masks
from sidecar_channel_only import find_channel_only_patches

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    s = State()
    meta = {}

    # ── Stage 1: X2 batched on top 600 ──────────────────────────────────
    print("\n=== cache: X2 (2x2 blocks, BATCHED) on top 600 ===", flush=True)
    t0 = time.time()
    x2_patches = {}
    x2_rejected = []
    for i, pi in enumerate(s.rank[:600]):
        pi = int(pi)
        m = s.masks_cpu[pi:pi+1].to(s.device).long()
        p = s.poses[pi:pi+1].to(s.device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_block_mask_batched(
            s.gen, m, p, gt_p, s.posenet, s.device,
            K=1, n_candidates=10, block=2)
        if accepted:
            x2_patches[pi] = accepted
        else:
            x2_rejected.append(pi)
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/600 (rej={len(x2_rejected)}) ({time.time()-t0:.0f}s)",
                  flush=True)
    meta['x2_seconds'] = time.time() - t0
    meta['x2_accepted'] = len(x2_patches)
    meta['x2_rejected'] = len(x2_rejected)
    print(f"X2 done: {len(x2_patches)} accepted / {len(x2_rejected)} rejected "
          f"({meta['x2_seconds']:.0f}s)", flush=True)

    masks_after_x2 = s.masks_cpu.clone()
    for pi, ps in x2_patches.items():
        for (x, y, c) in ps:
            masks_after_x2[pi, y:y+2, x:x+2] = c

    # ── Stage 2: CMA-ES K=2 batched on top 100 ──────────────────────────
    print("\n=== cache: CMA-ES K=2 (BATCHED) on top 100 ===", flush=True)
    t1 = time.time()
    cmaes_patches = {}
    for i, pi in enumerate(s.rank[:100]):
        pi = int(pi)
        m = masks_after_x2[pi:pi+1].to(s.device).long()
        p = s.poses[pi:pi+1].to(s.device).float()
        gt_p = p.clone()
        flips = cma_es_mask_for_pair_batched(s.gen, m, p, gt_p, s.posenet, s.device,
                                                K=2, pop=12, gens=15)
        if flips:
            cmaes_patches[pi] = flips
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/100 ({time.time()-t1:.0f}s)", flush=True)
    meta['cmaes_seconds'] = time.time() - t1
    meta['cmaes_accepted'] = len(cmaes_patches)
    print(f"CMA-ES done: {len(cmaes_patches)} pairs improved "
          f"({meta['cmaes_seconds']:.0f}s)", flush=True)

    masks_x5 = masks_after_x2.clone()
    for pi, fs in cmaes_patches.items():
        for (x, y, c) in fs:
            masks_x5[pi, y, x] = c

    # ── Stage 3: regen frames from x5 masks ─────────────────────────────
    print("\n=== cache: regen frames from x5 masks ===", flush=True)
    t2 = time.time()
    f1_x5, f2_x5 = regenerate_frames_from_masks(s.gen, masks_x5, s.poses, s.device)
    meta['regen_seconds'] = time.time() - t2
    print(f"regen done ({meta['regen_seconds']:.0f}s)", flush=True)

    # ── Stage 4: RGB patches on x5 frames ───────────────────────────────
    print("\n=== cache: find channel-only RGB on x5 frames ===", flush=True)
    t3 = time.time()
    p_top = find_channel_only_patches(f1_x5, f2_x5, s.poses, s.posenet,
                                        [int(x) for x in s.rank[:250]],
                                        K=5, n_iter=80, device=s.device)
    p_tail = find_channel_only_patches(f1_x5, f2_x5, s.poses, s.posenet,
                                         [int(x) for x in s.rank[250:500]],
                                         K=2, n_iter=80, device=s.device)
    rgb_patches = {**p_top, **p_tail}
    meta['rgb_seconds'] = time.time() - t3
    meta['rgb_pairs'] = len(rgb_patches)
    print(f"RGB done: {len(rgb_patches)} pairs ({meta['rgb_seconds']:.0f}s)", flush=True)

    # ── Save everything ─────────────────────────────────────────────────
    print("\n=== cache: saving artifacts ===", flush=True)
    with open(CACHE_DIR / "x2_patches.pkl", 'wb') as f:
        pickle.dump(x2_patches, f)
    with open(CACHE_DIR / "x2_rejected.pkl", 'wb') as f:
        pickle.dump(x2_rejected, f)
    torch.save(masks_after_x2, CACHE_DIR / "masks_after_x2.pt")
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", 'wb') as f:
        pickle.dump(cmaes_patches, f)
    torch.save(masks_x5, CACHE_DIR / "masks_x5.pt")
    torch.save({'f1': f1_x5, 'f2': f2_x5}, CACHE_DIR / "frames_x5.pt")
    with open(CACHE_DIR / "rgb_patches_x5.pkl", 'wb') as f:
        pickle.dump(rgb_patches, f)
    meta['total_seconds'] = time.time() - t0
    with open(CACHE_DIR / "cache_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nCache built. total={meta['total_seconds']:.0f}s")
    print(f"  X2: {meta['x2_seconds']:.0f}s   CMA-ES: {meta['cmaes_seconds']:.0f}s   "
          f"regen: {meta['regen_seconds']:.0f}s   RGB: {meta['rgb_seconds']:.0f}s")
    print(f"  files in {CACHE_DIR}")


if __name__ == "__main__":
    main()
