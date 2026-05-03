#!/usr/bin/env python
"""
PUSH further: mask K=2/K=3 verified greedy + smaller RGB tiers.

Hypothesis: bigger mask absorbs more pose error, allowing even smaller RGB.
Per-byte breakeven: each mask flip is 5B; adds ~0.0001-0.0002 pose reduction.
RGB patch is 7B with diminishing pose returns.

Tests:
  A. Mask K=2 top600 + RGB 200_K5+200_K2 (more mask, same RGB as runner-up)
  B. Mask K=2 top600 + RGB 150_K5+200_K2 (more mask, less RGB)
  C. Mask K=3 top600 + RGB 200_K5+200_K2
  D. Mask K=3 top300 + Mask K=1 top300 (tiered mask)
  E. Mask K=2 top300 + Mask K=1 next300 + RGB 200_K5+200_K2
"""
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
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_mask_verified import (verified_greedy_mask, mask_sidecar_size,
                                     regenerate_frames_from_masks)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def build_mask(gen, masks_cpu, poses, posenet, device, pair_K_pairs, n_candidates=10):
    """pair_K_pairs: list of (pair_indices, K) tuples. Builds combined mask sidecar."""
    mask_patches = {}
    t0 = time.time()
    for pair_indices, K in pair_K_pairs:
        for i, pi in enumerate(pair_indices):
            pi = int(pi)
            m = masks_cpu[pi:pi+1].to(device).long()
            # If we already have patches for this pair, apply them first
            if pi in mask_patches:
                for (x, y, c) in mask_patches[pi]:
                    m[0, y, x] = c
            p = poses[pi:pi+1].to(device).float()
            gt_p = p.clone()
            accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device,
                                                 K=K, n_candidates=n_candidates)
            if accepted:
                if pi in mask_patches:
                    mask_patches[pi].extend(accepted)
                else:
                    mask_patches[pi] = accepted
        print(f"  ... tier ({len(pair_indices)} pairs at K={K}) done ({time.time()-t0:.0f}s)", flush=True)
    return mask_patches


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
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    csv_path = OUTPUT_DIR / "push_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_mask_pairs", "K_mask_total", "sb_mask",
                                 "rgb_tier", "sb_rgb", "sb_total",
                                 "score", "pose_term", "delta", "delta_vs_311", "elapsed"])

    BEST_KNOWN_DELTA = -0.0287
    print(f"Best known so far: {BEST_KNOWN_DELTA:+.4f} (mask K=1 top600 + RGB 250_K5+250_K2)\n", flush=True)

    def run(spec, mask_specs, rgb_tiers):
        """mask_specs: list of (rank_slice, K). rgb_tiers: list of (start, end, K)."""
        print(f"\n=== {spec} ===", flush=True)
        t0 = time.time()
        # Build mask
        pair_K_pairs = [(rank[s:e].tolist(), K) for s, e, K in mask_specs]
        mask_patches = build_mask(gen, masks_cpu, poses, posenet, device, pair_K_pairs, n_candidates=10)
        sb_mask = mask_sidecar_size(mask_patches)
        K_total = sum(len(v) for v in mask_patches.values())
        print(f"  Mask: {len(mask_patches)} pairs accepted, K_total={K_total}, sb={sb_mask}B", flush=True)

        # Apply + regenerate
        new_masks = masks_cpu.clone()
        for pi, patches in mask_patches.items():
            for (x, y, c) in patches:
                new_masks[pi, y, x] = c
        f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)

        # Find RGB
        rgb_patches = {}
        for start, end, K in rgb_tiers:
            pairs = [int(x) for x in rank[start:end]]
            ps = find_pose_patches_for_pairs(
                f1_new, f2_new, poses, posenet,
                pairs, K=K, n_iter=80, device=device)
            rgb_patches.update(ps)
        sb_rgb = sparse_sidecar_size(rgb_patches)
        f1_combined = apply_sparse_patches(f1_new, rgb_patches)
        s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
        full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
        delta = full['score'] - base['score']
        d_vs_best = delta - BEST_KNOWN_DELTA
        elapsed = time.time() - t0
        print(f"  >> {spec}: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} d_vs_best={d_vs_best:+.4f} ({elapsed:.0f}s)", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, len(mask_patches), K_total, sb_mask,
                                     str(rgb_tiers), sb_rgb, sb_mask + sb_rgb,
                                     full['score'], full['pose_term'],
                                     delta, d_vs_best, elapsed])

    # Test A: K=2 mask + RGB 200_K5+200_K2 (matches runner-up RGB tier)
    run("A_mask_K2top600+rgb_200K5+200K2",
        [(0, 600, 2)],
        [(0, 200, 5), (200, 400, 2)])

    # Test B: K=2 mask + smaller RGB
    run("B_mask_K2top600+rgb_150K5+200K2",
        [(0, 600, 2)],
        [(0, 150, 5), (150, 350, 2)])

    # Test C: K=3 mask + RGB 200_K5+200_K2
    run("C_mask_K3top600+rgb_200K5+200K2",
        [(0, 600, 3)],
        [(0, 200, 5), (200, 400, 2)])

    # Test D: tiered mask K=3 top300 + K=1 next300
    run("D_mask_K3top300+K1next300+rgb_200K5+200K2",
        [(0, 300, 3), (300, 600, 1)],
        [(0, 200, 5), (200, 400, 2)])

    # Test E: K=2 top300 + K=1 next300
    run("E_mask_K2top300+K1next300+rgb_200K5+200K2",
        [(0, 300, 2), (300, 600, 1)],
        [(0, 200, 5), (200, 400, 2)])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
