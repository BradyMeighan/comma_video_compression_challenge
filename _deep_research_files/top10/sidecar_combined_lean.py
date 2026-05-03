#!/usr/bin/env python
"""
Combined LEAN: mask K=1 top600 + SMALLER RGB tier.
Hypothesis: mask absorbs some pose error, so smaller RGB tier suffices.
The byte savings on RGB outweigh the +1.8KB mask cost.

Test 4 RGB tier sizes:
  - 200_K5 (pure pose top200)
  - 200_K5 + 200_K2 (mid tier extension)
  - 250_K5 + 250_K2
  - 300_K5 + 300_K2
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

    # Step 1: build mask K=1 top600 ONCE
    N_CANDIDATES = int(os.environ.get("N_CANDIDATES", "10"))
    print(f"\n=== Building mask K=1 top600 (one-time, n_candidates={N_CANDIDATES}) ===", flush=True)
    t0 = time.time()
    mask_patches = {}
    for i, pi in enumerate(rank[:600]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device, K=1, n_candidates=N_CANDIDATES)
        if accepted:
            mask_patches[pi] = accepted
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/600 done ({time.time()-t0:.0f}s)", flush=True)
    sb_mask = mask_sidecar_size(mask_patches)
    print(f"Mask: {len(mask_patches)} pairs, sb={sb_mask}B ({time.time()-t0:.1f}s)", flush=True)

    # Apply mask + regenerate ONCE
    new_masks = masks_cpu.clone()
    for pi, patches in mask_patches.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)
    s_m, p_m = fast_eval(f1_new, f2_new, data["val_rgb"], device)
    mask_only = fast_compose(s_m, p_m, model_bytes, sb_mask)
    print(f"Mask-only: score={mask_only['score']:.4f} pose={mask_only['pose_term']:.4f} "
          f"delta={mask_only['score']-base['score']:+.4f}", flush=True)

    csv_path = OUTPUT_DIR / "combined_lean_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "rgb_tiers", "sb_mask", "sb_rgb", "sb_total",
                     "score", "pose_term", "delta_baseline", "delta_vs_rgb_only", "elapsed"])

    # Test multiple RGB tier configs on top of mask
    rgb_configs = [
        ("rgb_200_K5",            [(0, 200, 5)]),
        ("rgb_200_K5+200_K2",     [(0, 200, 5), (200, 400, 2)]),
        ("rgb_250_K5+250_K2",     [(0, 250, 5), (250, 500, 2)]),
        ("rgb_300_K5+300_K2",     [(0, 300, 5), (300, 600, 2)]),
        ("rgb_350_K6+250_K2",     [(0, 350, 6), (350, 600, 2)]),
        ("rgb_350_K7_only",       [(0, 350, 7)]),  # no tail
        ("rgb_250_K7+250_K2",     [(0, 250, 7), (250, 500, 2)]),
    ]

    rgb_baseline_known_delta = -0.0268
    print(f"\nKnown RGB-only best delta: {rgb_baseline_known_delta:+.4f}", flush=True)

    for spec, tiers in rgb_configs:
        print(f"\n=== {spec} ===", flush=True)
        t1 = time.time()
        rgb_patches = {}
        for start, end, K in tiers:
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
        delta_vs_rgb = delta - rgb_baseline_known_delta
        elapsed = time.time() - t1
        print(f"  >> {spec}: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} d_vs_rgb={delta_vs_rgb:+.4f} ({elapsed:.1f}s)",
              flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, str(tiers), sb_mask, sb_rgb, sb_mask + sb_rgb,
                                     full['score'], full['pose_term'],
                                     delta, delta_vs_rgb, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
