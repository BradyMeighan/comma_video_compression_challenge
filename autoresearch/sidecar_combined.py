#!/usr/bin/env python
"""
COMBINED sidecar: mask patches + RGB patches stacked.

Protocol:
  1. Apply mask patches (verified greedy, K=3 on top 200 hardest).
  2. Regenerate frames from new mask.
  3. Find RGB patches on new frames (350_K7+250_K2 tier).
  4. Apply RGB patches.
  5. Eval combined.

Since mask changes gen output, RGB patches need to be RE-FOUND on the new frames
(can't reuse existing RGB patches).
"""
import sys, os, time, csv, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      pack_pair_yuv6, estimate_model_bytes)
from train import Generator, load_data_full, coords
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs, per_pair_pose_mse)
from sidecar_mask_verified import (gen_forward_with_oh_mask, pose_loss_for_pair,
                                     verified_greedy_mask, mask_sidecar_size,
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

    # Generate baseline
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    seg, pose = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg, pose, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    csv_path = OUTPUT_DIR / "combined_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_mask_pairs", "K_mask_total", "n_rgb_pairs",
                                 "K_rgb_total", "sb_mask", "sb_rgb", "sb_total",
                                 "score", "pose_term", "delta", "elapsed"])

    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    # === RGB-only reference (control) ===
    print("\n=== Reference: RGB-only 350_K7+250_K2 ===", flush=True)
    t0 = time.time()
    p_top = find_pose_patches_for_pairs(
        f1_all, f2_all, poses, posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_all, f2_all, poses, posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    rgb_only = {**p_top, **p_tail}
    sb_rgb_only = sparse_sidecar_size(rgb_only)
    f1_rgb = apply_sparse_patches(f1_all, rgb_only)
    s, p = fast_eval(f1_rgb, f2_all, data["val_rgb"], device)
    full_rgb = fast_compose(s, p, model_bytes, sb_rgb_only)
    print(f"  RGB-only: sb={sb_rgb_only}B score={full_rgb['score']:.4f} "
          f"pose={full_rgb['pose_term']:.4f} delta={full_rgb['score']-base['score']:+.4f} "
          f"({time.time()-t0:.1f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["rgb_only_350K7+250K2", 0, 0, 600, 2950, 0,
                                 sb_rgb_only, sb_rgb_only,
                                 full_rgb['score'], full_rgb['pose_term'],
                                 full_rgb['score']-base['score'], time.time()-t0])

    # === Test 1: mask K=3 top100 + RGB on new frames ===
    print("\n=== Test 1: mask K=3 top100 + RGB 350_K7+250_K2 ===", flush=True)
    t0 = time.time()
    mask_patches = {}
    print("  Searching mask patches...", flush=True)
    for pi in rank[:100]:
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device, K=3, n_candidates=15)
        if accepted:
            mask_patches[pi] = accepted

    sb_mask = mask_sidecar_size(mask_patches)
    print(f"  Mask: {len(mask_patches)} pairs accepted, sb={sb_mask}B "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Apply mask + regenerate frames
    new_masks = masks_cpu.clone()
    for pi, patches in mask_patches.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)

    # Eval mask-only (sanity)
    s, p = fast_eval(f1_new, f2_new, data["val_rgb"], device)
    mask_only = fast_compose(s, p, model_bytes, sb_mask)
    print(f"  Mask-only: score={mask_only['score']:.4f} pose={mask_only['pose_term']:.4f} "
          f"delta={mask_only['score']-base['score']:+.4f}", flush=True)

    # Find RGB on NEW frames
    print("  Finding RGB on new (mask-patched) frames...", flush=True)
    t1 = time.time()
    p_top_new = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail_new = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    rgb_new = {**p_top_new, **p_tail_new}
    sb_rgb = sparse_sidecar_size(rgb_new)
    f1_combined = apply_sparse_patches(f1_new, rgb_new)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    elapsed = time.time() - t0
    delta = full['score'] - base['score']
    print(f"  >> Combined K=3_top100: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} pose={full['pose_term']:.4f} delta={delta:+.4f} "
          f"({elapsed:.1f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["combined_mask_K3_top100", len(mask_patches),
                                 sum(len(v) for v in mask_patches.values()),
                                 600, 2950, sb_mask, sb_rgb, sb_mask + sb_rgb,
                                 full['score'], full['pose_term'], delta, elapsed])

    # === Test 2: mask K=5 top200 + RGB ===
    print("\n=== Test 2: mask K=5 top200 + RGB ===", flush=True)
    t0 = time.time()
    mask_patches2 = {}
    print("  Searching mask patches...", flush=True)
    for pi in rank[:200]:
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device, K=5, n_candidates=15)
        if accepted:
            mask_patches2[pi] = accepted
    sb_mask2 = mask_sidecar_size(mask_patches2)
    print(f"  Mask: {len(mask_patches2)} pairs, K_total={sum(len(v) for v in mask_patches2.values())}, "
          f"sb={sb_mask2}B ({time.time()-t0:.1f}s)", flush=True)

    new_masks2 = masks_cpu.clone()
    for pi, patches in mask_patches2.items():
        for (x, y, c) in patches:
            new_masks2[pi, y, x] = c
    f1_new2, f2_new2 = regenerate_frames_from_masks(gen, new_masks2, poses, device)

    p_top_new2 = find_pose_patches_for_pairs(
        f1_new2, f2_new2, poses, posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail_new2 = find_pose_patches_for_pairs(
        f1_new2, f2_new2, poses, posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    rgb_new2 = {**p_top_new2, **p_tail_new2}
    sb_rgb2 = sparse_sidecar_size(rgb_new2)
    f1_combined2 = apply_sparse_patches(f1_new2, rgb_new2)
    s, p = fast_eval(f1_combined2, f2_new2, data["val_rgb"], device)
    full2 = fast_compose(s, p, model_bytes, sb_mask2 + sb_rgb2)
    elapsed = time.time() - t0
    delta = full2['score'] - base['score']
    print(f"  >> Combined K=5_top200: sb_mask={sb_mask2}B sb_rgb={sb_rgb2}B sb_total={sb_mask2+sb_rgb2}B "
          f"score={full2['score']:.4f} pose={full2['pose_term']:.4f} delta={delta:+.4f} "
          f"({elapsed:.1f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["combined_mask_K5_top200", len(mask_patches2),
                                 sum(len(v) for v in mask_patches2.values()),
                                 600, 2950, sb_mask2, sb_rgb2, sb_mask2 + sb_rgb2,
                                 full2['score'], full2['pose_term'], delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
