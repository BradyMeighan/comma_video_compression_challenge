#!/usr/bin/env python
"""
The CRITICAL combined test: mask K=1 top600 + RGB 350_K7+250_K2.
Should reveal if maximum mask + maximum RGB stack to push score below 0.30.
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

    # Step 1: Mask K=1 top600
    print("\n=== Step 1: mask K=1 top600 (verified greedy) ===", flush=True)
    t0 = time.time()
    mask_patches = {}
    for i, pi in enumerate(rank[:600]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device, K=1, n_candidates=10)
        if accepted:
            mask_patches[pi] = accepted
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/600 done, {len(mask_patches)} accepted ({time.time()-t0:.0f}s)", flush=True)
    sb_mask = mask_sidecar_size(mask_patches)
    print(f"  Mask: {len(mask_patches)}/600 accepted, sb={sb_mask}B ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Apply mask + regenerate
    new_masks = masks_cpu.clone()
    for pi, patches in mask_patches.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)
    s, p = fast_eval(f1_new, f2_new, data["val_rgb"], device)
    print(f"  Mask-only: score={fast_compose(s, p, model_bytes, sb_mask)['score']:.4f} "
          f"pose={np.sqrt(10*p):.4f}", flush=True)

    # Step 3: Find RGB patches on patched frames
    print("\n=== Step 2: RGB on patched frames (350_K7+250_K2) ===", flush=True)
    t1 = time.time()
    p_top = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = sparse_sidecar_size(rgb_patches)
    f1_combined = apply_sparse_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"  RGB+mask done ({time.time()-t1:.1f}s)", flush=True)
    print(f"\n=== FINAL ===", flush=True)
    print(f"  sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B", flush=True)
    print(f"  score={full['score']:.4f} pose={full['pose_term']:.4f} seg={full['seg_term']:.4f}", flush=True)
    print(f"  delta_vs_baseline={full['score']-base['score']:+.4f}", flush=True)
    print(f"  delta_vs_RGB_only=-0.0268: {full['score']-base['score']+0.0268:+.4f}", flush=True)

    csv_path = OUTPUT_DIR / "combined_full_results.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "n_mask_patches", "sb_mask", "sb_rgb", "sb_total",
                     "score", "pose_term", "delta_baseline", "delta_vs_rgb"])
        w.writerow(["mask_K1_top600+rgb_350K7_250K2", len(mask_patches),
                     sb_mask, sb_rgb, sb_mask + sb_rgb,
                     full['score'], full['pose_term'],
                     full['score'] - base['score'],
                     full['score'] - base['score'] + 0.0268])
    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
