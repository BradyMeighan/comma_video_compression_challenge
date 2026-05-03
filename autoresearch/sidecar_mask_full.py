#!/usr/bin/env python
"""
Full mask sidecar test: verified greedy K=1 across ALL 600 pairs.
Test whether mask alone can beat RGB at far fewer bytes.

Plus: try K=3 on hardest 300 + K=1 on tail 300 (tiered mask).
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
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    csv_path = OUTPUT_DIR / "mask_full_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_searched", "n_accepted", "K_total",
                                 "sidecar_bytes", "score", "pose_term", "delta", "elapsed"])

    def search_mask(spec, pair_indices_K_pairs):
        """pair_indices_K_pairs: list of (pair_indices, K) tuples"""
        t0 = time.time()
        mask_patches = {}
        n_searched_total = 0
        for pair_indices, K in pair_indices_K_pairs:
            for i, pi in enumerate(pair_indices):
                pi = int(pi)
                m = masks_cpu[pi:pi+1].to(device).long()
                p = poses[pi:pi+1].to(device).float()
                gt_p = p.clone()
                accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device, K=K, n_candidates=10)
                if accepted:
                    if pi in mask_patches:
                        mask_patches[pi].extend(accepted)
                    else:
                        mask_patches[pi] = accepted
            n_searched_total += len(pair_indices)
            print(f"  ... tier with {len(pair_indices)} pairs at K={K} done ({time.time()-t0:.1f}s)", flush=True)

        new_masks = masks_cpu.clone()
        for pi, patches in mask_patches.items():
            for (x, y, c) in patches:
                new_masks[pi, y, x] = c
        f1_p, f2_p = regenerate_frames_from_masks(gen, new_masks, poses, device)
        sb = mask_sidecar_size(mask_patches)
        s, p = fast_eval(f1_p, f2_p, data["val_rgb"], device)
        full = fast_compose(s, p, model_bytes, sb)
        delta = full['score'] - base['score']
        elapsed = time.time() - t0
        K_total = sum(len(v) for v in mask_patches.values())
        print(f"  >> {spec}: searched={n_searched_total} accepted={len(mask_patches)} "
              f"K_total={K_total} sb={sb}B score={full['score']:.4f} "
              f"pose={full['pose_term']:.4f} delta={delta:+.4f} ({elapsed:.1f}s)", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, n_searched_total, len(mask_patches), K_total,
                                     sb, full['score'], full['pose_term'], delta, elapsed])
        return mask_patches, f1_p, f2_p, sb, full

    # ─── Test 1: mask K=1 across all 600 pairs ───
    print("\n=== Test 1: mask K=1 top600 ===", flush=True)
    mp1, _, _, _, _ = search_mask("mask_K1_top600", [(rank[:600].tolist(), 1)])

    # ─── Test 2: mask K=3 across all 600 pairs ───
    print("\n=== Test 2: mask K=3 top600 ===", flush=True)
    mp2, _, _, _, _ = search_mask("mask_K3_top600", [(rank[:600].tolist(), 3)])

    # ─── Test 3: tiered mask (K=3 hardest + K=1 tail) ───
    print("\n=== Test 3: tiered mask K=3 top200 + K=1 next400 ===", flush=True)
    mp3, _, _, _, _ = search_mask("mask_tiered_K3_200+K1_400",
                                    [(rank[:200].tolist(), 3), (rank[200:600].tolist(), 1)])

    # ─── Test 4: combine BEST mask + RGB ───
    print("\n=== Test 4: best mask + RGB(350_K7+250_K2) ===", flush=True)
    # Use mp2 (K=3_top600) as the strongest mask sidecar
    new_masks = masks_cpu.clone()
    for pi, patches in mp2.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)
    t0 = time.time()
    p_top = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = sparse_sidecar_size(rgb_patches)
    sb_mask = mask_sidecar_size(mp2)
    f1_combined = apply_sparse_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    delta = full['score'] - base['score']
    elapsed = time.time() - t0
    print(f"  >> combined: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} pose={full['pose_term']:.4f} delta={delta:+.4f} "
          f"({elapsed:.1f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["mask_K3_top600+rgb_350K7_250K2", 600,
                                 len(mp2), sum(len(v) for v in mp2.values()),
                                 sb_mask + sb_rgb, full['score'], full['pose_term'],
                                 delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
