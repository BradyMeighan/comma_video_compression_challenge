#!/usr/bin/env python
"""
O4: ADMM-style joint mask + RGB optimization.

Currently we apply mask, regenerate, then find RGB. ADMM alternates:
- Fix RGB → optimize mask flips (gradient + verify on RGB-applied output)
- Fix mask → optimize RGB (standard)
- Repeat with consensus penalty pushing toward equilibrium.

Simplified version: just do 2 alternating rounds and see if it improves.
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
                            find_pose_patches_for_pairs, per_pair_pose_mse)
from sidecar_mask_verified import (verified_greedy_mask, mask_sidecar_size,
                                     regenerate_frames_from_masks,
                                     gen_forward_with_oh_mask, pose_loss_for_pair)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def main():
    device = torch.device("cuda")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    score_bl = bp['score']
    sb_mask_bl = bp['sb_mask_bz2']

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}")

    # Start from baseline mask + RGB
    mask_patches = dict(bp['mask_patches'])
    rgb_patches = dict(bp['rgb_patches'])

    print("\n=== O4: ADMM-style 3 rounds of alternating optimization ===")
    t0 = time.time()
    for round_i in range(3):
        # Apply current state
        new_masks = masks_cpu.clone()
        for pi, ps in mask_patches.items():
            for (x, y, c) in ps:
                new_masks[pi, y, x] = c
        f1_cur, f2_cur = regenerate_frames_from_masks(gen, new_masks, poses, device)

        # Round step 1: re-find RGB on current state
        print(f"\nRound {round_i+1}: re-finding RGB...")
        p_top = find_channel_only_patches(f1_cur, f2_cur, poses, posenet,
                                            [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
        p_tail = find_channel_only_patches(f1_cur, f2_cur, poses, posenet,
                                             [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
        rgb_patches = {**p_top, **p_tail}
        f1_with_rgb = apply_channel_patches(f1_cur, rgb_patches)
        # Eval
        s, p = fast_eval(f1_with_rgb, f2_cur, data["val_rgb"], device)
        sb_mask_cur = mask_sidecar_size(mask_patches)
        sb_rgb_cur = channel_sidecar_size(rgb_patches)
        full = fast_compose(s, p, model_bytes, sb_mask_cur + sb_rgb_cur)
        print(f"  After RGB: score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"sb_total={sb_mask_cur+sb_rgb_cur}B")

        if round_i == 2:
            break  # Last round, don't add more mask flips

        # Round step 2: find ADDITIONAL mask flips, considering RGB is applied
        # (The mask gradient is computed on f1 = gen(modified_mask), then RGB applied,
        #  then loss measured. Since mask change → gen output change → RGB patch positions
        #  may not be optimal, but we just look for incremental improvement.)
        print(f"Round {round_i+1}: finding additional mask flips...")
        n_added = 0
        for i, pi in enumerate(rank[:200]):
            pi = int(pi)
            m = new_masks[pi:pi+1].to(device).long()
            p_in = poses[pi:pi+1].to(device).float()
            gt_p = p_in.clone()
            accepted, _ = verified_greedy_mask(gen, m, p_in, gt_p, posenet, device, K=1, n_candidates=8)
            if accepted:
                if pi in mask_patches:
                    mask_patches[pi].extend(accepted)
                else:
                    mask_patches[pi] = accepted
                n_added += len(accepted)
        print(f"  Added {n_added} mask flips, mask now {len(mask_patches)} pairs")

    sb_mask_final = mask_sidecar_size(mask_patches)
    sb_rgb_final = channel_sidecar_size(rgb_patches)
    print(f"\nO4 final: sb_mask={sb_mask_final}B sb_rgb={sb_rgb_final}B "
          f"sb_total={sb_mask_final+sb_rgb_final}B score={full['score']:.4f} "
          f"delta={full['score']-score_bl:+.4f} ({time.time()-t0:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "o4_admm_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_mask", "sb_rgb", "sb_total", "score", "delta"])
        w.writerow(["o4_admm_3rounds", sb_mask_final, sb_rgb_final,
                    sb_mask_final + sb_rgb_final, full['score'], full['score']-score_bl])


if __name__ == "__main__":
    main()
