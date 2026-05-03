#!/usr/bin/env python
"""Build the BEST baseline sidecar (mask K=1 top600 + channel-only RGB 250_K5+250_K2)
ONCE and save patches to disk. Other experiments load and modify rather than rebuilding.

Saves:
  autoresearch/sidecar_results/baseline_patches.pkl  -- dict with mask_patches, rgb_patches, baseline_score
  autoresearch/sidecar_results/baseline_frames.pt    -- f1_new, f2_new (post-mask-regenerate)
"""
import sys, os, time, pickle
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
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                    apply_channel_patches)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    print(f"Baseline (no sidecar): score={base['score']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    # Step 1: mask K=1 top600 verified greedy
    print("\n=== Step 1: mask K=1 top600 verified greedy ===", flush=True)
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
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/600 ({time.time()-t0:.0f}s)", flush=True)
    print(f"Mask: {len(mask_patches)} pairs, {time.time()-t0:.0f}s", flush=True)

    # Apply mask + regenerate
    new_masks = masks_cpu.clone()
    for pi, patches in mask_patches.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)

    # Step 2: channel-only RGB 250_K5+250_K2 on patched frames
    print("\n=== Step 2: channel-only RGB on regenerated frames ===", flush=True)
    t1 = time.time()
    p_top = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                         [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    print(f"RGB: {len(rgb_patches)} pairs, {time.time()-t1:.0f}s", flush=True)

    # Eval current baseline
    sb_mask = mask_sidecar_size(mask_patches)
    sb_rgb = channel_sidecar_size(rgb_patches)
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"\n=== Current baseline ===")
    print(f"sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B")
    print(f"score={full['score']:.4f} pose={full['pose_term']:.4f}")
    print(f"delta_vs_no_sidecar={full['score']-base['score']:+.4f}")

    # Save everything
    out = {
        'mask_patches': mask_patches,
        'rgb_patches': rgb_patches,
        'sb_mask_bz2': sb_mask,
        'sb_rgb_bz2': sb_rgb,
        'sb_total_bz2': sb_mask + sb_rgb,
        'score': full['score'],
        'seg_dist': s,
        'pose_dist': p,
        'model_bytes': model_bytes,
        'baseline_no_sidecar_score': base['score'],
    }
    pkl_path = OUTPUT_DIR / "baseline_patches.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved patches to {pkl_path}")
    # Save the mask-patched frames so other experiments don't have to regen
    torch.save({'f1_new': f1_new, 'f2_new': f2_new, 'new_masks': new_masks},
                OUTPUT_DIR / "baseline_frames.pt")
    print(f"Saved frames to {OUTPUT_DIR / 'baseline_frames.pt'}")


if __name__ == "__main__":
    main()
