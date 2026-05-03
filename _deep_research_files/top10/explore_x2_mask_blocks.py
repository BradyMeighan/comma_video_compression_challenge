#!/usr/bin/env python
"""
X2: Multi-pixel mask BLOCK flips (2x2 mask region as one unit).

Idea: instead of flipping 1 mask pixel, flip a 2x2 block to a SINGLE class.
- 4× the receptive field impact per byte
- Storage: u16 x, u16 y, u8 class = 5 bytes/block (same as 1-pixel mask flip)

Verified greedy: try top-N candidate (x,y) positions × 5 candidate classes.
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      estimate_model_bytes)
from train import Generator, load_data_full, coords
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose)
from sidecar_mask_verified import (mask_sidecar_size, regenerate_frames_from_masks,
                                     gen_forward_with_oh_mask, pose_loss_for_pair)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def verified_greedy_block_mask(gen, m_init, p, gt_p, posenet, device, K, n_candidates=20, block=2):
    """Find K block-flip mask patches per pair via verified greedy."""
    cur_m = m_init.clone()
    accepted = []
    for k_iter in range(K):
        m_oh = F.one_hot(cur_m, num_classes=5).float().requires_grad_(True)
        f1u, f2u = gen_forward_with_oh_mask(gen, m_oh, p, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        baseline_loss = loss.item()
        grad = torch.autograd.grad(loss, m_oh)[0]

        # Pool gradient over block_size
        # For each (y, x) block top-left, sum |grad| over the block
        cur_class = cur_m
        grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
        # delta to flip to class c: grad[..., c] - grad_cur
        # Block score: for block at (y, x), sum over block of best alt class delta
        candidate_delta = grad - grad_cur.unsqueeze(-1)  # (1, H, W, 5)
        for cls in range(5):
            candidate_delta[..., cls][cur_class == cls] = float('inf')
        best_delta, best_class = candidate_delta.min(dim=-1)  # (1, H, W)

        # Pool over 2x2 blocks: sum negative deltas (negative = improvement)
        neg_delta = (-best_delta).clamp_min(0)  # treat negatives as zeros
        pooled = F.avg_pool2d(neg_delta.unsqueeze(0), kernel_size=block, stride=1) * (block * block)
        pooled = pooled.squeeze(0)  # (1, H-1, W-1)

        # Exclude positions already used
        for (x, y, _) in accepted:
            for dy in range(block):
                for dx in range(block):
                    yy = y + dy; xx = x + dx
                    if yy < pooled.shape[1] and xx < pooled.shape[2]:
                        pooled[0, yy, xx] = 0

        flat = pooled.contiguous().reshape(-1)
        topk_vals, topk_idx = torch.topk(flat, n_candidates)
        H_p = pooled.shape[1]; W_p = pooled.shape[2]
        cand_ys = (topk_idx // W_p).long().cpu().numpy()
        cand_xs = (topk_idx % W_p).long().cpu().numpy()

        # Verify each candidate: also pick the best class (from best_class for top-left, but block can be heterogeneous)
        # Strategy: try each class for the WHOLE block (5 classes × n_candidates evaluations)
        best_actual = float('inf'); best_choice = None
        for k in range(n_candidates):
            yy = int(cand_ys[k]); xx = int(cand_xs[k])
            for new_cls in range(5):
                test_m = cur_m.clone()
                # Set 2x2 block to new_cls
                test_m[0, yy:yy+block, xx:xx+block] = new_cls
                test_oh = F.one_hot(test_m, num_classes=5).float()
                with torch.no_grad():
                    new_loss = pose_loss_for_pair(gen, test_oh, p, gt_p, posenet, device)
                actual_delta = new_loss - baseline_loss
                if actual_delta < best_actual:
                    best_actual = actual_delta
                    best_choice = (xx, yy, new_cls)

        if best_choice is None or best_actual >= 0:
            break
        x, y, new_cls = best_choice
        cur_m[0, y:y+block, x:x+block] = new_cls
        accepted.append((x, y, new_cls))
    return accepted, cur_m


def block_mask_sidecar_size(mask_block_patches):
    """5 bytes per block patch."""
    if not mask_block_patches:
        return 0
    parts = [struct.pack("<H", len(mask_block_patches))]
    for pi in sorted(mask_block_patches.keys()):
        ps = mask_block_patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c) in ps:
            parts.append(struct.pack("<HHB", x, y, c))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


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

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}")
    print("\n=== X2: 2x2 block mask flips, K=1, top600 ===")
    t0 = time.time()
    block_mask_patches = {}
    for i, pi in enumerate(rank[:600]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_block_mask(gen, m, p, gt_p, posenet, device,
                                                    K=1, n_candidates=10, block=2)
        if accepted:
            block_mask_patches[pi] = accepted
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/600 ({time.time()-t0:.0f}s)", flush=True)
    sb_mask = block_mask_sidecar_size(block_mask_patches)
    print(f"Mask block: {len(block_mask_patches)} pairs, sb={sb_mask}B")

    # Apply + regen
    new_masks = masks_cpu.clone()
    for pi, ps in block_mask_patches.items():
        for (x, y, c) in ps:
            new_masks[pi, y:y+2, x:x+2] = c
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
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"X2: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} delta={full['score']-score_bl:+.4f} ({time.time()-t0:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "x2_mask_blocks_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_mask", "sb_rgb", "sb_total", "score", "delta"])
        w.writerow(["x2_2x2_blocks_K1_top600", sb_mask, sb_rgb, sb_mask+sb_rgb,
                    full['score'], full['score']-score_bl])


if __name__ == "__main__":
    main()
