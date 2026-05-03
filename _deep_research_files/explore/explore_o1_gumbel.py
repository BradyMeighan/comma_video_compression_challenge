#!/usr/bin/env python
"""
O1: End-to-end differentiable patch selection via Gumbel-Softmax + Sinkhorn-K.

Per Compass: "your per-pair K=7 saturation suggests that the discrete greedy is
finding only locally optimal supports. A continuous-then-discretize formulation,
optimized with Adam against the actual PoseNet loss for 50–100 steps and then
snapped to top-K, should consistently dominate the alternating greedy by 5–15%."

Method:
- For each pair: parameterize logits over all OUT_H × OUT_W positions
- Use Gumbel-softmax to sample K continuous-relaxed positions
- Optimize logits + channels + deltas END-to-end via PoseNet loss
- Anneal temperature → sharpen to discrete top-K
- Snap final positions, encode normally

Memory consideration: full position grid = 874*1164 = 1M positions per pair.
Need to be smart: maintain only a candidate POOL (top-256 by initial gradient)
and apply Gumbel-K within the pool.
"""
import sys, os, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, get_pose6, load_posenet, estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose)
from sidecar_channel_only import channel_sidecar_size, apply_channel_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_gumbel_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                          K, n_iter, device, pool_size=128, tau_init=2.0, tau_final=0.05):
    """Use Gumbel-softmax over a top-pool_size candidate pool to select K positions."""
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Initial gradient: pick TOP pool_size candidates (much larger than K=5-7)
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()
        max_chan_grad, best_chan = grad_abs.max(dim=1)  # (b, H, W)
        flat = max_chan_grad.contiguous().reshape(b, -1)
        _, pool = torch.topk(flat, pool_size, dim=1)  # (b, pool_size)
        pool_ys = (pool // OUT_W).long()
        pool_xs = (pool % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, pool_size)
        pool_chans = best_chan[batch_idx, pool_ys, pool_xs]  # (b, pool_size)

        # Gumbel-softmax over pool: select K
        # Logits shape (b, K, pool_size) — K independent slot selections
        logits = torch.zeros((b, K, pool_size), device=device, requires_grad=True)
        # Init logits with grad-magnitude-based prior to start near reasonable positions
        with torch.no_grad():
            grad_mag_in_pool = max_chan_grad[batch_idx, pool_ys, pool_xs]  # (b, pool_size)
            # Make first K slots prefer top-K, rest of slots get spread of next-best
            for k in range(K):
                logits[:, k, k::K] = 1.0  # rough init
        logits.requires_grad_(True)

        # Optimize: logits + deltas
        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([logits, cur_d], lr=0.1)

        for it in range(n_iter):
            tau = tau_init * (tau_final / tau_init) ** (it / max(1, n_iter - 1))
            opt.zero_grad()
            # Gumbel-softmax sample
            soft = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)  # (b, K, pool_size)

            # Soft positions: weighted avg of pool positions
            # x_soft = sum_p soft[b,k,p] * pool_xs[b,p]
            soft_x = (soft * pool_xs.unsqueeze(1).float()).sum(dim=-1)  # (b, K)
            soft_y = (soft * pool_ys.unsqueeze(1).float()).sum(dim=-1)  # (b, K)
            # For channels, take best_chan at the argmax position (use hard for channel)
            hard_idx = soft.argmax(dim=-1)  # (b, K)
            hard_chan = pool_chans.gather(1, hard_idx)  # (b, K)

            # Apply via differentiable bilinear splat
            f1_p = f1.clone()
            for c in range(3):
                mask_c = (hard_chan == c)
                if mask_c.any():
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    sx = soft_x[rows_b, cols_k]
                    sy = soft_y[rows_b, cols_k]
                    sd = cur_d[rows_b, cols_k]
                    x0 = torch.floor(sx).long().clamp(0, OUT_W - 1)
                    y0 = torch.floor(sy).long().clamp(0, OUT_H - 1)
                    x1 = (x0 + 1).clamp(0, OUT_W - 1)
                    y1 = (y0 + 1).clamp(0, OUT_H - 1)
                    wx1 = (sx - x0.float()).clamp(0, 1); wy1 = (sy - y0.float()).clamp(0, 1)
                    wx0 = 1 - wx1; wy0 = 1 - wy1
                    f1_p[rows_b, c, y0, x0] = f1_p[rows_b, c, y0, x0] + sd * wx0 * wy0
                    f1_p[rows_b, c, y0, x1] = f1_p[rows_b, c, y0, x1] + sd * wx1 * wy0
                    f1_p[rows_b, c, y1, x0] = f1_p[rows_b, c, y1, x0] + sd * wx0 * wy1
                    f1_p[rows_b, c, y1, x1] = f1_p[rows_b, c, y1, x1] + sd * wx1 * wy1
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss_p = ((fp - gt_p) ** 2).sum()
            loss_p.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)

        # SNAP: take argmax for each slot
        with torch.no_grad():
            hard_idx = logits.argmax(dim=-1)  # (b, K)
            final_xs = pool_xs.gather(1, hard_idx).cpu().numpy().astype(np.uint16)
            final_ys = pool_ys.gather(1, hard_idx).cpu().numpy().astype(np.uint16)
            final_chans = pool_chans.gather(1, hard_idx).cpu().numpy().astype(np.uint8)
            final_d = cur_d.detach().cpu().numpy().round().astype(np.int8)

        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(final_xs[bi].tolist(), final_ys[bi].tolist(),
                                final_chans[bi].tolist(), final_d[bi].tolist()))
            out[pair_i] = patches
    return out


def main():
    device = torch.device("cuda")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
    f1_new, f2_new = bf['f1_new'], bf['f2_new']
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    sb_mask = bp['sb_mask_bz2']
    score_bl = bp['score']

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}, sb_mask={sb_mask}B")
    print("\n=== O1: Gumbel-Softmax differentiable patch selection ===")
    t0 = time.time()
    p_top = find_gumbel_patches(f1_new, f2_new, poses, posenet,
                                  [int(x) for x in rank[:250]], K=5, n_iter=100, device=device)
    p_tail = find_gumbel_patches(f1_new, f2_new, poses, posenet,
                                   [int(x) for x in rank[250:500]], K=2, n_iter=100, device=device)
    rgb_patches = {**p_top, **p_tail}
    elapsed = time.time() - t0

    sb_rgb = channel_sidecar_size(rgb_patches)
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"O1: sb_total={sb_mask+sb_rgb}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
          f"delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "o1_gumbel_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "pose_term", "delta"])
        w.writerow(["o1_gumbel_250_K5+250_K2", sb_mask+sb_rgb, full['score'], full['pose_term'],
                    full['score']-score_bl])


if __name__ == "__main__":
    main()
