#!/usr/bin/env python
"""
O3: Adversarial Hessian eigenvector targeting.

Compute top eigenvector of PoseNet Hessian w.r.t. f1 (per pair).
Place patch deltas along this direction (where the network is MOST sensitive).
The hypothesis: the pose loss is locally well-modeled by quadratic, and the top
eigenvector indicates the maximum-variance perturbation direction.

Method: power iteration to estimate top eigenvector. Project patch deltas onto it.
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


def find_hessian_aligned_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                                    K, n_iter, device, n_power=3):
    """For each pair, estimate top Hessian eigenvector via power iteration,
    pick top-K positions where eigenvector magnitude is largest."""
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # First-order gradient: gives optimal direction for L1 attack
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad1 = torch.autograd.grad(loss, f1_param, create_graph=True)[0]

        # Power iteration: v = H v / ||H v||
        v = grad1.detach().clone()
        v = v / v.flatten(1).norm(dim=1, keepdim=True).view(b, 1, 1, 1)
        for _ in range(n_power):
            # Compute Hv = grad of (grad . v) w.r.t. f1
            Hv = torch.autograd.grad(grad1, f1_param, grad_outputs=v, retain_graph=True)[0]
            v = Hv.detach()
            norm = v.flatten(1).norm(dim=1, keepdim=True).view(b, 1, 1, 1).clamp_min(1e-12)
            v = v / norm

        # Pick top-K positions by |v|.sum_over_channels
        v_abs = v.abs().sum(dim=1)  # (b, H, W)
        max_chan_v, best_chan = v.abs().max(dim=1)
        flat = max_chan_v.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        chan_t = best_chan[batch_idx, ys_t, xs_t]

        # Optimize 1-channel deltas at these positions
        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            for c in range(3):
                mask_c = (chan_t == c)
                if mask_c.any():
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    yy = ys_t[rows_b, cols_k]; xx = xs_t[rows_b, cols_k]
                    dd = cur_d[rows_b, cols_k]
                    f1_p[rows_b, c, yy, xx] = f1_p[rows_b, c, yy, xx] + dd
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss_p = ((fp - gt_p) ** 2).sum()
            loss_p.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)

        xs_np = xs_t.cpu().numpy().astype(np.uint16)
        ys_np = ys_t.cpu().numpy().astype(np.uint16)
        chan_np = chan_t.cpu().numpy().astype(np.uint8)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
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
    print("\n=== O3: Hessian eigenvector targeting 250_K5+250_K2 ===")
    t0 = time.time()
    p_top = find_hessian_aligned_patches(f1_new, f2_new, poses, posenet,
                                            [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_hessian_aligned_patches(f1_new, f2_new, poses, posenet,
                                             [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = channel_sidecar_size(rgb_patches)
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    elapsed = time.time() - t0
    print(f"O3: sb_total={sb_mask+sb_rgb}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
          f"delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "o3_hessian_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "pose_term", "delta"])
        w.writerow(["o3_hessian_250_K5+250_K2", sb_mask+sb_rgb, full['score'], full['pose_term'],
                    full['score']-score_bl])


if __name__ == "__main__":
    main()
