#!/usr/bin/env python
"""
X1: Combined sub-pixel + 3-channel-best scoring (the actual Compass Rank 2).

Compass said both should be combined: sub-pixel positions PLUS evaluating all 3
channels per candidate. This combines a1 + a2.
"""
import sys, os, pickle, time, struct, bz2
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

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_subpixel_3chan_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                                  K, n_iter, device):
    """Sub-pixel positions + all-3-channel scoring per position."""
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()

        # Pick top-K by SUM-channel grad (let optimizer pick channel)
        flat = grad_abs.sum(dim=1).contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_int = (topk // OUT_W).long()
        xs_int = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)

        # Optimize all 3 channels with sub-pixel offsets
        x_off = torch.zeros((b, K), device=device, requires_grad=True)
        y_off = torch.zeros((b, K), device=device, requires_grad=True)
        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([x_off, y_off, cur_d], lr=1.0)

        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            xc = xs_int.float() + x_off; yc = ys_int.float() + y_off
            x0 = torch.floor(xc).long().clamp(0, OUT_W - 1)
            y0 = torch.floor(yc).long().clamp(0, OUT_H - 1)
            x1 = (x0 + 1).clamp(0, OUT_W - 1); y1 = (y0 + 1).clamp(0, OUT_H - 1)
            wx1 = (xc - x0.float()).clamp(0, 1); wy1 = (yc - y0.float()).clamp(0, 1)
            wx0 = 1 - wx1; wy0 = 1 - wy1
            for c in range(3):
                d_c = cur_d[..., c]
                f1_p[batch_idx, c, y0, x0] = f1_p[batch_idx, c, y0, x0] + d_c * wx0 * wy0
                f1_p[batch_idx, c, y0, x1] = f1_p[batch_idx, c, y0, x1] + d_c * wx1 * wy0
                f1_p[batch_idx, c, y1, x0] = f1_p[batch_idx, c, y1, x0] + d_c * wx0 * wy1
                f1_p[batch_idx, c, y1, x1] = f1_p[batch_idx, c, y1, x1] + d_c * wx1 * wy1
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss_p = ((fp - gt_p) ** 2).sum()
            loss_p.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127); x_off.clamp_(-0.5, 0.5); y_off.clamp_(-0.5, 0.5)

        # Per (b, k): pick channel with biggest |delta|
        d_abs = cur_d.detach().abs()
        best_chan = d_abs.argmax(dim=-1)  # (b, K)
        d_chosen = torch.zeros((b, K), device=device)
        for c in range(3):
            mask_c = (best_chan == c)
            d_chosen[mask_c] = cur_d.detach()[..., c][mask_c]

        x_frac = ((x_off.detach() + 0.5) * 16).round().clamp(0, 15).long()
        y_frac = ((y_off.detach() + 0.5) * 16).round().clamp(0, 15).long()
        xs_np = xs_int.cpu().numpy().astype(np.uint16)
        ys_np = ys_int.cpu().numpy().astype(np.uint16)
        xf_np = x_frac.cpu().numpy().astype(np.uint8)
        yf_np = y_frac.cpu().numpy().astype(np.uint8)
        chan_np = best_chan.cpu().numpy().astype(np.uint8)
        d_np = d_chosen.cpu().numpy().round().astype(np.int8)

        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                xf_np[bi].tolist(), yf_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
            out[pair_i] = patches
    return out


def subpixel_sidecar_size(patches):
    """8 bytes per patch."""
    if not patches:
        return 0
    parts = [struct.pack("<H", len(patches))]
    for pi in sorted(patches.keys()):
        ps = patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, xf, yf, c, d) in ps:
            packed_frac = ((xf & 0xF) << 4) | (yf & 0xF)
            parts.append(struct.pack("<HHBBb", x, y, packed_frac, c, d))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def apply_subpixel_patches(f_all, patches):
    out = f_all.clone()
    H, W = out.shape[1], out.shape[2]
    for pair_i in patches:
        arr = out[pair_i].float().numpy()
        for (x, y, xf, yf, c, d) in patches[pair_i]:
            x_off = (xf / 16.0) - 0.5; y_off = (yf / 16.0) - 0.5
            xc = x + x_off; yc = y + y_off
            x0 = max(0, min(int(np.floor(xc)), W-1)); y0 = max(0, min(int(np.floor(yc)), H-1))
            x1 = min(x0 + 1, W-1); y1 = min(y0 + 1, H-1)
            wx1 = max(0, min(xc - x0, 1)); wy1 = max(0, min(yc - y0, 1))
            wx0 = 1 - wx1; wy0 = 1 - wy1
            arr[y0, x0, c] += d * wx0 * wy0
            arr[y0, x1, c] += d * wx1 * wy0
            arr[y1, x0, c] += d * wx0 * wy1
            arr[y1, x1, c] += d * wx1 * wy1
        out[pair_i] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
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
    print("\n=== X1: sub-pixel + 3-channel-best 250_K5+250_K2 ===")
    t0 = time.time()
    p_top = find_subpixel_3chan_patches(f1_new, f2_new, poses, posenet,
                                          [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_subpixel_3chan_patches(f1_new, f2_new, poses, posenet,
                                            [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = subpixel_sidecar_size(rgb_patches)
    f1_combined = apply_subpixel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    elapsed = time.time() - t0
    print(f"X1: sb_total={sb_mask+sb_rgb}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
          f"delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "x1_subpixel_3chan_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "pose_term", "delta"])
        w.writerow(["x1_subpixel_3chan_250_K5+250_K2", sb_mask+sb_rgb, full['score'], full['pose_term'],
                    full['score']-score_bl])


if __name__ == "__main__":
    main()
