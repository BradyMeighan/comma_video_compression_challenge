#!/usr/bin/env python
"""
A1: Sub-pixel coordinates with bilinear splat at decode.

Storage: u16 x_int + u16 y_int + u4 x_frac + u4 y_frac (1B for fracs) + u8 channel + i8 delta
       = 8 bytes/patch (vs 6B for integer-only). Tradeoff: 33% MORE bytes per patch but
         each patch can hit sub-pixel positions, more impactful at PoseNet's 384x512 input.

Decode: instead of arr[y, x] += d, distribute d across 4 neighbor pixels with bilinear weights.
Encode: optimize position as continuous float, quantize fractional part to 4 bits each.

This script trains patches in continuous coord space, then quantizes.
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
from sidecar_mask_verified import mask_sidecar_size

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_subpixel_channel_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                                    K, n_iter, device):
    """Find K patches per pair with sub-pixel positions and channel selection.
    Each patch: (x_float, y_float, channel, delta).
    """
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Initial gradient to find top-K positions + dominant channels
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()
        max_chan_grad, best_chan = grad_abs.max(dim=1)
        flat = max_chan_grad.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_int = (topk // OUT_W).long()
        xs_int = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        chan_t = best_chan[batch_idx, ys_int, xs_int]

        # Optimize CONTINUOUS sub-pixel offsets in [-0.5, 0.5] + delta
        x_off = torch.zeros((b, K), device=device, requires_grad=True)
        y_off = torch.zeros((b, K), device=device, requires_grad=True)
        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([x_off, y_off, cur_d], lr=1.0)

        def apply_bilinear_splat(f1_base, x_int, y_int, x_off, y_off, chan, delta):
            """Apply delta at (x_int + x_off, y_int + y_off) via bilinear splat to 4 neighbors."""
            f1_p = f1_base.clone()
            # Continuous position
            xc = x_int.float() + x_off
            yc = y_int.float() + y_off
            # 4 neighbors
            x0 = torch.floor(xc).long().clamp(0, OUT_W - 1)
            y0 = torch.floor(yc).long().clamp(0, OUT_H - 1)
            x1 = (x0 + 1).clamp(0, OUT_W - 1)
            y1 = (y0 + 1).clamp(0, OUT_H - 1)
            wx1 = (xc - x0.float()).clamp(0, 1)
            wy1 = (yc - y0.float()).clamp(0, 1)
            wx0 = 1 - wx1
            wy0 = 1 - wy1
            for c_test in range(3):
                mask_c = (chan == c_test)
                if mask_c.any():
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    d_use = delta[rows_b, cols_k]
                    yy0 = y0[rows_b, cols_k]; yy1 = y1[rows_b, cols_k]
                    xx0 = x0[rows_b, cols_k]; xx1 = x1[rows_b, cols_k]
                    w00 = wx0[rows_b, cols_k] * wy0[rows_b, cols_k]
                    w01 = wx1[rows_b, cols_k] * wy0[rows_b, cols_k]
                    w10 = wx0[rows_b, cols_k] * wy1[rows_b, cols_k]
                    w11 = wx1[rows_b, cols_k] * wy1[rows_b, cols_k]
                    f1_p[rows_b, c_test, yy0, xx0] = f1_p[rows_b, c_test, yy0, xx0] + d_use * w00
                    f1_p[rows_b, c_test, yy0, xx1] = f1_p[rows_b, c_test, yy0, xx1] + d_use * w01
                    f1_p[rows_b, c_test, yy1, xx0] = f1_p[rows_b, c_test, yy1, xx0] + d_use * w10
                    f1_p[rows_b, c_test, yy1, xx1] = f1_p[rows_b, c_test, yy1, xx1] + d_use * w11
            return f1_p.clamp(0, 255)

        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = apply_bilinear_splat(f1, xs_int, ys_int, x_off, y_off, chan_t, cur_d)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)
                x_off.clamp_(-0.5, 0.5)
                y_off.clamp_(-0.5, 0.5)

        # Quantize fractional offsets to 4 bits each (16 levels per axis)
        # x_frac, y_frac in [0, 15] representing (offset + 0.5) * 16
        x_frac = ((x_off.detach() + 0.5) * 16).round().clamp(0, 15).long()
        y_frac = ((y_off.detach() + 0.5) * 16).round().clamp(0, 15).long()
        # Final coordinates after rounding (will become x_int + (x_frac/16 - 0.5) at decode)

        xs_np = xs_int.cpu().numpy().astype(np.uint16)
        ys_np = ys_int.cpu().numpy().astype(np.uint16)
        xf_np = x_frac.cpu().numpy().astype(np.uint8)
        yf_np = y_frac.cpu().numpy().astype(np.uint8)
        chan_np = chan_t.cpu().numpy().astype(np.uint8)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                xf_np[bi].tolist(), yf_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
            out[pair_i] = patches
    return out


def subpixel_sidecar_size(patches):
    """8 bytes per patch: u16 x, u16 y, u8 (xfrac<<4|yfrac), u8 channel, i8 delta = 7B."""
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
    """Apply with bilinear splat at decode."""
    out = f_all.clone()
    H, W = out.shape[1], out.shape[2]
    for pair_i in patches:
        arr = out[pair_i].float().numpy()
        for (x, y, xf, yf, c, d) in patches[pair_i]:
            x_off = (xf / 16.0) - 0.5
            y_off = (yf / 16.0) - 0.5
            xc = x + x_off; yc = y + y_off
            x0 = int(np.floor(xc)); y0 = int(np.floor(yc))
            x0 = max(0, min(x0, W-1)); y0 = max(0, min(y0, H-1))
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
    print("Loading...", flush=True)
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
    print("\n=== A1: sub-pixel + 3-channel-best 250_K5+250_K2 ===")
    t0 = __import__('time').time()
    p_top = find_subpixel_channel_patches(f1_new, f2_new, poses, posenet,
                                            [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_subpixel_channel_patches(f1_new, f2_new, poses, posenet,
                                             [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    elapsed = __import__('time').time() - t0

    sb_rgb = subpixel_sidecar_size(rgb_patches)
    f1_combined = apply_subpixel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"A1: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
          f"delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "a1_subpixel_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "pose_term", "delta_vs_baseline"])
        w.writerow(["baseline", sb_mask + bp['sb_rgb_bz2'], score_bl, 0, 0])
        w.writerow(["a1_subpixel_250_K5+250_K2", sb_mask+sb_rgb, full['score'], full['pose_term'],
                    full['score']-score_bl])


if __name__ == "__main__":
    main()
