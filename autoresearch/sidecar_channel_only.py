#!/usr/bin/env python
"""
CHANNEL-ONLY RGB patches: each patch modifies ONE color channel.

Format: u16 x, u16 y, u8 channel_id (0-2), i8 delta = 6 bytes/patch (vs 7B for full RGB).
~14% byte savings per patch.

Method: gradient through PoseNet has 3 channels per pixel. For each top-K pixel,
pick the channel with biggest |grad|, optimize delta for just that channel.
"""
import sys, os, time, csv, struct, bz2
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
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_mask_verified import (verified_greedy_mask, mask_sidecar_size,
                                     regenerate_frames_from_masks)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_channel_only_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                                K, n_iter, device):
    """Find K patches per pair, each modifying ONLY one channel."""
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Initial gradient through posenet
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]  # (b, 3, H, W)
        grad_abs = grad.abs()
        # For each pixel, pick the channel with biggest |grad|
        max_chan_grad, best_chan = grad_abs.max(dim=1)  # (b, H, W)
        # Top-K positions by max-channel grad
        flat = max_chan_grad.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()
        # Get the best channel for each top-K position
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        chan_t = best_chan[batch_idx, ys_t, xs_t]  # (b, K)

        # Optimize one delta per (pos, channel)
        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            for c in range(3):
                # Mask: at positions where chan_t == c, add delta
                mask_c = (chan_t == c)
                if mask_c.any():
                    # Get coords + deltas where channel is c
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    yy = ys_t[rows_b, cols_k]
                    xx = xs_t[rows_b, cols_k]
                    dd = cur_d[rows_b, cols_k]
                    f1_p[rows_b, c, yy, xx] = f1_p[rows_b, c, yy, xx] + dd
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            loss.backward()
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


def channel_sidecar_size(patches):
    if not patches:
        return 0
    parts = [struct.pack("<H", len(patches))]
    for pair_i, ps in sorted(patches.items()):
        parts.append(struct.pack("<HH", pair_i, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def apply_channel_patches(f_all, patches):
    out = f_all.clone()
    H, W = out.shape[1], out.shape[2]
    for pair_i, ps in patches.items():
        arr = out[pair_i].float().numpy()
        for (x, y, c, d) in ps:
            arr[y, x, c] += d
        out[pair_i] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


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

    # Build mask K=1 top600 base
    print("\n=== Building mask K=1 top600 base ===", flush=True)
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
    sb_mask = mask_sidecar_size(mask_patches)
    print(f"Mask: {len(mask_patches)} pairs, sb={sb_mask}B", flush=True)

    new_masks = masks_cpu.clone()
    for pi, patches in mask_patches.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)

    csv_path = OUTPUT_DIR / "channel_only_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "K_total", "sb_rgb", "sb_total",
                                 "score", "pose_term", "delta", "d_vs_best", "elapsed"])

    BEST = -0.0287

    # Test channel-only RGB at various K
    for spec, tiers in [
        ("ch_200_K6+200_K3", [(0, 200, 6), (200, 400, 3)]),  # 6,3 vs 5,2 to compensate for less expressiveness
        ("ch_250_K6+250_K3", [(0, 250, 6), (250, 500, 3)]),
        ("ch_200_K5+200_K2", [(0, 200, 5), (200, 400, 2)]),
        ("ch_250_K5+250_K2", [(0, 250, 5), (250, 500, 2)]),
        ("ch_300_K7+300_K3", [(0, 300, 7), (300, 600, 3)]),
    ]:
        print(f"\n=== {spec} (channel-only RGB) ===", flush=True)
        t1 = time.time()
        rgb_patches = {}
        for start, end, K in tiers:
            pairs = [int(x) for x in rank[start:end]]
            ps = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                              pairs, K=K, n_iter=80, device=device)
            rgb_patches.update(ps)
        sb_rgb = channel_sidecar_size(rgb_patches)
        f1_combined = apply_channel_patches(f1_new, rgb_patches)
        s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
        full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
        delta = full['score'] - base['score']
        d_vs_best = delta - BEST
        K_total = sum(len(v) for v in rgb_patches.values())
        elapsed = time.time() - t1
        print(f"  >> {spec}: K_total={K_total} sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} d_vs_best={d_vs_best:+.4f} ({elapsed:.0f}s)", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, K_total, sb_rgb, sb_mask + sb_rgb,
                                     full['score'], full['pose_term'], delta, d_vs_best, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
