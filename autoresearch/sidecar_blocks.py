#!/usr/bin/env python
"""
Test BLOCK patches: a single (x, y, dr, dg, db) entry but applied as a NxN
block at decode time. Same byte cost as a single-pixel patch, but covers
more area at MODEL resolution after bilinear downsample.

Math: OUT (874, 1164) → MODEL (384, 512). Scale: ~2.27. So a 1-pixel patch
at OUT covers ~0.44 model pixels (sub-pixel, often interpolated to ~0.2 max).
A 3x3 block covers ~1.3 model pixels — much stronger signal.

Block size 1 = current best. Test 2, 3, 4, 5.
Also: smaller K with bigger block to keep similar pose reduction at fewer bytes.
"""
import sys, os, time, csv
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
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_block_pose_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                              K, n_iter, device, block_size=1):
    """Find K patches per pair, each applied as block_size x block_size square."""
    out = {}
    bs = 8
    half = block_size // 2
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Initial gradient — pool the abs gradient over block_size x block_size
        # so we pick block centers, not single pixels
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        gmag = grad.abs().sum(dim=1).unsqueeze(1)  # (b, 1, H, W)

        if block_size > 1:
            # Pool gradient magnitudes over block_size x block_size to score block centers
            pooled = F.avg_pool2d(gmag, kernel_size=block_size, stride=1, padding=half) * (block_size * block_size)
            # Trim back to original H, W if needed
            pooled = pooled[:, :, :OUT_H, :OUT_W]
            scoring = pooled.squeeze(1)
        else:
            scoring = gmag.squeeze(1)

        flat = scoring.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)

        # Precompute block offsets
        if block_size == 1:
            offsets = [(0, 0)]
        else:
            tail = block_size - half
            offsets = [(dy, dx) for dy in range(-half, tail) for dx in range(-half, tail)]

        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            # Vectorized block apply: scatter same delta at each of block_size^2 offsets
            for (dy, dx) in offsets:
                ys_b = (ys_t + dy).clamp(0, OUT_H - 1)
                xs_b = (xs_t + dx).clamp(0, OUT_W - 1)
                for c in range(3):
                    f1_p[batch_idx, c, ys_b, xs_b] = f1_p[batch_idx, c, ys_b, xs_b] + cur_d[..., c]
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
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            xy = np.stack([xs_np[bi], ys_np[bi]], axis=1)
            out[pair_i] = (xy, d_np[bi])
    return out


def apply_block_patches(f_all, patches_dict, block_size=1):
    """Apply patches as block_size x block_size blocks."""
    out = f_all.clone()
    H, W = out.shape[1], out.shape[2]
    half = block_size // 2
    for pair_i, (xy, d) in patches_dict.items():
        arr = out[pair_i].float().numpy()
        K = xy.shape[0]
        for j in range(K):
            x, y = int(xy[j, 0]), int(xy[j, 1])
            di = d[j].astype(np.float32)
            if not di.any(): continue
            if block_size == 1:
                arr[y, x] += di
            else:
                y0 = max(0, y - half); y1 = min(H, y + half + (block_size % 2))
                x0 = max(0, x - half); x1 = min(W, x + half + (block_size % 2))
                arr[y0:y1, x0:x1] += di[None, None, :]
        out[pair_i] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def block_sidecar_size(patches_dict, block_size):
    """Same sparse format as before, +1 byte for block_size header."""
    import struct, bz2
    if not patches_dict:
        return 0
    parts = [struct.pack("<BH", block_size, len(patches_dict))]
    for pair_i, (xy, d) in sorted(patches_dict.items()):
        K = xy.shape[0]
        parts.append(struct.pack("<HH", pair_i, K))
        for j in range(K):
            parts.append(struct.pack("<HHbbb",
                int(xy[j, 0]), int(xy[j, 1]),
                int(d[j, 0]), int(d[j, 1]), int(d[j, 2])))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


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

    csv_path = OUTPUT_DIR / "blocks_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "block_size", "tier_specs", "pairs",
                                 "K_total", "sidecar_bytes", "score", "pose_term", "delta", "elapsed"])

    def run(spec, tiers, block_size):
        all_patches = {}
        t0 = time.time()
        for start, end, K in tiers:
            pairs = [int(x) for x in rank[start:end]]
            if not pairs or K <= 0: continue
            ps = find_block_pose_patches(
                f1_all, f2_all, data["val_poses"], posenet,
                pairs, K=K, n_iter=80, device=device, block_size=block_size)
            all_patches.update(ps)
        elapsed = time.time() - t0
        sb = block_sidecar_size(all_patches, block_size)
        f1_p = apply_block_patches(f1_all, all_patches, block_size=block_size)
        seg, pose = fast_eval(f1_p, f2_all, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb)
        delta = full['score'] - base['score']
        K_total = sum(xy.shape[0] for xy, d in all_patches.values())
        npairs = len(all_patches)
        print(f"  >> {spec}: pairs={npairs} K_total={K_total} sb={sb}B score={full['score']:.4f} "
              f"pose={full['pose_term']:.4f} delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, block_size, str(tiers), npairs, K_total, sb,
                                     full['score'], full['pose_term'], delta, elapsed])

    # ── Block size sweep on the winning tier ──
    print("\n=== Block size sweep (350_K7+250_K2) ===", flush=True)
    for bs in [1, 2, 3, 4]:
        run(f"350_K7+250_K2_block{bs}", [(0, 350, 7), (350, 600, 2)], bs)

    # ── Smaller K with bigger blocks (saves bytes) ──
    print("\n=== Lower K with bigger blocks ===", flush=True)
    run("350_K5+250_K2_block2", [(0, 350, 5), (350, 600, 2)], 2)
    run("350_K4+250_K2_block3", [(0, 350, 4), (350, 600, 2)], 3)
    run("300_K5+300_K2_block2", [(0, 300, 5), (300, 600, 2)], 2)
    run("400_K4+200_K1_block3", [(0, 400, 4), (400, 600, 1)], 3)

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
