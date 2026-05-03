#!/usr/bin/env python
"""
Chain DIFFERENT methods sequentially. Apply method A, evaluate, apply B on
top, evaluate, apply C on top... continue while score improves.

Methods (each writes to either f1 or f2 with bz2-compressed sparse format):
  M1: pose_f1_K7_top400      — gradient through PoseNet on f1, hardest 400 pairs
  M2: pose_f2_K3_top200      — gradient through PoseNet on f2, hardest 200 pairs
                                  (different signal: temporal motion flow)
  M3: seg_f2_K5_top200       — gradient through SegNet on f2, on pairs with
                                  highest seg disagreement
  M4: pose_f1_dim0_K3_top200 — re-target dim 0 (yaw) only on residual hardest

Each step measured: cumulative score, marginal byte cost, marginal score gain.
Stops when marginal score gain < marginal byte cost (in score units).
"""
import sys, os, time, csv, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      load_segnet, MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE,
                      estimate_model_bytes, gpu_cleanup)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs, per_pair_pose_mse,
                            merge_patches)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pose_f2_patches(cur_f1, cur_f2, gt_poses, posenet, pair_indices,
                          K, n_iter, device, exclusion=None):
    """Find K patches on f2 that fix pose. Different gradient signal than f1."""
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = cur_f1[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = cur_f2[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        f2_param = f2.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1, f2_param)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f2_param)[0]
        gmag = grad.abs().sum(dim=1)
        if exclusion is not None:
            for bi, pair_i in enumerate(idx_list):
                for (xx, yy) in exclusion.get(pair_i, set()):
                    gmag[bi, yy, xx] = -1.0

        flat = gmag.view(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        for _ in range(n_iter):
            opt.zero_grad()
            f2_p = f2.clone()
            for c in range(3):
                f2_p[batch_idx, c, ys_t, xs_t] = f2_p[batch_idx, c, ys_t, xs_t] + cur_d[..., c]
            f2_p = f2_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1, f2_p)
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


def find_seg_f2_patches(cur_f2, gt_masks, segnet, pair_indices, K, n_iter, device):
    """Find K patches on f2 to reduce segnet disagreement at MODEL_H × MODEL_W."""
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f2 = cur_f2[sel].to(device).float().permute(0, 3, 1, 2)
        gt_cls = gt_masks[sel].to(device).long()

        f2_param = f2.clone().requires_grad_(True)
        f2_r = F.interpolate(f2_param, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
        logits = segnet(f2_r).float()
        loss = F.cross_entropy(logits, gt_cls, reduction='sum')
        grad = torch.autograd.grad(loss, f2_param)[0]
        gmag = grad.abs().sum(dim=1)

        flat = gmag.view(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        for _ in range(n_iter):
            opt.zero_grad()
            f2_p = f2.clone()
            for c in range(3):
                f2_p[batch_idx, c, ys_t, xs_t] = f2_p[batch_idx, c, ys_t, xs_t] + cur_d[..., c]
            f2_p = f2_p.clamp(0, 255)
            f2_pr = F.interpolate(f2_p, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            logits = segnet(f2_pr).float()
            loss = F.cross_entropy(logits, gt_cls, reduction='sum')
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


def find_pose_dim_patches(cur_f1, cur_f2, gt_poses, posenet, pair_indices,
                           K, n_iter, device, dim_weights, exclusion=None):
    """Pose patches with custom dim weighting (e.g. dim 0 only)."""
    out = {}
    bs = 8
    dim_w = torch.tensor(dim_weights, device=device).float()
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = cur_f1[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = cur_f2[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = (((fp - gt_p) ** 2) * dim_w).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        gmag = grad.abs().sum(dim=1)
        if exclusion is not None:
            for bi, pair_i in enumerate(idx_list):
                for (xx, yy) in exclusion.get(pair_i, set()):
                    gmag[bi, yy, xx] = -1.0

        flat = gmag.view(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            for c in range(3):
                f1_p[batch_idx, c, ys_t, xs_t] = f1_p[batch_idx, c, ys_t, xs_t] + cur_d[..., c]
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss = (((fp - gt_p) ** 2) * dim_w).sum()
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


def per_pair_seg_disagree(cur_f2, gt_masks, segnet, device):
    """Per-pair seg disagreement rate at MODEL_H x MODEL_W."""
    n = cur_f2.shape[0]
    out = np.zeros(n, dtype=np.float32)
    bs = 16
    with torch.inference_mode():
        for i in range(0, n, bs):
            e = min(i + bs, n)
            f2 = cur_f2[i:e].to(device).float().permute(0, 3, 1, 2)
            f2_r = F.interpolate(f2, (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            pred = segnet(f2_r).argmax(1)
            gt = gt_masks[i:e].to(device).long()
            disagree = (pred != gt).float().mean(dim=(1, 2))
            out[i:e] = disagree.cpu().numpy()
    return out


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("Loading...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    segnet = load_segnet(device)
    model_bytes = estimate_model_bytes(gen)

    print("Generating frames...", flush=True)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    seg, pose = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg, pose, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} seg={base['seg_term']:.4f} "
          f"pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    seg_per_pair = np.load(OUTPUT_DIR / "seg_per_pair.npy")

    csv_path = OUTPUT_DIR / "chain_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["step", "method", "delta_bytes", "delta_score",
                                 "cum_bytes", "cum_score", "cum_pose", "cum_seg",
                                 "elapsed"])

    # ─────────────────────────────────────────────────────────────────
    # CHAIN: each step modifies ONE frame side based on a NEW gradient
    # signal computed on the PATCHED frames (residual error).
    # ─────────────────────────────────────────────────────────────────
    cur_f1 = f1_all.clone()
    cur_f2 = f2_all.clone()
    f1_patches = {}    # pair_idx → (xy, d) for f1 modifications
    f2_patches = {}    # pair_idx → (xy, d) for f2 modifications
    cum_bytes = 0
    cum_score = base['score']
    f1_excl = {}  # exclusion map for f1 across waves
    f2_excl = {}  # exclusion map for f2 across waves

    def evaluate_state():
        f1_use = apply_sparse_patches(f1_all, f1_patches) if f1_patches else f1_all
        f2_use = apply_sparse_patches(f2_all, f2_patches) if f2_patches else f2_all
        s, p = fast_eval(f1_use, f2_use, data["val_rgb"], device)
        sb = (sparse_sidecar_size(f1_patches) if f1_patches else 0) + \
             (sparse_sidecar_size(f2_patches) if f2_patches else 0)
        return s, p, sb, f1_use, f2_use

    def log_step(step, method, prev_bytes, prev_score, t0):
        nonlocal cum_bytes, cum_score
        s, p, sb, f1_use, f2_use = evaluate_state()
        full = fast_compose(s, p, model_bytes, sb)
        elapsed = time.time() - t0
        delta_b = sb - prev_bytes
        delta_s = full['score'] - prev_score
        cum_bytes = sb; cum_score = full['score']
        print(f"  [{step}] {method}: +{delta_b:>5d}B d_score={delta_s:+.4f} "
              f"cum_score={full['score']:.4f} cum_pose={full['pose_term']:.4f} "
              f"cum_seg={full['seg_term']:.4f}  ({elapsed:.1f}s)", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([step, method, delta_b, delta_s, sb,
                                     full['score'], full['pose_term'],
                                     full['seg_term'], elapsed])
        return f1_use, f2_use, sb, full['score']

    rank_pose = np.argsort(pose_per_pair)[::-1]
    rank_seg = np.argsort(seg_per_pair)[::-1]

    # ─── Step 1: pose_f1 K=7 on top 400 (the established winner) ───
    print("\n=== Step 1: pose_f1 K=7 top400 ===", flush=True)
    t0 = time.time()
    pairs_400 = [int(x) for x in rank_pose[:400]]
    p1 = find_pose_patches_for_pairs(f1_all, f2_all, data["val_poses"],
                                       posenet, pairs_400, K=7, n_iter=80, device=device)
    f1_patches.update(p1)
    for pi, (xy, d) in p1.items():
        f1_excl[pi] = set((int(xy[j, 0]), int(xy[j, 1])) for j in range(xy.shape[0]))
    cur_f1, cur_f2, cum_bytes, cum_score = log_step("1", "pose_f1_K7_top400",
                                                      0, base['score'], t0)

    # ─── Step 2: pose_f2 K=3 on top 200 by RESIDUAL pose error ───
    # Now that f1 is patched, f2 modifications give independent gradient
    print("\n=== Step 2: pose_f2 K=3 top200 (post-f1) ===", flush=True)
    t0 = time.time()
    res_pose = per_pair_pose_mse(cur_f1, cur_f2, data["val_poses"], posenet, device)
    res_rank = np.argsort(res_pose)[::-1]
    res_top_200 = [int(x) for x in res_rank[:200]]
    p2 = find_pose_f2_patches(cur_f1, cur_f2, data["val_poses"],
                               posenet, res_top_200, K=3, n_iter=80, device=device)
    f2_patches = {**f2_patches, **p2}
    for pi, (xy, d) in p2.items():
        f2_excl[pi] = set((int(xy[j, 0]), int(xy[j, 1])) for j in range(xy.shape[0]))
    cur_f1, cur_f2, cum_bytes, cum_score = log_step("2", "pose_f2_K3_top200_residual",
                                                      cum_bytes, cum_score, t0)

    # ─── Step 3: seg_f2 K=5 on pairs with highest seg disagreement ───
    print("\n=== Step 3: seg_f2 K=5 on top200 worst seg pairs ===", flush=True)
    t0 = time.time()
    seg_res = per_pair_seg_disagree(cur_f2, data["val_masks"], segnet, device)
    seg_rank = np.argsort(seg_res)[::-1]
    seg_top_200 = [int(x) for x in seg_rank[:200] if seg_res[x] > 0]
    print(f"  {len(seg_top_200)} pairs with disagreement > 0 (max={seg_res[seg_rank[0]]:.4f})", flush=True)
    if len(seg_top_200) > 0:
        p3 = find_seg_f2_patches(cur_f2, data["val_masks"], segnet,
                                   seg_top_200, K=5, n_iter=80, device=device)
        # Merge with existing f2_patches (avoid coord collision)
        for pi, (xy, d) in p3.items():
            if pi in f2_patches:
                exist_xy, exist_d = f2_patches[pi]
                # Filter out collisions with existing f2 patches
                exist_set = set((int(exist_xy[j, 0]), int(exist_xy[j, 1])) for j in range(exist_xy.shape[0]))
                keep = [j for j in range(xy.shape[0])
                        if (int(xy[j, 0]), int(xy[j, 1])) not in exist_set]
                if keep:
                    new_xy = np.concatenate([exist_xy, xy[keep]], axis=0)
                    new_d = np.concatenate([exist_d, d[keep]], axis=0)
                    f2_patches[pi] = (new_xy, new_d)
            else:
                f2_patches[pi] = (xy, d)
    cur_f1, cur_f2, cum_bytes, cum_score = log_step("3", "seg_f2_K5_top200",
                                                      cum_bytes, cum_score, t0)

    # ─── Step 4: pose_f1 dim0-only K=3 on residual hardest 200 ───
    print("\n=== Step 4: pose_f1 dim0-only K=3 top200 (residual) ===", flush=True)
    t0 = time.time()
    res_pose = per_pair_pose_mse(cur_f1, cur_f2, data["val_poses"], posenet, device)
    res_rank = np.argsort(res_pose)[::-1]
    res_top_200 = [int(x) for x in res_rank[:200]]
    p4 = find_pose_dim_patches(cur_f1, cur_f2, data["val_poses"], posenet,
                                res_top_200, K=3, n_iter=80, device=device,
                                dim_weights=[1, 0, 0, 0, 0, 0],
                                exclusion=f1_excl)
    # Merge into f1_patches (concat per pair)
    for pi, (xy, d) in p4.items():
        if pi in f1_patches:
            old_xy, old_d = f1_patches[pi]
            f1_patches[pi] = (np.concatenate([old_xy, xy], axis=0),
                               np.concatenate([old_d, d], axis=0))
        else:
            f1_patches[pi] = (xy, d)
        f1_excl.setdefault(pi, set()).update(
            (int(xy[j, 0]), int(xy[j, 1])) for j in range(xy.shape[0]))
    cur_f1, cur_f2, cum_bytes, cum_score = log_step("4", "pose_f1_dim0_K3_top200_residual",
                                                      cum_bytes, cum_score, t0)

    # ─── Step 5: pose_f1 K=3 on next-tier 200 (pairs 400-600) ───
    print("\n=== Step 5: pose_f1 K=3 on tail 200 pairs ===", flush=True)
    t0 = time.time()
    tail_pairs = [int(x) for x in rank_pose[400:600]]
    p5 = find_pose_patches_for_pairs(f1_all, f2_all, data["val_poses"],
                                       posenet, tail_pairs, K=3, n_iter=80, device=device)
    for pi, (xy, d) in p5.items():
        if pi in f1_patches:
            old_xy, old_d = f1_patches[pi]
            f1_patches[pi] = (np.concatenate([old_xy, xy], axis=0),
                               np.concatenate([old_d, d], axis=0))
        else:
            f1_patches[pi] = (xy, d)
    cur_f1, cur_f2, cum_bytes, cum_score = log_step("5", "pose_f1_K3_tail200",
                                                      cum_bytes, cum_score, t0)

    print(f"\nDone. Final score={cum_score:.4f} (delta={cum_score-base['score']:+.4f})", flush=True)
    print(f"Results: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
