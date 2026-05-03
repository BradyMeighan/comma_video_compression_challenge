#!/usr/bin/env python
"""
DEEP ANALYSIS on the BEST-PATCHED output (350_K7+250_K2).

Goal: discover NEW attack surfaces beyond pose_f1 patches we've exhausted.

Reports:
1. Per-pair RESIDUAL pose error (after best patches applied)
   - Has the distribution shifted? Are there new "hardest" pairs?
   - Concentration: what % in top 10/25/50% of pairs after patching?

2. Per-pose-dim residual error breakdown
   - Did patches reduce all dims uniformly or skew?

3. Spatial gradient on PATCHED frames
   - Where does PoseNet still want to "see" changes?
   - Do we have hot spots that were too low-grad before to be top-K?

4. Coordinate sharing analysis
   - How many of the 2950 patches sit at same x,y across pairs?
   - Could a shared-coordinate dictionary save bytes?

5. Per-class seg confusion on PATCHED frame 2
   - Which (gt, pred) pairs are still being confused?
   - Where exactly in the frame?

6. PoseNet dim-wise contribution to remaining error
   - For each pair, what fraction of remaining MSE comes from each dim?
   - Are there pairs where ONE dim is the only error source?

7. Pose error vs YUV channel sensitivity
   - PoseNet uses YUV6 (Y of f1, Y of f2, UV of avg). Which channel gives most signal?

8. Cross-pair coord clustering
   - Run k-means on ALL patch coordinates (across pairs).
   - If 2950 patches cluster into 100 unique locations, shared dictionary feasible.
"""
import sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      load_segnet, pack_pair_yuv6, diff_rgb_to_yuv6,
                      estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs, per_pair_pose_mse)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def per_pair_pose_dim_errs(cur_f1, cur_f2, val_poses, posenet, device):
    """Per-pair, per-dim pose squared error after applying patches."""
    n = cur_f1.shape[0]
    out = np.zeros((n, 6), dtype=np.float32)
    bs = 16
    with torch.inference_mode():
        for i in range(0, n, bs):
            e = min(i + bs, n)
            f1 = cur_f1[i:e].to(device).float().permute(0, 3, 1, 2)
            f2 = cur_f2[i:e].to(device).float().permute(0, 3, 1, 2)
            pin = se.diff_posenet_input(f1, f2)
            fp = get_pose6(posenet, pin).float()
            err = (fp - val_poses[i:e].to(device).float()).pow(2)
            out[i:e] = err.cpu().numpy()
    return out


def spatial_gradient_heatmap(cur_f1, cur_f2, val_poses, posenet, device):
    """Sum |grad| over all pairs to find pixels PoseNet still wants to change."""
    n = cur_f1.shape[0]
    grad_acc = torch.zeros((3, OUT_H, OUT_W), device=device)
    bs = 4
    for i in range(0, n, bs):
        e = min(i + bs, n)
        f1 = cur_f1[i:e].to(device).float().permute(0, 3, 1, 2)
        f2 = cur_f2[i:e].to(device).float().permute(0, 3, 1, 2)
        gt_p = val_poses[i:e].to(device).float()
        f1p = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1p, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        g = torch.autograd.grad(loss, f1p)[0]
        grad_acc += g.abs().sum(dim=0)
    return grad_acc.cpu().numpy()


def coord_clustering(patches_dict, n_clusters=50):
    """How concentrated are patch coords across pairs?"""
    all_coords = []
    for pair_i, (xy, d) in patches_dict.items():
        for j in range(xy.shape[0]):
            all_coords.append((int(xy[j, 0]), int(xy[j, 1])))
    n = len(all_coords)

    # Use Counter for robust unique counting
    from collections import Counter
    cnt = Counter(all_coords)
    n_unique = len(cnt)
    n_repeated = sum(1 for c in cnt.values() if c > 1)

    top_coords = [(x, y, c) for (x, y), c in cnt.most_common(20)]
    return n, n_unique, n_repeated, top_coords


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

    # Generate baseline + apply best sidecar
    print("Generating frames...", flush=True)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    seg_b, pose_b = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg_b, pose_b, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    # Apply best sidecar (350_K7 + 250_K2)
    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    print("\nApplying best sidecar (350_K7+250_K2)...", flush=True)
    p_top = find_pose_patches_for_pairs(
        f1_all, f2_all, data["val_poses"], posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_all, f2_all, data["val_poses"], posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    best_patches = {**p_top, **p_tail}
    cur_f1 = apply_sparse_patches(f1_all, best_patches)
    cur_f2 = f2_all
    seg_p, pose_p = fast_eval(cur_f1, cur_f2, data["val_rgb"], device)
    sb = sparse_sidecar_size(best_patches)
    cur = fast_compose(seg_p, pose_p, model_bytes, sb)
    print(f"With sidecar: score={cur['score']:.4f} pose={cur['pose_term']:.4f} "
          f"sb={sb}B (delta {cur['score']-base['score']:+.4f})", flush=True)

    # ─────────────────────────────────────────────────────────────────
    # 1. Per-pair RESIDUAL pose error
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("1. RESIDUAL POSE ERROR PER PAIR (after best sidecar)")
    print("="*70)
    res_pose = per_pair_pose_mse(cur_f1, cur_f2, data["val_poses"], posenet, device)
    res_rank = np.argsort(res_pose)[::-1]
    n = len(res_pose)
    total = res_pose.sum()
    print(f"  Pose MSE: mean={res_pose.mean():.5f} median={np.median(res_pose):.5f} "
          f"max={res_pose.max():.5f}")
    print(f"  Original mean: {pose_per_pair.mean():.5f} (so {(1-res_pose.mean()/pose_per_pair.mean())*100:.1f}% reduction)")
    for pct in [10, 25, 50]:
        k = max(1, n * pct // 100)
        top_sum = np.sort(res_pose)[-k:].sum()
        print(f"  top {pct}% ({k} pairs) hold {100*top_sum/total:.1f}% of residual error")
    print(f"  worst 10 pairs (by residual): {res_rank[:10].tolist()}")
    print(f"    residual MSE: {res_pose[res_rank[:10]].round(5).tolist()}")

    # Cross-reference: were these the worst originally too?
    orig_rank_set = set(rank[:60].tolist())
    new_rank_set = set(res_rank[:60].tolist())
    overlap = orig_rank_set & new_rank_set
    print(f"  Overlap top-60 BEFORE/AFTER: {len(overlap)}/60 — "
          f"{60-len(overlap)} pairs are NEW hardest after patching")

    # ─────────────────────────────────────────────────────────────────
    # 2. Per-dim residual breakdown
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("2. PER-POSE-DIM RESIDUAL (sqrt mean MSE per dim)")
    print("="*70)
    res_dim = per_pair_pose_dim_errs(cur_f1, cur_f2, data["val_poses"], posenet, device)
    rms_after = np.sqrt(res_dim.mean(axis=0))
    rms_before = np.sqrt(np.load(OUTPUT_DIR / "pose_dim_errs.npy").mean(axis=0))
    print(f"        BEFORE   AFTER    RATIO   absolute reduction")
    for d in range(6):
        ratio = rms_after[d] / rms_before[d]
        print(f"  dim {d}: {rms_before[d]:.5f}  {rms_after[d]:.5f}  {ratio:.3f}   {rms_before[d]-rms_after[d]:+.5f}")
    print(f"  total var captured by dim 0: {(res_dim[:, 0].sum() / res_dim.sum()) * 100:.1f}% (was {(np.load(OUTPUT_DIR / 'pose_dim_errs.npy')[:,0].sum() / np.load(OUTPUT_DIR / 'pose_dim_errs.npy').sum())*100:.1f}%)")

    # ─────────────────────────────────────────────────────────────────
    # 3. Spatial gradient heatmap on PATCHED frames
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("3. SPATIAL GRADIENT ON PATCHED FRAMES (where PoseNet still wants change)")
    print("="*70)
    print("  Computing gradient heatmap (slow, ~30s)...", flush=True)
    hm = spatial_gradient_heatmap(cur_f1, cur_f2, data["val_poses"], posenet, device)
    flat = hm.sum(axis=0).flatten()  # collapse channels
    top = np.argpartition(flat, -20)[-20:]
    top = top[np.argsort(flat[top])[::-1]]
    print(f"  Top-20 hottest residual-gradient pixels:")
    print(f"  (max grad value: {flat.max():.6f}, mean: {flat.mean():.6f})")
    for idx in top:
        y = idx // OUT_W; x = idx % OUT_W
        print(f"    ({x:>4d}, {y:>4d}): magnitude {flat[idx]:.6f}")
    np.save(OUTPUT_DIR / "residual_grad_heatmap.npy", hm)

    # ─────────────────────────────────────────────────────────────────
    # 4. Coordinate sharing analysis
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("4. COORDINATE CLUSTERING ACROSS PAIRS")
    print("="*70)
    n_total, n_unique, n_repeated, top_coords = coord_clustering(best_patches)
    print(f"  Total patches: {n_total}")
    print(f"  Unique (x,y) positions: {n_unique}")
    print(f"  Positions used by >1 pair: {n_repeated}")
    print(f"  Sharing potential: if we used only top {n_unique} as shared dict,")
    print(f"    coord overhead drops from {n_total*4}B to {n_unique*4}B (saving ~{(n_total-n_unique)*4}B raw)")
    print(f"  Top-20 most-reused coords (count of pairs using them):")
    for x, y, c in top_coords:
        print(f"    ({x:>4d}, {y:>4d}): {c} pairs")

    # ─────────────────────────────────────────────────────────────────
    # 5. Per-class seg confusion on patched f2
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("5. SEG CONFUSION ON PATCHED FRAMES")
    print("="*70)
    seg_disagree_count = np.zeros((5, 5), dtype=np.int64)
    bs = 16
    n_disagree_total = 0; n_pixels_total = 0
    with torch.inference_mode():
        for i in range(0, cur_f2.shape[0], bs):
            e = min(i + bs, cur_f2.shape[0])
            f2 = cur_f2[i:e].to(device).float().permute(0, 3, 1, 2)
            f2_r = F.interpolate(f2, (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            pred = segnet(f2_r).argmax(1)
            gt = data["val_masks"][i:e].to(device).long()
            disagree = (pred != gt)
            n_disagree_total += disagree.sum().item()
            n_pixels_total += disagree.numel()
            for b_idx in range(pred.shape[0]):
                gt_f = gt[b_idx].cpu().numpy().flatten()
                pr_f = pred[b_idx].cpu().numpy().flatten()
                m = gt_f != pr_f
                gts = gt_f[m]; prs = pr_f[m]
                for g_c in range(5):
                    for p_c in range(5):
                        seg_disagree_count[g_c, p_c] += int(((gts == g_c) & (prs == p_c)).sum())
    print(f"  Total seg disagree pixels: {n_disagree_total} ({100*n_disagree_total/n_pixels_total:.4f}%)")
    print(f"        Pred:  0       1       2       3       4")
    for g_c in range(5):
        row = "    GT " + str(g_c) + ": "
        for p_c in range(5):
            row += f"{seg_disagree_count[g_c, p_c]:>8d}"
        print(row)

    # ─────────────────────────────────────────────────────────────────
    # 6. Per-pair "dominant dim" count
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("6. PER-PAIR DOMINANT POSE DIM (which dim accounts for >50% of residual MSE?)")
    print("="*70)
    dim_dom_counts = np.zeros(7, dtype=int)  # 0-5 + "balanced" (no >50%)
    for i in range(n):
        if res_dim[i].sum() < 1e-10:
            continue
        frac = res_dim[i] / res_dim[i].sum()
        if frac.max() > 0.5:
            dim_dom_counts[frac.argmax()] += 1
        else:
            dim_dom_counts[6] += 1
    for d in range(6):
        print(f"  dim {d} dominant in {dim_dom_counts[d]:>3d} pairs ({100*dim_dom_counts[d]/n:.1f}%)")
    print(f"  no dim >50%:    {dim_dom_counts[6]:>3d} pairs ({100*dim_dom_counts[6]/n:.1f}%)")

    # ─────────────────────────────────────────────────────────────────
    # 7. YUV channel gradient comparison
    # ─────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("7. YUV CHANNEL SENSITIVITY (gradient magnitude per channel)")
    print("="*70)
    # Compute gradient on YUV6 input directly to PoseNet
    bs = 4
    yuv_grad_norm = torch.zeros(6, device=device)
    n_batches = 0
    for i in range(0, cur_f1.shape[0], bs):
        e = min(i + bs, cur_f1.shape[0])
        f1 = cur_f1[i:e].to(device).float().permute(0, 3, 1, 2)
        f2 = cur_f2[i:e].to(device).float().permute(0, 3, 1, 2)
        gt_p = data["val_poses"][i:e].to(device).float()
        # Build YUV6 with gradient
        f1_r = F.interpolate(f1, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
        f2_r = F.interpolate(f2, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
        yuv6 = pack_pair_yuv6(f1_r, f2_r).requires_grad_(True)
        fp = get_pose6(posenet, yuv6).float()
        loss = ((fp - gt_p) ** 2).sum()
        g = torch.autograd.grad(loss, yuv6)[0]
        yuv_grad_norm += g.abs().mean(dim=(0, 2, 3))
        n_batches += 1
    yuv_grad_norm /= n_batches
    yuv_names = ["Y(f1)", "Y(f2)", "U(f1)", "U(f2)", "V(f1)", "V(f2)"]
    print(f"  Mean |grad| per YUV6 channel:")
    for i, name in enumerate(yuv_names):
        print(f"    {name}: {yuv_grad_norm[i].item():.5f}")

    print("\n" + "="*70)
    print("DONE. Saved residual_grad_heatmap.npy to results dir.")
    print("="*70)


if __name__ == "__main__":
    main()
