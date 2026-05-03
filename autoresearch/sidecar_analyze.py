#!/usr/bin/env python
"""
Analyze where our model's error comes from to design targeted sidecars.

Reports:
1. Per-pair seg & pose error distribution (which pairs are hardest?)
2. Per-class seg error (which classes confuse the model?)
3. Per-pose-dim error (which of 6 pose dims has most error?)
4. Spatial heatmap of seg disagreement (where in the frame do errors concentrate?)
"""
import sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import einops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1")
os.environ.setdefault("CONFIG", "B")

from prepare import (load_data, evaluate, gpu_cleanup, MODEL_H, MODEL_W, OUT_H, OUT_W,
                     diff_round, pack_pair_yuv6, get_pose6, kl_on_logits,
                     load_segnet, load_posenet)
from train import Generator, load_data_full

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")

def main():
    device = torch.device("cuda")
    print(f"Loading model + data...", flush=True)
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)
    data = load_data_full(device)

    from prepare import apply_fp4_to_model
    apply_fp4_to_model(gen)
    gen.eval()

    segnet = load_segnet(device)
    posenet = load_posenet(device)

    rgb = data["val_rgb"]; masks = data["val_masks"]; poses = data["val_poses"]
    n = rgb.shape[0]
    print(f"Eval pairs: {n}", flush=True)

    # Per-pair metrics
    seg_per_pair = np.zeros(n, dtype=np.float32)
    pose_per_pair = np.zeros(n, dtype=np.float32)
    pose_dim_errs = np.zeros((n, 6), dtype=np.float32)
    class_confusion = np.zeros((5, 5), dtype=np.int64)  # GT class → pred class
    seg_disagree_map = np.zeros((MODEL_H, MODEL_W), dtype=np.float64)  # spatial heatmap

    bs = 16
    with torch.inference_mode():
        for i in range(0, n, bs):
            e = min(i + bs, n)
            m = masks[i:e].to(device).long()
            p = poses[i:e].to(device).float()
            p1, p2 = gen(m, p)
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode='bilinear', align_corners=False).clamp(0, 255).round()
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode='bilinear', align_corners=False).clamp(0, 255).round()
            comp = torch.stack([f1u, f2u], dim=1).to(torch.uint8).float()
            gt = rgb[i:e].to(device)
            gt_chw = einops.rearrange(gt, 'b t h w c -> b t c h w').float()

            # SegNet on f2 of comp (predicted) and gt
            r2_pred = F.interpolate(comp[:, 1], (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            r2_gt = F.interpolate(gt_chw[:, 1], (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            seg_pred = segnet(r2_pred).argmax(1)
            seg_gt = segnet(r2_gt).argmax(1)
            disagree = (seg_pred != seg_gt).float()
            seg_per_pair[i:e] = disagree.mean(dim=(1, 2)).cpu().numpy()
            seg_disagree_map += disagree.sum(dim=0).cpu().numpy()

            # Class confusion (sample subset of pixels)
            for b_idx in range(seg_pred.shape[0]):
                gt_flat = seg_gt[b_idx].cpu().numpy().flatten()
                pred_flat = seg_pred[b_idx].cpu().numpy().flatten()
                # Only count disagreements (faster + more informative)
                mask_d = gt_flat != pred_flat
                gts = gt_flat[mask_d]; preds = pred_flat[mask_d]
                for g_c in range(5):
                    for p_c in range(5):
                        class_confusion[g_c, p_c] += int(((gts == g_c) & (preds == p_c)).sum())

            # PoseNet
            x_pred = einops.rearrange(comp, 'b t c h w -> b t c h w')
            x_gt = einops.rearrange(gt_chw, 'b t c h w -> b t c h w')
            pose_pred = get_pose6(posenet, posenet.preprocess_input(x_pred)).float()
            pose_gt = get_pose6(posenet, posenet.preprocess_input(x_gt)).float()
            err = (pose_pred - pose_gt).pow(2)
            pose_per_pair[i:e] = err.mean(dim=1).cpu().numpy()
            pose_dim_errs[i:e] = err.cpu().numpy()

    # ─── REPORT ───
    print("\n" + "="*60)
    print("PER-PAIR ERROR DISTRIBUTION")
    print("="*60)
    seg_sorted = np.argsort(seg_per_pair)[::-1]
    pose_sorted = np.argsort(pose_per_pair)[::-1]
    print(f"\nSeg disagreement rate:")
    print(f"  mean: {seg_per_pair.mean():.4f} median: {np.median(seg_per_pair):.4f}")
    print(f"  worst 10 pairs: {seg_sorted[:10].tolist()}  (rates: {seg_per_pair[seg_sorted[:10]].round(3).tolist()})")
    print(f"  best 10 pairs:  {seg_sorted[-10:].tolist()}  (rates: {seg_per_pair[seg_sorted[-10:]].round(3).tolist()})")
    # Concentration: what fraction of total error is in top X% of pairs?
    total = seg_per_pair.sum()
    for pct in [10, 25, 50]:
        k = max(1, n * pct // 100)
        top_sum = np.sort(seg_per_pair)[-k:].sum()
        print(f"  top {pct}% of pairs ({k}) hold {100*top_sum/total:.1f}% of seg error")

    print(f"\nPose MSE:")
    print(f"  mean: {pose_per_pair.mean():.5f} median: {np.median(pose_per_pair):.5f}")
    print(f"  worst 10 pairs: {pose_sorted[:10].tolist()}  (mse: {pose_per_pair[pose_sorted[:10]].round(4).tolist()})")
    total = pose_per_pair.sum()
    for pct in [10, 25, 50]:
        k = max(1, n * pct // 100)
        top_sum = np.sort(pose_per_pair)[-k:].sum()
        print(f"  top {pct}% of pairs ({k}) hold {100*top_sum/total:.1f}% of pose error")

    # Overlap between hardest seg + hardest pose pairs
    top_seg = set(seg_sorted[:60].tolist())
    top_pose = set(pose_sorted[:60].tolist())
    overlap = top_seg & top_pose
    print(f"\nOverlap of top-10% seg-hard and pose-hard pairs: {len(overlap)} / 60")

    print("\n" + "="*60)
    print("PER-POSE-DIM ERROR (sqrt of mean MSE per dim)")
    print("="*60)
    rms = np.sqrt(pose_dim_errs.mean(axis=0))
    print(f"  dim 0: {rms[0]:.5f}")
    print(f"  dim 1: {rms[1]:.5f}")
    print(f"  dim 2: {rms[2]:.5f}")
    print(f"  dim 3: {rms[3]:.5f}")
    print(f"  dim 4: {rms[4]:.5f}")
    print(f"  dim 5: {rms[5]:.5f}")

    print("\n" + "="*60)
    print("SEG CLASS CONFUSION (rows=GT, cols=Pred, only disagreements)")
    print("="*60)
    print("       Pred:  0       1       2       3       4")
    for g_c in range(5):
        row = "    GT " + str(g_c) + ": "
        for p_c in range(5):
            row += f"{class_confusion[g_c, p_c]:>8d}"
        print(row)

    print("\n" + "="*60)
    print("SPATIAL DISAGREEMENT HEATMAP (top 10 hottest pixel locations)")
    print("="*60)
    hm = seg_disagree_map  # shape (H, W), counts per pixel
    flat = hm.flatten()
    top = np.argpartition(flat, -10)[-10:]
    top = top[np.argsort(flat[top])[::-1]]
    print(f"  Total disagreements summed across all pairs:")
    for idx in top:
        y = idx // MODEL_W; x = idx % MODEL_W
        print(f"    pixel ({x:>3d}, {y:>3d}): {int(flat[idx])} disagreements (out of {n} pairs)")

    # Save heatmap as numpy for further use
    np.save("autoresearch/sidecar_results/seg_heatmap.npy", hm)
    np.save("autoresearch/sidecar_results/seg_per_pair.npy", seg_per_pair)
    np.save("autoresearch/sidecar_results/pose_per_pair.npy", pose_per_pair)
    np.save("autoresearch/sidecar_results/pose_dim_errs.npy", pose_dim_errs)
    print(f"\n[saved analysis arrays to autoresearch/sidecar_results/]")


if __name__ == "__main__":
    main()
