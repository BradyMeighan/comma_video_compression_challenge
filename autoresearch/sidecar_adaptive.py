#!/usr/bin/env python
"""
Adaptive sidecar: spend patch budget on the hardest pairs.

Findings from sidecar_analyze:
  - top 10% of pairs hold 32.8% of pose error
  - top 25% hold 58.2%
  - dim 0 dominates (RMS 0.066, ~2x next dim)
  - seg is almost solved; don't waste budget there

Variants tested:
  A. concentrated_topN_K — only top-N hardest pose pairs get K patches
  B. ramp_KminKmax     — K_per_pair = lerp(Kmin, Kmax) by pose-error rank
  C. iterative_residual — apply K, recompute pose grad, add K more

Sparse sidecar format (better than full (N, K, 2/3) tensors when most pairs have 0):
  uint16 num_pairs
  for each pair:
    uint16 pair_idx
    uint16 K_i
    K_i * (uint16 x, uint16 y, int8 dr, int8 dg, int8 db)
"""
import sys, os, time, csv, struct, bz2, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (MODEL_H, MODEL_W, OUT_H, OUT_W, pack_pair_yuv6, get_pose6,
                     load_posenet, MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE,
                     estimate_model_bytes, gpu_cleanup)
from train import Generator, load_data_full
import sidecar_explore as se

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pose_patches_per_pair(f1_all, f2_all, gt_poses, posenet, k_per_pair, n_iter, device):
    """Variable K per pair. k_per_pair: array of length N giving K_i for each pair.

    Returns dict: pair_idx -> (xy(K_i,2) uint16, d(K_i,3) int8)
    Pairs with K_i=0 not included.
    """
    n = f1_all.shape[0]
    out = {}
    indices = np.where(k_per_pair > 0)[0]
    if len(indices) == 0:
        return out

    # Group pairs by K (so we can batch same-K together)
    k_groups = {}
    for idx in indices:
        K = int(k_per_pair[idx])
        k_groups.setdefault(K, []).append(int(idx))

    for K, idx_list in sorted(k_groups.items()):
        bs = max(1, min(8, 32 // max(1, K // 5)))  # smaller batches for larger K
        for start in range(0, len(idx_list), bs):
            batch_idx_list = idx_list[start:start + bs]
            b = len(batch_idx_list)
            sel = torch.tensor(batch_idx_list, dtype=torch.long)
            f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
            f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
            gt_p = gt_poses[sel].to(device).float()

            # Initial gradient via posenet
            f1_param = f1.clone().requires_grad_(True)
            pin = se.diff_posenet_input(f1_param, f2)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            grad = torch.autograd.grad(loss, f1_param)[0]
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
                f1_p = f1.clone()
                for c in range(3):
                    f1_p[batch_idx, c, ys_t, xs_t] = f1_p[batch_idx, c, ys_t, xs_t] + cur_d[..., c]
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
            for bi, pair_i in enumerate(batch_idx_list):
                xy = np.stack([xs_np[bi], ys_np[bi]], axis=1)  # (K, 2)
                out[pair_i] = (xy, d_np[bi])
    return out


def apply_sparse_patches(f1_all, patches_dict):
    """patches_dict: pair_idx -> (xy, d). Apply only to those pairs."""
    out = f1_all.clone()
    H, W = out.shape[1], out.shape[2]
    for pair_i, (xy, d) in patches_dict.items():
        arr = out[pair_i].float().numpy()
        K = xy.shape[0]
        for j in range(K):
            x, y = int(xy[j, 0]), int(xy[j, 1])
            di = d[j].astype(np.float32)
            if not di.any(): continue
            arr[y, x] += di
        out[pair_i] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def sparse_sidecar_size(patches_dict):
    """Pack as: u16 num_pairs, then for each pair: u16 idx, u16 K, K*7 bytes."""
    if not patches_dict:
        return 0
    parts = [struct.pack("<H", len(patches_dict))]
    for pair_i, (xy, d) in sorted(patches_dict.items()):
        K = xy.shape[0]
        parts.append(struct.pack("<HH", pair_i, K))
        # interleaved: x, y, dr, dg, db per patch
        for j in range(K):
            parts.append(struct.pack("<HHbbb",
                int(xy[j, 0]), int(xy[j, 1]),
                int(d[j, 0]), int(d[j, 1]), int(d[j, 2])))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Loading model + data...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)

    # Load per-pair pose error from analyze step
    pose_per_pair_path = OUTPUT_DIR / "pose_per_pair.npy"
    if not pose_per_pair_path.exists():
        print(f"ERROR: missing {pose_per_pair_path}. Run sidecar_analyze.py first.")
        sys.exit(1)
    pose_per_pair = np.load(pose_per_pair_path)
    n = len(pose_per_pair)
    rank = np.argsort(pose_per_pair)[::-1]  # hardest first
    print(f"  pose mean={pose_per_pair.mean():.5f} max={pose_per_pair.max():.5f} n={n}", flush=True)

    # Generate baseline frames
    print("Generating frames...", flush=True)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    base = se.eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=gen)
    base_score = se.compose_score(base["seg_dist"], base["pose_dist"], base["model_bytes"], 0)
    print(f"Baseline: score={base_score['score']:.4f} seg={base_score['seg_term']:.4f} "
          f"pose={base_score['pose_term']:.4f} rate={base_score['rate_term']:.4f}", flush=True)

    posenet = load_posenet(device)

    csv_path = OUTPUT_DIR / "adaptive_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["variant", "spec", "n_patched_pairs", "total_patches",
                                 "sidecar_bytes", "score", "seg_term", "pose_term",
                                 "rate_term", "delta", "elapsed_sec"])

    def run_variant(label, spec, k_per_pair, n_iter=80):
        print(f"\n=== {label}: {spec} ===", flush=True)
        n_patched = int((k_per_pair > 0).sum())
        total_K = int(k_per_pair.sum())
        print(f"  n_patched_pairs={n_patched} total_patches={total_K}", flush=True)
        t0 = time.time()
        patches = find_pose_patches_per_pair(f1_all, f2_all, data["val_poses"], posenet,
                                              k_per_pair, n_iter=n_iter, device=device)
        elapsed = time.time() - t0
        sb = sparse_sidecar_size(patches)
        f1_p = apply_sparse_patches(f1_all, patches)
        result = se.eval_with_frames(f1_p, f2_all, data, device, gen_for_bytes=gen)
        full = se.compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        delta = full['score'] - base_score['score']
        print(f"  search: {elapsed:.1f}s  sidecar={sb}B  score={full['score']:.4f} "
              f"seg={full['seg_term']:.4f} pose={full['pose_term']:.4f} "
              f"rate={full['rate_term']:.4f} delta={delta:+.4f}", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([label, spec, n_patched, total_K, sb,
                                     full['score'], full['seg_term'], full['pose_term'],
                                     full['rate_term'], delta, elapsed])

    # ── Variant A: concentrated_topN_K (only top-N pairs get patches) ──
    #   Reference: uniform K=5 across 600 pairs gave -0.026 (sidecar 13.4KB).
    #   Hypothesis: skipping easy pairs frees budget for more patches on hard ones.
    for n_top, K in [(60, 30), (100, 20), (150, 15), (200, 10),
                     (300, 7),  (60, 50), (100, 30)]:
        k_arr = np.zeros(n, dtype=np.int32)
        k_arr[rank[:n_top]] = K
        run_variant("concentrated", f"top{n_top}_K{K}", k_arr)

    # ── Variant B: ramp Kmax→Kmin by rank ──
    #   Smooth allocation: pair at rank i gets K(i) = round(Kmax*(1 - i/N) + Kmin*i/N)
    for Kmax, Kmin in [(20, 2), (25, 3), (15, 3)]:
        k_arr = np.zeros(n, dtype=np.int32)
        for j, idx in enumerate(rank):
            frac = j / max(1, n - 1)
            k_arr[idx] = int(round(Kmax * (1 - frac) + Kmin * frac))
        run_variant("ramp", f"Kmax{Kmax}_Kmin{Kmin}", k_arr)

    # ── Variant C: error-proportional K (K = scale * sqrt(pose_err) clipped) ──
    for scale, k_clip in [(700, 30), (500, 25), (400, 20)]:
        k_arr = np.zeros(n, dtype=np.int32)
        for i in range(n):
            ki = int(round(scale * math.sqrt(max(0, pose_per_pair[i]))))
            k_arr[i] = max(0, min(k_clip, ki))
        run_variant("proportional", f"scale{scale}_clip{k_clip}", k_arr)

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
