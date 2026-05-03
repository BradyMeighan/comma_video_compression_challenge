#!/usr/bin/env python
"""
Iterative residual sidecar:
  Wave 1: find K1 patches per pair via posenet gradient.
  Apply them (in place).
  Wave 2: recompute posenet gradient on PATCHED frames.
           This finds the NEXT-most-impactful pixels (residual error).
  Add K2 patches per pair at those new locations.
  Continue for n_waves.

Hypothesis: a single big-K wave fights itself (later patches need to undo
earlier ones). Smaller waves with re-gradient between them get cleaner
attribution and may use bytes more efficiently.

We test on the top-N hardest pairs only (pose-error-concentrated regime).
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

from prepare import (OUT_H, OUT_W, get_pose6, load_posenet, gpu_cleanup)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_residual_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                           K_per_wave, n_iter, device, exclusion_set=None):
    """For pairs in pair_indices, find K_per_wave patches via posenet gradient.
    exclusion_set: dict pair_i -> set of (x, y) already used (skip these in topk).
    Returns dict pair_i -> (xy(K, 2), d(K, 3)).
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

        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        gmag = grad.abs().sum(dim=1)  # (b, H, W)

        # Mask out excluded pixels (already-used) before topk
        if exclusion_set is not None:
            for bi, pair_i in enumerate(idx_list):
                excl = exclusion_set.get(pair_i, set())
                for (xx, yy) in excl:
                    gmag[bi, yy, xx] = -1.0  # below all valid grad mags

        flat = gmag.view(b, -1)
        _, topk = torch.topk(flat, K_per_wave, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        cur_d = torch.zeros((b, K_per_wave, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K_per_wave)
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
        for bi, pair_i in enumerate(idx_list):
            xy = np.stack([xs_np[bi], ys_np[bi]], axis=1)
            out[pair_i] = (xy, d_np[bi])
    return out


def merge_patches(d1, d2):
    """Merge two patch dicts: per-pair concat. Pairs only in one keep alone."""
    keys = set(d1.keys()) | set(d2.keys())
    out = {}
    for k in keys:
        if k in d1 and k in d2:
            xy = np.concatenate([d1[k][0], d2[k][0]], axis=0)
            d = np.concatenate([d1[k][1], d2[k][1]], axis=0)
            out[k] = (xy, d)
        elif k in d1:
            out[k] = d1[k]
        else:
            out[k] = d2[k]
    return out


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Loading model + data...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    n = len(pose_per_pair)
    rank = np.argsort(pose_per_pair)[::-1]

    print("Generating frames...", flush=True)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    base = se.eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=gen)
    base_score = se.compose_score(base["seg_dist"], base["pose_dist"], base["model_bytes"], 0)
    print(f"Baseline: score={base_score['score']:.4f} pose={base_score['pose_term']:.4f}", flush=True)

    posenet = load_posenet(device)
    csv_path = OUTPUT_DIR / "iter_residual_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_pairs", "total_K", "sidecar_bytes",
                                 "score", "pose_term", "delta", "elapsed"])

    # Tests: vary n_top, K_per_wave, n_waves, n_iter_per_wave.
    # All run on the hardest n_top pairs (not all 600).
    tests = [
        # spec, n_top, K_per_wave, n_waves, n_iter
        ("top100_K10x2_iter80", 100, 10, 2, 80),
        ("top100_K5x4_iter60",  100,  5, 4, 60),
        ("top60_K15x2_iter80",   60, 15, 2, 80),
        ("top60_K10x3_iter80",   60, 10, 3, 80),
        ("top150_K7x2_iter80",  150,  7, 2, 80),
        ("top200_K5x2_iter80",  200,  5, 2, 80),
    ]

    for spec, n_top, K, n_waves, n_iter in tests:
        print(f"\n=== {spec}: n_top={n_top} K={K} waves={n_waves} iter={n_iter} ===", flush=True)
        pair_indices = [int(x) for x in rank[:n_top]]
        # Track applied patches; current frames after each wave
        cur_f1 = f1_all.clone()
        all_patches = {}
        exclusion = {pi: set() for pi in pair_indices}

        t0 = time.time()
        for wave in range(n_waves):
            wave_patches = find_residual_patches(
                cur_f1, f2_all, data["val_poses"], posenet,
                pair_indices, K_per_wave=K, n_iter=n_iter, device=device,
                exclusion_set=exclusion)
            # Merge into all_patches
            all_patches = merge_patches(all_patches, wave_patches)
            # Update exclusion + cur_f1
            for pi, (xy, d) in wave_patches.items():
                for j in range(xy.shape[0]):
                    exclusion[pi].add((int(xy[j, 0]), int(xy[j, 1])))
            cur_f1 = apply_sparse_patches(f1_all, all_patches)  # apply all from scratch
        elapsed = time.time() - t0

        sb = sparse_sidecar_size(all_patches)
        total_K = sum(xy.shape[0] for xy, d in all_patches.values())
        result = se.eval_with_frames(cur_f1, f2_all, data, device, gen_for_bytes=gen)
        full = se.compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        delta = full['score'] - base_score['score']
        print(f"  {spec}: total_K={total_K} sidecar={sb}B score={full['score']:.4f} "
              f"pose={full['pose_term']:.4f} delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, n_top, total_K, sb,
                                     full['score'], full['pose_term'], delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
