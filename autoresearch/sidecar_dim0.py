#!/usr/bin/env python
"""
Pose-dim-0 targeted sidecar.

From sidecar_analyze: dim 0 RMS=0.066 dominates (~2x dim 1 = 0.037).
Hypothesis: optimizing patches against dim-0 squared-error only (rather than
all 6 dims) gives the optimizer cleaner signal for the dominant component,
yielding better dim-0 reduction per byte spent.

Experiments:
  A. dim0_only loss for both gradient ranking AND optimization
  B. dim0+1+2 weighted (dims 0/1/2 contribute most; weight by RMS)
  C. baseline full-pose for comparison at matched K
"""
import sys, os, time, csv
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


def find_pose_patches_weighted(f1_all, f2_all, gt_poses, posenet, K, n_iter,
                                device, dim_weights):
    """Find K patches per pair, weighting pose-dim losses by dim_weights (6,)."""
    n = f1_all.shape[0]
    out = {}
    bs = 4
    dim_w = torch.tensor(dim_weights, device=device).float()
    for i in range(0, n, bs):
        e = min(i + bs, n); b = e - i
        f1 = f1_all[i:e].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[i:e].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[i:e].to(device).float()

        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = (((fp - gt_p) ** 2) * dim_w).sum()
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
            loss = (((fp - gt_p) ** 2) * dim_w).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)

        xs_np = xs_t.cpu().numpy().astype(np.uint16)
        ys_np = ys_t.cpu().numpy().astype(np.uint16)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi in range(b):
            xy = np.stack([xs_np[bi], ys_np[bi]], axis=1)
            out[i + bi] = (xy, d_np[bi])
    return out


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"Loading...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)

    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    base = se.eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=gen)
    base_score = se.compose_score(base["seg_dist"], base["pose_dist"], base["model_bytes"], 0)
    print(f"Baseline: score={base_score['score']:.4f} pose={base_score['pose_term']:.4f}", flush=True)

    posenet = load_posenet(device)
    csv_path = OUTPUT_DIR / "dim0_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "K", "n_iter", "sidecar_bytes",
                                 "score", "pose_term", "delta", "elapsed"])

    # dim_weights for various tests; len-6 vector
    weights_tests = [
        # name, weights, K, n_iter
        ("uniform_K5",    [1, 1, 1, 1, 1, 1],     5, 80),  # baseline reproduction
        ("dim0only_K5",   [1, 0, 0, 0, 0, 0],     5, 80),
        ("dim0only_K7",   [1, 0, 0, 0, 0, 0],     7, 80),
        ("dim012_K5",     [4, 1, 0.7, 0, 0, 0.7], 5, 80),  # rms-weighted
        ("dim012_K7",     [4, 1, 0.7, 0, 0, 0.7], 7, 80),
        ("rms_weighted_K5", [4, 1, 0.7, 0.1, 0.05, 0.7], 5, 80),
    ]

    for spec, w, K, n_iter in weights_tests:
        print(f"\n=== {spec} K={K} ===", flush=True)
        t0 = time.time()
        patches = find_pose_patches_weighted(f1_all, f2_all, data["val_poses"], posenet,
                                              K=K, n_iter=n_iter, device=device, dim_weights=w)
        elapsed = time.time() - t0
        sb = sparse_sidecar_size(patches)
        f1_p = apply_sparse_patches(f1_all, patches)
        result = se.eval_with_frames(f1_p, f2_all, data, device, gen_for_bytes=gen)
        full = se.compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        delta = full['score'] - base_score['score']
        print(f"  {spec}: sidecar={sb}B score={full['score']:.4f} "
              f"pose={full['pose_term']:.4f} delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, K, n_iter, sb, full['score'],
                                     full['pose_term'], delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
