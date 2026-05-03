#!/usr/bin/env python
"""
M1: Cross-pair codebook of compressed patch-bundles.

Cluster the 600 pairs in 6D pose space (k-means, K=16 clusters). For each cluster:
- Compute a single "shared correction template" of N template positions + deltas
  optimized to minimize the AVERAGE pose loss across pairs in the cluster.
- Per-pair sidecar: cluster id + small per-pair offsets to template positions.

Storage:
- Header: K templates, each = N positions + N (channel, delta) → K*N*7 bytes
- Per pair: cluster_id (4 bits if K≤16) + activation mask (N bits) + small residual

Test K=8, 16, 32 clusters, N=10, 20 templates.
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, get_pose6, load_posenet, estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_cluster_template(f1_all, f2_all, gt_poses, posenet, pair_indices,
                            N, n_iter, device):
    """Find N positions + (channel, delta) shared across pair_indices that
    minimize AVERAGE pose loss."""
    if not pair_indices:
        return []
    sel = torch.tensor(pair_indices, dtype=torch.long)
    f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
    f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
    gt_p = gt_poses[sel].to(device).float()
    b = len(pair_indices)

    # Initial gradient: AVERAGED across pairs in cluster
    f1_param = f1.clone().requires_grad_(True)
    pin = se.diff_posenet_input(f1_param, f2)
    fp = get_pose6(posenet, pin).float()
    loss = ((fp - gt_p) ** 2).sum()
    grad = torch.autograd.grad(loss, f1_param)[0]
    # Sum across batch to get average impact
    grad_mean = grad.abs().sum(dim=0)  # (3, H, W)
    max_chan_grad, best_chan = grad_mean.max(dim=0)  # (H, W)
    flat = max_chan_grad.contiguous().reshape(-1)
    _, topk = torch.topk(flat, N)
    ys_t = (topk // OUT_W).long()
    xs_t = (topk % OUT_W).long()
    chan_t = best_chan.reshape(-1)[topk]

    # Optimize SHARED delta per template position (averaged over pairs)
    cur_d = torch.zeros((N,), device=device, requires_grad=True)
    opt = torch.optim.Adam([cur_d], lr=2.0)
    batch_idx = torch.arange(b, device=device)
    for _ in range(n_iter):
        opt.zero_grad()
        f1_p = f1.clone()
        for k in range(N):
            c = chan_t[k].item()
            yy = ys_t[k]; xx = xs_t[k]
            f1_p[:, c, yy, xx] = f1_p[:, c, yy, xx] + cur_d[k]
        f1_p = f1_p.clamp(0, 255)
        pin = se.diff_posenet_input(f1_p, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).mean()  # MEAN across cluster
        loss.backward()
        opt.step()
        with torch.no_grad():
            cur_d.clamp_(-127, 127)

    template = []
    xs_np = xs_t.cpu().numpy().astype(np.uint16)
    ys_np = ys_t.cpu().numpy().astype(np.uint16)
    chan_np = chan_t.cpu().numpy().astype(np.uint8)
    d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
    for k in range(N):
        template.append((int(xs_np[k]), int(ys_np[k]), int(chan_np[k]), int(d_np[k])))
    return template


def codebook_sidecar_size(templates, pair_assignments, pair_residuals=None):
    """Templates is list of K templates (each list of N (x,y,c,d) tuples).
    pair_assignments: dict pair_i -> cluster_id.
    pair_residuals: dict pair_i -> list of (x,y,c,d) extra patches (can be None).
    """
    K = len(templates)
    parts = []
    # Header: u8 K, u8 N (assume same N per template), then templates
    if not templates or not templates[0]:
        return 0
    N = len(templates[0])
    parts.append(struct.pack("<BB", K, N))
    for tmpl in templates:
        for (x, y, c, d) in tmpl:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    # Per-pair: u16 pair_idx, u8 cluster_id, u8 n_residuals, then residuals (7B each)
    parts.append(struct.pack("<H", len(pair_assignments)))
    for pi in sorted(pair_assignments.keys()):
        cid = pair_assignments[pi]
        residuals = pair_residuals.get(pi, []) if pair_residuals else []
        parts.append(struct.pack("<HBB", pi, cid, len(residuals)))
        for (x, y, c, d) in residuals:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    raw = b''.join(parts)
    return len(bz2.compress(raw, compresslevel=9))


def apply_codebook_patches(f_all, templates, pair_assignments, pair_residuals=None):
    out = f_all.clone()
    for pi in pair_assignments:
        cid = pair_assignments[pi]
        arr = out[pi].float().numpy()
        for (x, y, c, d) in templates[cid]:
            arr[y, x, c] += d
        if pair_residuals and pi in pair_residuals:
            for (x, y, c, d) in pair_residuals[pi]:
                arr[y, x, c] += d
        out[pi] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
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
    poses = data["val_poses"].cpu().numpy()

    print(f"Baseline: {score_bl:.4f}, sb_mask={sb_mask}B")

    # Cluster pairs by 6D pose
    import csv
    csv_path = OUTPUT_DIR / "m1_codebook_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "K", "N_templates", "sb_total", "score", "pose_term", "delta", "elapsed"])

    for K, N in [(8, 10), (16, 10), (16, 20), (32, 10)]:
        print(f"\n=== M1: K={K} clusters x N={N} templates ===")
        t0 = time.time()
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=5).fit(poses[:600])
        labels = kmeans.labels_  # (600,)
        # Build cluster -> pair list
        cluster_pairs = {c: [] for c in range(K)}
        for pi, c in enumerate(labels):
            cluster_pairs[int(c)].append(pi)

        # Build template per cluster
        templates = []
        for c in range(K):
            tmpl = find_cluster_template(f1_new, f2_new, data["val_poses"], posenet,
                                            cluster_pairs[c], N=N, n_iter=80, device=device)
            templates.append(tmpl)

        pair_assignments = {pi: int(labels[pi]) for pi in range(600)}
        sb = codebook_sidecar_size(templates, pair_assignments)
        f1_combined = apply_codebook_patches(f1_new, templates, pair_assignments)
        s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
        full = fast_compose(s, p, model_bytes, sb_mask + sb)
        elapsed = time.time() - t0
        print(f"  M1 K={K} N={N}: sb={sb}B sb_total={sb_mask+sb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f"K{K}_N{N}", K, N, sb_mask+sb,
                                     full['score'], full['pose_term'],
                                     full['score']-score_bl, elapsed])


if __name__ == "__main__":
    main()
