#!/usr/bin/env python
"""
Stacked sidecars: apply wave 1, recompute pose error per pair, add wave 2 at
locations chosen by gradient on the PATCHED frames. Test compounding.

Speedups vs sidecar_adaptive:
  - DistortionNet loaded ONCE (cached) — saves ~3s/eval
  - Initial PoseNet grad maps cached per pair (per-baseline-frame)
  - Larger batch sizes (bs=8 for patch search) where memory allows
"""
import sys, os, time, csv, struct, bz2, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, get_pose6, load_posenet, gpu_cleanup,
                     MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE, estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Cached DistortionNet (lifetime of process) ────────────────────────────
_DIST_NET = None

def get_dist_net(device):
    global _DIST_NET
    if _DIST_NET is None:
        from modules import DistortionNet
        from prepare import segnet_sd_path, posenet_sd_path
        _DIST_NET = DistortionNet().eval().to(device)
        _DIST_NET.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    return _DIST_NET


def fast_eval(f1_all, f2_all, val_rgb, device):
    """eval_with_frames but with cached DistortionNet."""
    dn = get_dist_net(device)
    n = f1_all.shape[0]
    t_seg, t_pose = 0.0, 0.0
    bs = 16
    with torch.inference_mode():
        for i in range(0, n, bs):
            f1 = f1_all[i:i+bs].to(device)
            f2 = f2_all[i:i+bs].to(device)
            comp = torch.stack([f1, f2], dim=1)
            gt = val_rgb[i:i+bs].to(device)
            pd, sd = dn.compute_distortion(gt, comp)
            t_seg += sd.sum().item(); t_pose += pd.sum().item()
    return t_seg / n, t_pose / n


def fast_compose(seg_dist, pose_dist, model_bytes, sidecar_bytes):
    total = MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes
    rate = total / UNCOMPRESSED_SIZE
    return {
        "score": 100*seg_dist + math.sqrt(max(0, 10*pose_dist)) + 25*rate,
        "seg_term": 100 * seg_dist,
        "pose_term": math.sqrt(max(0, 10*pose_dist)),
        "rate_term": 25 * rate,
    }


def per_pair_pose_mse(f1_all, f2_all, val_poses, posenet, device):
    """Compute per-pair pose MSE for given (possibly patched) frames."""
    n = f1_all.shape[0]
    out = np.zeros(n, dtype=np.float32)
    bs = 16
    with torch.inference_mode():
        for i in range(0, n, bs):
            e = min(i + bs, n)
            f1 = f1_all[i:e].to(device).float().permute(0, 3, 1, 2)
            f2 = f2_all[i:e].to(device).float().permute(0, 3, 1, 2)
            pin = se.diff_posenet_input(f1, f2)
            fp = get_pose6(posenet, pin).float()
            err = (fp - val_poses[i:e].to(device).float()).pow(2).mean(dim=1)
            out[i:e] = err.cpu().numpy()
    return out


def find_pose_patches_for_pairs(cur_f1, f2_all, gt_poses, posenet,
                                  pair_indices, K, n_iter, device,
                                  exclusion=None):
    """Find K patches per pair from cur_f1 (already-patched) frames.
    exclusion: dict pair_i -> set of (x, y) to skip in topk.
    Returns dict pair_i -> (xy(K,2) uint16, d(K,3) int8).
    """
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = cur_f1[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        gmag = grad.abs().sum(dim=1)

        if exclusion is not None:
            for bi, pair_i in enumerate(idx_list):
                excl = exclusion.get(pair_i, set())
                for (xx, yy) in excl:
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

    print("Loading model + data...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    print("Generating frames...", flush=True)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)

    # Baseline (cached eval)
    seg, pose = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg, pose, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} seg={base['seg_term']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair_orig = per_pair_pose_mse(f1_all, f2_all, data["val_poses"], posenet, device)
    rank_orig = np.argsort(pose_per_pair_orig)[::-1]

    csv_path = OUTPUT_DIR / "stack_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "wave_specs", "n_pairs_total", "total_K",
                                 "sidecar_bytes", "score", "pose_term", "delta", "elapsed"])

    def write_row(spec, wave_specs, all_patches, t_total):
        sb = sparse_sidecar_size(all_patches)
        f1_p = apply_sparse_patches(f1_all, all_patches)
        seg, pose = fast_eval(f1_p, f2_all, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb)
        delta = full['score'] - base['score']
        npairs = len(all_patches)
        total_K = sum(xy.shape[0] for xy, d in all_patches.values())
        print(f"  >> {spec}: pairs={npairs} K_total={total_K} sidecar={sb}B score={full['score']:.4f} "
              f"pose={full['pose_term']:.4f} delta={delta:+.4f} elapsed={t_total:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, wave_specs, npairs, total_K, sb,
                                     full['score'], full['pose_term'], delta, t_total])
        return f1_p

    # ── Test 1: stack on top of best (top300_K7) ──
    # Wave 1: top300_K7 (the current best adaptive)
    # Wave 2: extend to next 100/200 pairs OR add more patches at residual locations on same 300
    print("\n=== Stack experiment: top300_K7 base + waves ===", flush=True)
    t0 = time.time()
    base_pairs = [int(x) for x in rank_orig[:300]]
    base_patches = find_pose_patches_for_pairs(
        f1_all, f2_all, data["val_poses"], posenet, base_pairs, K=7, n_iter=80, device=device)
    t_base = time.time() - t0
    print(f"  wave1 (top300_K7): {t_base:.1f}s", flush=True)
    write_row("base_top300_K7", "[K7]", base_patches, t_base)

    # ── Variant A: same 300 pairs, add wave 2 with K=5 at residual locations ──
    cur_f1 = apply_sparse_patches(f1_all, base_patches)
    exclusion = {pi: set((int(xy[j, 0]), int(xy[j, 1]))
                         for j in range(xy.shape[0]))
                 for pi, (xy, d) in base_patches.items()}

    t1 = time.time()
    wave2_a = find_pose_patches_for_pairs(
        cur_f1, f2_all, data["val_poses"], posenet, base_pairs, K=5, n_iter=80,
        device=device, exclusion=exclusion)
    merged_a = merge_patches(base_patches, wave2_a)
    t_a = time.time() - t1
    print(f"  wave2 (same 300, K=5 residual): {t_a:.1f}s", flush=True)
    write_row("stack_top300_K7+K5_same", "[K7,K5_same]", merged_a, t_base + t_a)

    # ── Variant B: extend to next 100 pairs (rank 300-400), K=7 ──
    next_100 = [int(x) for x in rank_orig[300:400]]
    t1 = time.time()
    wave2_b = find_pose_patches_for_pairs(
        f1_all, f2_all, data["val_poses"], posenet, next_100, K=7, n_iter=80, device=device)
    merged_b = merge_patches(base_patches, wave2_b)
    t_b = time.time() - t1
    print(f"  wave2 (next 100, K=7): {t_b:.1f}s", flush=True)
    write_row("stack_top300_K7+next100_K7", "[K7,K7_next100]", merged_b, t_base + t_b)

    # ── Variant C: combine A + B (residual on 300 + new 100) ──
    merged_c = merge_patches(merged_a, wave2_b)
    write_row("stack_top300_K7+K5_same+next100_K7", "[K7,K5_same,K7_next100]", merged_c,
              t_base + t_a + t_b)

    # ── Variant D: 3 waves on same 300 pairs (K=7, K=5, K=3) ──
    cur_f1 = apply_sparse_patches(f1_all, merged_a)
    excl_a = {pi: set((int(xy[j, 0]), int(xy[j, 1]))
                      for j in range(xy.shape[0]))
              for pi, (xy, d) in merged_a.items()}
    t1 = time.time()
    wave3_d = find_pose_patches_for_pairs(
        cur_f1, f2_all, data["val_poses"], posenet, base_pairs, K=3, n_iter=80,
        device=device, exclusion=excl_a)
    merged_d = merge_patches(merged_a, wave3_d)
    t_d = time.time() - t1
    print(f"  wave3 (same 300, K=3 residual^2): {t_d:.1f}s", flush=True)
    write_row("stack_top300_K7+K5+K3_same", "[K7,K5,K3_same]", merged_d, t_base + t_a + t_d)

    # ── Variant E: re-rank by residual error after wave 1, target NEW top-300 ──
    # This finds pairs that became HARD after wave 1 (overshoot/undercorrected)
    cur_f1 = apply_sparse_patches(f1_all, base_patches)
    pose_after_w1 = per_pair_pose_mse(cur_f1, f2_all, data["val_poses"], posenet, device)
    new_rank = np.argsort(pose_after_w1)[::-1]
    new_top_100 = [int(x) for x in new_rank[:100]]
    print(f"  rerank after w1: top-100 hardest now have pose_mse={pose_after_w1[new_rank[:5]].round(5).tolist()}", flush=True)
    t1 = time.time()
    wave2_e = find_pose_patches_for_pairs(
        cur_f1, f2_all, data["val_poses"], posenet, new_top_100, K=7, n_iter=80,
        device=device,
        exclusion={pi: set((int(xy[j, 0]), int(xy[j, 1]))
                            for j in range(xy.shape[0]))
                   for pi, (xy, d) in base_patches.items() if pi in new_top_100})
    merged_e = merge_patches(base_patches, wave2_e)
    t_e = time.time() - t1
    write_row("stack_top300_K7+rerank100_K7", "[K7,rerank100_K7]", merged_e, t_base + t_e)

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
