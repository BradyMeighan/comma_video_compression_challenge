"""Brute-force tournament: try many sidecar hypotheses on the top-N hardest pairs,
report per-hypothesis byte cost and per-pair score contribution.

Output: ranked table with cost/benefit per hypothesis. Best winner can then be
scaled up to full pair set.

Hypotheses tested:
  H1  f1 RGB pose-gradient K=5  (CURRENT METHOD)
  H2  f2 RGB pose-gradient K=5
  H3  f1 K=2 + f2 K=3 pose-gradient (split)
  H4  f1 K=3 + f2 K=2 pose-gradient (split)
  H5  f2 RGB seg-CE-gradient K=5
  H6  f2 RGB seg-CE-gradient K=10
  H7  f2 RGB seg-CE-gradient K=2
  H8  f1 RGB pose-gradient K=2 (less)
  H9  f1 RGB pose-gradient K=8 (more)
  H10 f2 RGB pose-gradient K=8 (more)

Each hypothesis is evaluated on the SAME 100 pairs (top 100 by pose loss after mask methods).
We compute the seg/pose distortion delta vs no-patch baseline + byte cost.
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import State, compose_score
from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6,
                      load_segnet, load_posenet, MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE)
import sidecar_explore as se
from sidecar_stack import fast_eval
from sidecar_mask_verified import regenerate_frames_from_masks
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from sidecar_seg_only import (find_seg_patches_f2, seg_patches_sidecar_size,
                                apply_seg_patches_f2, _segnet_logits)

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
N_TEST_PAIRS = int(os.environ.get("N_TEST", "100"))


def find_pose_patches_f2(f1_all, f2_all, gt_poses, posenet, pair_indices,
                            K, n_iter, device):
    """Like find_channel_only_patches but operates on F2 (modifying f2)."""
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        f2_param = f2.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1, f2_param)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f2_param)[0]
        grad_abs = grad.abs()
        max_chan_grad, best_chan = grad_abs.max(dim=1)
        flat = max_chan_grad.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        chan_t = best_chan[batch_idx, ys_t, xs_t]

        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        for _ in range(n_iter):
            opt.zero_grad()
            f2_p = f2.clone()
            for c in range(3):
                mask_c = (chan_t == c)
                if mask_c.any():
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    yy = ys_t[rows_b, cols_k]
                    xx = xs_t[rows_b, cols_k]
                    dd = cur_d[rows_b, cols_k]
                    f2_p[rows_b, c, yy, xx] = f2_p[rows_b, c, yy, xx] + dd
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
        chan_np = chan_t.cpu().numpy().astype(np.uint8)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
            patches = [p for p in patches if p[3] != 0]
            if patches:
                out[int(pair_i)] = patches
    return out


def apply_channel_patches_f2(f2_all, patches):
    """Apply RGB-style patches to f2 (modify f2 instead of f1)."""
    out = f2_all.clone()
    for pair_i, ps in patches.items():
        for tup in ps:
            x, y, c, d = tup[0], tup[1], tup[2], tup[3]
            v = int(out[pair_i, y, x, c]) + int(d)
            out[pair_i, y, x, c] = max(0, min(255, v))
    return out


def score_subset(f1_subset, f2_subset, val_rgb_subset, model_bytes, sb_extra, device,
                   reference_seg, reference_pose):
    """Score on a SUBSET of pairs. Returns (seg, pose, score_contribution).
    score_contribution = how much this subset contributes to the full 600-pair score
    using the actual mean-based formula.
    """
    seg_d, pose_d = fast_eval(f1_subset, f2_subset, val_rgb_subset, device)
    return seg_d, pose_d


def main():
    import csv
    s = State()
    print(f"\n=== Hypothesis tournament on top {N_TEST_PAIRS} pairs ===", flush=True)

    # Load cache
    if not (CACHE_DIR / "frames_x5.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder first.")
    cache_frames = torch.load(CACHE_DIR / "frames_x5.pt", weights_only=False)
    f1_base = cache_frames['f1']  # uint8 (n, OUT_H, OUT_W, 3)
    f2_base = cache_frames['f2']
    n = f1_base.shape[0]
    print(f"  cache loaded: {n} pairs base frames", flush=True)

    # Pick test pairs: top N hardest by current pose distortion (against base frames)
    print("  computing per-pair pose loss on base frames ...", flush=True)
    from sidecar_stack import per_pair_pose_mse
    pose_per_pair = per_pair_pose_mse(f1_base, f2_base, s.poses, s.posenet, s.device)
    rank = np.argsort(pose_per_pair)[::-1]
    test_pairs = [int(x) for x in rank[:N_TEST_PAIRS]]
    print(f"  top-3 pose losses: {pose_per_pair[rank[:3]]}", flush=True)

    # val_rgb subset
    val_rgb = s.data["val_rgb"]
    val_rgb_test = val_rgb[test_pairs]

    # Subset frames
    f1_test = f1_base[test_pairs].clone()
    f2_test = f2_base[test_pairs].clone()

    # Baseline (no patches)
    seg_base, pose_base = fast_eval(f1_test, f2_test, val_rgb_test, s.device)
    print(f"\n  BASELINE (no patches, {N_TEST_PAIRS} pairs): seg={seg_base:.6f} pose={pose_base:.6f}", flush=True)

    segnet = load_segnet(s.device)
    f2_target_full = val_rgb[:, 1] if val_rgb.dim() == 5 else val_rgb

    results = []

    def run_hypothesis(name, frames_fn, sidecar_bytes_fn, **kwargs):
        """frames_fn: returns (f1_patched, f2_patched) for test pairs.
           sidecar_bytes_fn: returns total bytes of patches."""
        t0 = time.time()
        f1_h, f2_h, sb = frames_fn(**kwargs)
        seg_d, pose_d = fast_eval(f1_h, f2_h, val_rgb_test, s.device)
        elapsed = time.time() - t0
        # Score formula contribution: seg/pose are means. The byte cost contributes 25 * sb / U
        # to FULL-pipeline rate. Per-pair impact on score:
        # delta_score = 100*(seg_d - seg_base) + sqrt(10*pose_d) - sqrt(10*pose_base) + 25 * sb / UNCOMP
        # But for ranking on this subset, just compute the seg/pose deltas + byte cost
        d_seg = (seg_d - seg_base) * 100
        from math import sqrt
        d_pose = sqrt(max(0, 10*pose_d)) - sqrt(max(0, 10*pose_base))
        # Approximate byte contribution to full score: 25 * sb / UNCOMPRESSED_SIZE
        # but sb here is for N_TEST_PAIRS, so for full 600 we'd scale
        d_bytes = 25 * sb / UNCOMPRESSED_SIZE
        net = d_seg + d_pose + d_bytes
        results.append({
            'name': name, 'sb': sb, 'seg_d': seg_d, 'pose_d': pose_d,
            'd_seg': d_seg, 'd_pose': d_pose, 'd_bytes': d_bytes, 'net': net,
            'time': elapsed,
        })
        print(f"  {name:<35} sb={sb:>5}B seg={seg_d:.5f} pose={pose_d:.5f}  "
              f"Δseg={d_seg:+.4f} Δpose={d_pose:+.4f} Δbytes={d_bytes:+.4f}  "
              f"NET={net:+.4f} ({elapsed:.0f}s)", flush=True)

    # ── H1: f1 RGB pose-gradient K=5 (current) ──────────────────────────
    def h1():
        patches = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet,
                                              test_pairs, K=5, n_iter=80, device=s.device)
        f1_p = apply_channel_patches(f1_base, patches)[test_pairs].clone()
        f2_p = f2_base[test_pairs].clone()
        sb = channel_sidecar_size({pi - test_pairs[0] if False else pi: ps for pi, ps in patches.items()})
        return f1_p, f2_p, sb
    run_hypothesis("H1_f1_pose_K5 (current)", lambda: h1(), None)

    # ── H2: f2 RGB pose-gradient K=5 ────────────────────────────────────
    def h2():
        patches = find_pose_patches_f2(f1_base, f2_base, s.poses, s.posenet,
                                          test_pairs, K=5, n_iter=80, device=s.device)
        f1_p = f1_base[test_pairs].clone()
        f2_p = apply_channel_patches_f2(f2_base, patches)[test_pairs].clone()
        sb = channel_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H2_f2_pose_K5", lambda: h2(), None)

    # ── H3: f1 K=2 + f2 K=3 ─────────────────────────────────────────────
    def h3():
        p1 = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet,
                                         test_pairs, K=2, n_iter=80, device=s.device)
        f1_after = apply_channel_patches(f1_base, p1)
        p2 = find_pose_patches_f2(f1_after, f2_base, s.poses, s.posenet,
                                     test_pairs, K=3, n_iter=80, device=s.device)
        f2_after = apply_channel_patches_f2(f2_base, p2)
        sb = channel_sidecar_size(p1) + channel_sidecar_size(p2)
        return f1_after[test_pairs].clone(), f2_after[test_pairs].clone(), sb
    run_hypothesis("H3_f1K2+f2K3_pose", lambda: h3(), None)

    # ── H4: f1 K=3 + f2 K=2 ─────────────────────────────────────────────
    def h4():
        p1 = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet,
                                         test_pairs, K=3, n_iter=80, device=s.device)
        f1_after = apply_channel_patches(f1_base, p1)
        p2 = find_pose_patches_f2(f1_after, f2_base, s.poses, s.posenet,
                                     test_pairs, K=2, n_iter=80, device=s.device)
        f2_after = apply_channel_patches_f2(f2_base, p2)
        sb = channel_sidecar_size(p1) + channel_sidecar_size(p2)
        return f1_after[test_pairs].clone(), f2_after[test_pairs].clone(), sb
    run_hypothesis("H4_f1K3+f2K2_pose", lambda: h4(), None)

    # ── H5: f2 RGB seg-CE-gradient K=5 ──────────────────────────────────
    def h5():
        patches = find_seg_patches_f2(f2_base, f2_target_full, segnet,
                                         test_pairs, K=5, n_iter=80, device=s.device)
        f1_p = f1_base[test_pairs].clone()
        f2_p = apply_seg_patches_f2(f2_base, patches)[test_pairs].clone()
        sb, _ = seg_patches_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H5_f2_seg_K5", lambda: h5(), None)

    # ── H6: f2 RGB seg-CE-gradient K=10 ─────────────────────────────────
    def h6():
        patches = find_seg_patches_f2(f2_base, f2_target_full, segnet,
                                         test_pairs, K=10, n_iter=80, device=s.device)
        f1_p = f1_base[test_pairs].clone()
        f2_p = apply_seg_patches_f2(f2_base, patches)[test_pairs].clone()
        sb, _ = seg_patches_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H6_f2_seg_K10", lambda: h6(), None)

    # ── H7: f2 RGB seg-CE-gradient K=2 ──────────────────────────────────
    def h7():
        patches = find_seg_patches_f2(f2_base, f2_target_full, segnet,
                                         test_pairs, K=2, n_iter=80, device=s.device)
        f1_p = f1_base[test_pairs].clone()
        f2_p = apply_seg_patches_f2(f2_base, patches)[test_pairs].clone()
        sb, _ = seg_patches_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H7_f2_seg_K2", lambda: h7(), None)

    # ── H8: f1 RGB pose-gradient K=2 ────────────────────────────────────
    def h8():
        patches = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet,
                                              test_pairs, K=2, n_iter=80, device=s.device)
        f1_p = apply_channel_patches(f1_base, patches)[test_pairs].clone()
        f2_p = f2_base[test_pairs].clone()
        sb = channel_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H8_f1_pose_K2", lambda: h8(), None)

    # ── H9: f1 RGB pose-gradient K=8 ────────────────────────────────────
    def h9():
        patches = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet,
                                              test_pairs, K=8, n_iter=80, device=s.device)
        f1_p = apply_channel_patches(f1_base, patches)[test_pairs].clone()
        f2_p = f2_base[test_pairs].clone()
        sb = channel_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H9_f1_pose_K8", lambda: h9(), None)

    # ── H10: f2 RGB pose-gradient K=8 ───────────────────────────────────
    def h10():
        patches = find_pose_patches_f2(f1_base, f2_base, s.poses, s.posenet,
                                          test_pairs, K=8, n_iter=80, device=s.device)
        f1_p = f1_base[test_pairs].clone()
        f2_p = apply_channel_patches_f2(f2_base, patches)[test_pairs].clone()
        sb = channel_sidecar_size(patches)
        return f1_p, f2_p, sb
    run_hypothesis("H10_f2_pose_K8", lambda: h10(), None)

    # ── Sort + report ───────────────────────────────────────────────────
    print(f"\n=== TOURNAMENT RESULTS (sorted by NET score impact, lower=better) ===")
    print(f"{'rank':<5}{'name':<35}{'sb':>6}{'seg_term':>10}{'pose_term':>10}{'NET':>10}")
    print("-" * 85)
    results.sort(key=lambda r: r['net'])
    for i, r in enumerate(results):
        print(f"{i+1:<5}{r['name']:<35}{r['sb']:>6}{100*r['seg_d']:>10.4f}{r['d_pose']+ (10*pose_base)**0.5:>10.4f}{r['net']:>+10.4f}")

    out_csv = OUTPUT_DIR / "v2_hypothesis_tournament_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "sb_bytes", "seg_d", "pose_d",
                     "d_seg", "d_pose", "d_bytes", "net", "elapsed_s"])
        for i, r in enumerate(results):
            w.writerow([i+1, r['name'], r['sb'], r['seg_d'], r['pose_d'],
                        r['d_seg'], r['d_pose'], r['d_bytes'], r['net'], r['time']])
    print(f"\nresults: {out_csv}")
    print(f"\n[hint] Best hypothesis to scale up: {results[0]['name']}")


if __name__ == "__main__":
    main()
