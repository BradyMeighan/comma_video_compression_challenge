#!/usr/bin/env python
"""
Novel sidecar variants based on deep analysis findings:

Finding 1: After best sidecar, dim 0 is SOLVED (RMS 0.014), but dims 1/2/5
           are untouched. Full-pose gradient is now dominated by their tiny
           residuals; targeted dim-1+2+5 loss should give different direction.

Finding 2: Top-100 patch coords are reused 17-62× across pairs. A shared
           coordinate dictionary could halve the byte cost of coords.

Finding 3: 33/60 NEW hardest pairs after patching weren't in original
           top-60 — patches shifted the distribution.

Variants:
  A. base + wave2 on residual hardest 200 with dim-1+2+5 loss
  B. base + wave2 on dominant-dim targeting (per-pair custom dim mask)
  C. shared-coord dictionary format (different storage, same patches)
  D. base + wave2 with FULL loss but on residual top-200 (control vs A)
"""
import sys, os, time, csv, struct, bz2
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
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs, per_pair_pose_mse)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_pose_patches_weighted(cur_f1, f2_all, gt_poses, posenet, pair_indices,
                                 K, n_iter, device, dim_weights, exclusion=None):
    """K patches per pair, optimizing weighted-dim pose loss."""
    out = {}
    bs = 8
    dim_w = torch.tensor(dim_weights, device=device).float()
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


def find_pose_patches_per_pair_dim(cur_f1, f2_all, gt_poses, posenet,
                                     pair_indices, K, n_iter, device,
                                     per_pair_weights, exclusion=None):
    """Different dim_weights per pair (custom dominant-dim targeting).
    per_pair_weights: dict pair_i -> 6-tuple weights.
    """
    out = {}
    # Group pairs by their weight tuple to batch similar ones
    from collections import defaultdict
    groups = defaultdict(list)
    for pi in pair_indices:
        key = tuple(per_pair_weights[pi])
        groups[key].append(pi)

    for weight_key, pi_list in groups.items():
        result = find_pose_patches_weighted(
            cur_f1, f2_all, gt_poses, posenet, pi_list, K=K, n_iter=n_iter,
            device=device, dim_weights=list(weight_key), exclusion=exclusion)
        out.update(result)
    return out


def shared_coord_sidecar_size(patches_dict, n_dict=128):
    """Sparse format with a shared coordinate dictionary.
    Header:  u16 n_dict, then n_dict × (u16 x, u16 y) = n_dict*4 bytes
    Per pair:
      u16 pair_idx, u16 num_shared, num_shared × (u8 dict_idx, 3×i8 delta) = 4 bytes
                    u16 num_unique, num_unique × (u16 x, u16 y, 3×i8 delta) = 7 bytes
    """
    if not patches_dict:
        return 0
    # Build dictionary from top-N most-frequent coords
    from collections import Counter
    cnt = Counter()
    for pi, (xy, d) in patches_dict.items():
        for j in range(xy.shape[0]):
            cnt[(int(xy[j, 0]), int(xy[j, 1]))] += 1
    dictionary = [coord for coord, c in cnt.most_common(n_dict)]
    dict_idx = {coord: i for i, coord in enumerate(dictionary)}

    parts = [struct.pack("<H", len(dictionary))]
    for x, y in dictionary:
        parts.append(struct.pack("<HH", x, y))

    parts.append(struct.pack("<H", len(patches_dict)))
    for pair_i, (xy, d) in sorted(patches_dict.items()):
        shared = []; unique = []
        for j in range(xy.shape[0]):
            coord = (int(xy[j, 0]), int(xy[j, 1]))
            if coord in dict_idx and dict_idx[coord] < 256:
                shared.append((dict_idx[coord], int(d[j, 0]), int(d[j, 1]), int(d[j, 2])))
            else:
                unique.append((coord[0], coord[1], int(d[j, 0]), int(d[j, 1]), int(d[j, 2])))
        parts.append(struct.pack("<HHH", pair_i, len(shared), len(unique)))
        for di, dr, dg, db in shared:
            parts.append(struct.pack("<Bbbb", di, dr, dg, db))
        for x, y, dr, dg, db in unique:
            parts.append(struct.pack("<HHbbb", x, y, dr, dg, db))
    raw = b''.join(parts)
    return len(bz2.compress(raw, compresslevel=9))


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
    seg_b, pose_b = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg_b, pose_b, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    # Apply best sidecar (350_K7 + 250_K2)
    print("Applying best sidecar 350_K7+250_K2...", flush=True)
    p_top = find_pose_patches_for_pairs(
        f1_all, f2_all, data["val_poses"], posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_all, f2_all, data["val_poses"], posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    best_patches = {**p_top, **p_tail}
    cur_f1_base = apply_sparse_patches(f1_all, best_patches)
    sb_base = sparse_sidecar_size(best_patches)
    seg_p, pose_p = fast_eval(cur_f1_base, f2_all, data["val_rgb"], device)
    base_after = fast_compose(seg_p, pose_p, model_bytes, sb_base)
    print(f"After best base: score={base_after['score']:.4f} pose={base_after['pose_term']:.4f} "
          f"sb={sb_base}B (delta {base_after['score']-base['score']:+.4f})", flush=True)

    csv_path = OUTPUT_DIR / "novel_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_patched_pairs", "K_total",
                                 "sidecar_bytes", "score", "pose_term", "delta_vs_base", "delta_vs_baseline", "elapsed"])

    # Compute residual error and exclusion map
    res_pose = per_pair_pose_mse(cur_f1_base, f2_all, data["val_poses"], posenet, device)
    res_rank = np.argsort(res_pose)[::-1]
    print(f"Residual top-3 pairs: {res_rank[:3].tolist()} (MSE={res_pose[res_rank[:3]].round(5).tolist()})", flush=True)

    f1_excl = {}
    for pi, (xy, d) in best_patches.items():
        f1_excl[pi] = set((int(xy[j, 0]), int(xy[j, 1])) for j in range(xy.shape[0]))

    def run_variant(spec, label, patches_extra):
        all_patches = {}
        # merge base with extras
        all_keys = set(best_patches.keys()) | set(patches_extra.keys())
        for k in all_keys:
            if k in best_patches and k in patches_extra:
                xy = np.concatenate([best_patches[k][0], patches_extra[k][0]], axis=0)
                d = np.concatenate([best_patches[k][1], patches_extra[k][1]], axis=0)
                all_patches[k] = (xy, d)
            elif k in best_patches:
                all_patches[k] = best_patches[k]
            else:
                all_patches[k] = patches_extra[k]
        f1_p = apply_sparse_patches(f1_all, all_patches)
        seg, pose = fast_eval(f1_p, f2_all, data["val_rgb"], device)
        sb = sparse_sidecar_size(all_patches)
        full = fast_compose(seg, pose, model_bytes, sb)
        d_base = full['score'] - base_after['score']
        d_baseline = full['score'] - base['score']
        K_total = sum(xy.shape[0] for xy, d in all_patches.values())
        npairs = len(all_patches)
        print(f"  >> {spec}: pairs={npairs} K_total={K_total} sb={sb}B score={full['score']:.4f} "
              f"pose={full['pose_term']:.4f} d_vs_base={d_base:+.4f} d_vs_baseline={d_baseline:+.4f}", flush=True)
        return all_patches, sb, full

    # ─── Variant A: dim-1+2+5 weighted on residual hardest 200 ───
    print("\n=== A: dim 1+2+5 patches on residual top200 ===", flush=True)
    t0 = time.time()
    res200 = [int(x) for x in res_rank[:200]]
    # weights: emphasize dims 1, 2, 5 (use rms-inverse so each dim contributes equally to optimizer)
    rms = [0.014, 0.037, 0.029, 0.011, 0.008, 0.027]  # post-base residual rms per dim
    # reciprocal weights with cap to avoid blowing up small dims
    weights = [0, 1, 1, 0, 0, 1]  # binary mask on dims 1, 2, 5
    pA = find_pose_patches_weighted(
        cur_f1_base, f2_all, data["val_poses"], posenet,
        res200, K=3, n_iter=80, device=device, dim_weights=weights, exclusion=f1_excl)
    elapsed = time.time() - t0
    pA_all, sb_A, full_A = run_variant("A_dim125_K3_res200", "A", pA)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["A_dim125_K3_res200", len(pA_all),
                                 sum(xy.shape[0] for xy, d in pA_all.values()),
                                 sb_A, full_A['score'], full_A['pose_term'],
                                 full_A['score']-base_after['score'],
                                 full_A['score']-base['score'], elapsed])

    # ─── Variant A2: dim-1+2+5 K=5 (more bytes) ───
    print("\n=== A2: dim 1+2+5 K=5 res200 ===", flush=True)
    t0 = time.time()
    pA2 = find_pose_patches_weighted(
        cur_f1_base, f2_all, data["val_poses"], posenet,
        res200, K=5, n_iter=80, device=device, dim_weights=weights, exclusion=f1_excl)
    elapsed = time.time() - t0
    pA2_all, sb_A2, full_A2 = run_variant("A2_dim125_K5_res200", "A2", pA2)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["A2_dim125_K5_res200", len(pA2_all),
                                 sum(xy.shape[0] for xy, d in pA2_all.values()),
                                 sb_A2, full_A2['score'], full_A2['pose_term'],
                                 full_A2['score']-base_after['score'],
                                 full_A2['score']-base['score'], elapsed])

    # ─── Variant B: per-pair dominant-dim targeting on residual top 200 ───
    print("\n=== B: per-pair dominant-dim K=3 res200 ===", flush=True)
    t0 = time.time()
    # Compute per-pair residual dim errors
    from sidecar_deep_analyze import per_pair_pose_dim_errs
    res_dim = per_pair_pose_dim_errs(cur_f1_base, f2_all, data["val_poses"], posenet, device)
    per_pair_w = {}
    for pi in res200:
        if res_dim[pi].sum() < 1e-9:
            per_pair_w[pi] = (0, 1, 1, 0, 0, 1)  # default to dim125
        else:
            frac = res_dim[pi] / res_dim[pi].sum()
            # Activate dims that contribute >15% of pair's residual
            w = tuple(1 if f > 0.15 else 0 for f in frac)
            if sum(w) == 0:
                w = (0, 1, 1, 0, 0, 1)
            per_pair_w[pi] = w
    pB = find_pose_patches_per_pair_dim(
        cur_f1_base, f2_all, data["val_poses"], posenet,
        res200, K=3, n_iter=80, device=device,
        per_pair_weights=per_pair_w, exclusion=f1_excl)
    elapsed = time.time() - t0
    pB_all, sb_B, full_B = run_variant("B_perdim_K3_res200", "B", pB)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["B_perdim_K3_res200", len(pB_all),
                                 sum(xy.shape[0] for xy, d in pB_all.values()),
                                 sb_B, full_B['score'], full_B['pose_term'],
                                 full_B['score']-base_after['score'],
                                 full_B['score']-base['score'], elapsed])

    # ─── Variant C: shared-coord dictionary on BASE patches (storage only) ───
    print("\n=== C: shared-coord dictionary on best base patches ===", flush=True)
    for n_dict in [64, 128, 256]:
        sb_c = shared_coord_sidecar_size(best_patches, n_dict=n_dict)
        savings = sb_base - sb_c
        rate_save = savings / 37545489 * 25
        new_score = base_after['score'] - rate_save
        print(f"  dict_size={n_dict}: sb={sb_c}B (saved {savings}B vs sparse), "
              f"would-be score={new_score:.4f} delta={new_score-base['score']:+.4f}", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f"C_shared_dict_{n_dict}", len(best_patches),
                                     sum(xy.shape[0] for xy, d in best_patches.values()),
                                     sb_c, new_score, base_after['pose_term'],
                                     new_score-base_after['score'],
                                     new_score-base['score'], 0])

    # ─── Variant D: full-pose K=3 on residual top 200 (control vs A) ───
    print("\n=== D: control: full-pose K=3 res200 (no dim weighting) ===", flush=True)
    t0 = time.time()
    pD = find_pose_patches_for_pairs(
        cur_f1_base, f2_all, data["val_poses"], posenet,
        res200, K=3, n_iter=80, device=device, exclusion=f1_excl)
    elapsed = time.time() - t0
    pD_all, sb_D, full_D = run_variant("D_fullpose_K3_res200", "D", pD)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["D_fullpose_K3_res200", len(pD_all),
                                 sum(xy.shape[0] for xy, d in pD_all.values()),
                                 sb_D, full_D['score'], full_D['pose_term'],
                                 full_D['score']-base_after['score'],
                                 full_D['score']-base['score'], elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
