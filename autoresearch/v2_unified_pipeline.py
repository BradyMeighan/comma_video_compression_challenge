"""Unified pipeline — combine ALL known sidecar wins for the e80 model.

Stacks:
  1. X2 + CMA-ES + S2 patterns + C3 pose deltas (per-pair selected)
     - C3 uses WIDER delta range (-5..5) per tournament v2 winner
  2. RGB on f1 — K=2 iterative (2 passes of K=1) per tournament v2 winner
     - Tier reallocated: top T1 K=2 + T2 K=1 (sweep T1, T2 to find best)
  3. LZMA2 wrap on combined raw stream

Goal: drive score below 0.290, ideally 0.2899 or better.
"""
import sys, os, pickle, time, struct, bz2, lzma
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import State, compose_score, serialize_pattern_mask, serialize_block_mask_v2
from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, MASK_BYTES,
                      POSE_BYTES, UNCOMPRESSED_SIZE)
from v2_s2_strip_cmaes import cma_es_pattern_for_pair, PATTERN_SIZES
from v2_c3_pose_vector import (find_pose_deltas_gridsearch, serialize_pose_deltas,
                                  apply_pose_deltas_and_regen_full)
from v2_per_pair_select import (per_pair_select, collect_pattern_patches)
from v2_tournament_v2 import find_channel_iterative
from sidecar_stack import fast_eval, per_pair_pose_mse
from sidecar_mask_verified import (regenerate_frames_from_masks, mask_sidecar_size)
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from explore_x2_mask_blocks import block_mask_sidecar_size

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def build_combined_raw_v8(x2_p, cma_p, pat_p, pose_p, rgb_p, target_dims=(1, 2, 5)):
    """V8 raw stream — same as V6 but with per-stream lengths in u16 and pose dims encoded in header."""
    parts = [b'V8\x00\x01']
    parts.append(struct.pack("<I", len(x2_p)))
    for pi in sorted(x2_p.keys()):
        ps = x2_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for tup in ps:
            parts.append(struct.pack("<HHB", tup[0], tup[1], tup[2]))
    parts.append(struct.pack("<I", len(cma_p)))
    for pi in sorted(cma_p.keys()):
        ps = cma_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c) in ps:
            parts.append(struct.pack("<HHB", x, y, c))
    parts.append(struct.pack("<I", len(pat_p)))
    for pi in sorted(pat_p.keys()):
        ps = pat_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, p_id, c) in ps:
            parts.append(struct.pack("<HHBB", x, y, p_id, c))
    parts.append(struct.pack("<I", len(pose_p)))
    n_dims = len(target_dims)
    for pi in sorted(pose_p.keys()):
        d = pose_p[pi]
        parts.append(struct.pack("<H", pi))
        for dim in target_dims:
            parts.append(struct.pack("<b", int(d[dim])))
    parts.append(struct.pack("<I", len(rgb_p)))
    for pi in sorted(rgb_p.keys()):
        ps = rgb_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    return b''.join(parts)


def main():
    import csv
    s = State()
    target_dims = (1, 2, 5)
    wider_range = (-5, -3, -1, 0, 1, 3, 5)
    print(f"\n=== Unified pipeline (e80) ===", flush=True)
    print(f"  upgrades: C3 wider range {wider_range}, RGB iterative K=1x2, slim RGB tier", flush=True)

    # ── Load cache ──────────────────────────────────────────────────────
    if not (CACHE_DIR / "masks_x5.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder first.")
    masks_after_x2 = torch.load(CACHE_DIR / "masks_after_x2.pt", weights_only=False)
    masks_x5 = torch.load(CACHE_DIR / "masks_x5.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", 'rb') as f:
        x2_patches = pickle.load(f)
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", 'rb') as f:
        cmaes_patches = pickle.load(f)
    print(f"  cache loaded: x2={len(x2_patches)} cmaes={len(cmaes_patches)}", flush=True)

    # ── Re-collect S2 patterns ──────────────────────────────────────────
    print("\n=== unified: collecting S2 patterns ===", flush=True)
    pattern_patches = collect_pattern_patches(s, masks_after_x2,
                                                 [int(x) for x in s.rank[:100]])

    # ── Re-collect C3 pose deltas with WIDER range ──────────────────────
    print("\n=== unified: collecting C3 pose deltas (WIDER range ±5) ===", flush=True)
    t0 = time.time()
    pose_deltas = find_pose_deltas_gridsearch(
        s.gen, masks_x5, s.poses, s.posenet,
        [int(x) for x in s.rank[:200]], s.device,
        target_dims=target_dims, delta_range=wider_range)
    print(f"  pose_deltas: {len(pose_deltas)} pairs ({time.time()-t0:.0f}s)", flush=True)

    # ── Per-pair selection ──────────────────────────────────────────────
    print("\n=== unified: per-pair selection over top 600 ===", flush=True)
    top_pairs = [int(x) for x in s.rank[:600]]
    selected = per_pair_select(s, s.masks_cpu, x2_patches, cmaes_patches,
                                  pattern_patches, pose_deltas, top_pairs, s.poses)
    sel_x2 = {pi: x2_patches[pi] for pi, c in selected.items() if c[0] and pi in x2_patches}
    sel_cmaes = {pi: cmaes_patches[pi] for pi, c in selected.items() if c[1] and pi in cmaes_patches}
    sel_pattern = {pi: pattern_patches[pi] for pi, c in selected.items() if c[2] and pi in pattern_patches}
    sel_pose = {pi: pose_deltas[pi] for pi, c in selected.items() if c[3] and pi in pose_deltas}

    # ── Build mask state + apply pose deltas + regen ────────────────────
    final_masks = s.masks_cpu.clone()
    for pi, ps in sel_x2.items():
        for (x, y, c) in ps:
            final_masks[pi, y:y+2, x:x+2] = c
    for pi, ps in sel_pattern.items():
        for (x, y, p_id, c) in ps:
            ph, pw = PATTERN_SIZES[int(p_id)]
            yy_end = min(y + ph, MODEL_H); xx_end = min(x + pw, MODEL_W)
            final_masks[pi, y:yy_end, x:xx_end] = c
    for pi, ps in sel_cmaes.items():
        for (x, y, c) in ps:
            final_masks[pi, y, x] = c

    print("\n=== unified: regen frames ===", flush=True)
    scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=s.device)
    if sel_pose:
        f1_new, f2_new = apply_pose_deltas_and_regen_full(
            s.gen, final_masks, s.poses, sel_pose, s.device, scale, target_dims)
    else:
        f1_new, f2_new = regenerate_frames_from_masks(s.gen, final_masks, s.poses, s.device)

    # ── RGB tier sweep — try several allocations to find best ──────────
    print("\n=== unified: sweeping RGB tier configs ===", flush=True)

    # Order test pairs by current pose loss against new frames (post per-pair selection)
    from sidecar_stack import per_pair_pose_mse as ppmse
    pose_per_pair_now = ppmse(f1_new, f2_new, s.poses, s.posenet, s.device)
    rgb_rank = np.argsort(pose_per_pair_now)[::-1]

    # Try several allocations
    rgb_configs = [
        ("rgb_top500_K2single", [(500, 2)]),                # 500 pairs K=2 single pass
        ("rgb_top500_K1x2iter", [(500, ('iter', 1, 2))]),   # 500 pairs K=1 with 2 iter passes (=K=2 effective)
        ("rgb_top700_K1x2iter", [(700, ('iter', 1, 2))]),   # extend coverage
        ("rgb_top250_K3+250_K1", [(250, 3), (250, 1)]),     # tier
        ("rgb_top100_K3+400_K1", [(100, 3), (400, 1)]),     # heavier on top 100
        ("rgb_top250_K5+250_K2", [(250, 5), (250, 2)]),     # current allocation (reference)
    ]

    best_score = float('inf')
    best_config = None
    rgb_results = []

    for cfg_name, tiers in rgb_configs:
        print(f"\n  -- RGB config: {cfg_name} --", flush=True)
        cur_f1 = f1_new.clone()
        all_rgb_patches = {}
        idx = 0
        for tier_idx, (n_pairs, K_spec) in enumerate(tiers):
            tier_pair_ids = [int(x) for x in rgb_rank[idx:idx+n_pairs]]
            idx += n_pairs
            if isinstance(K_spec, tuple) and K_spec[0] == 'iter':
                _, K_each, n_passes = K_spec
                p = find_channel_iterative(cur_f1, f2_new, s.poses, s.posenet,
                                              tier_pair_ids, K_each=K_each, n_passes=n_passes,
                                              n_iter=80, device=s.device)
            else:
                K = int(K_spec)
                p = find_channel_only_patches(cur_f1, f2_new, s.poses, s.posenet,
                                                 tier_pair_ids, K=K, n_iter=80, device=s.device)
            cur_f1 = apply_channel_patches(cur_f1, p)
            for pi, ps in p.items():
                if pi in all_rgb_patches:
                    all_rgb_patches[pi].extend(ps)
                else:
                    all_rgb_patches[pi] = list(ps)

        # Compute bytes & score
        sb_rgb = channel_sidecar_size(all_rgb_patches)
        sb_x2 = block_mask_sidecar_size(sel_x2)
        sb_cma = mask_sidecar_size(sel_cmaes)
        sb_pat = len(serialize_pattern_mask(sel_pattern))
        sb_pose = len(serialize_pose_deltas(sel_pose))
        sb_total_bz2 = sb_x2 + sb_cma + sb_pat + sb_pose + sb_rgb

        seg_d, pose_d = fast_eval(cur_f1, f2_new, s.data["val_rgb"], s.device)
        full = compose_score(seg_d, pose_d, s.model_bytes, sb_total_bz2)

        # LZMA2 combined
        raw = build_combined_raw_v8(sel_x2, sel_cmaes, sel_pattern, sel_pose,
                                       all_rgb_patches, target_dims=target_dims)
        sb_lzma = len(lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6))
        full_lzma = compose_score(seg_d, pose_d, s.model_bytes, sb_lzma)

        rgb_results.append({
            'cfg': cfg_name, 'sb_rgb': sb_rgb, 'sb_total_bz2': sb_total_bz2,
            'sb_lzma': sb_lzma, 'score_bz2': full['score'],
            'score_lzma': full_lzma['score'], 'seg_term': full['seg_term'],
            'pose_term': full['pose_term'],
        })
        print(f"    sb_rgb={sb_rgb}B sb_total_bz2={sb_total_bz2}B sb_lzma={sb_lzma}B")
        print(f"    score_bz2={full['score']:.4f}  score_lzma={full_lzma['score']:.4f}  "
              f"(seg_term={full['seg_term']:.4f} pose_term={full['pose_term']:.4f})")

        if full_lzma['score'] < best_score:
            best_score = full_lzma['score']
            best_config = cfg_name

    # ── Final report ────────────────────────────────────────────────────
    print(f"\n=== UNIFIED PIPELINE RESULTS ===")
    print(f"{'config':<28}{'sb_rgb':>8}{'sb_lzma':>9}{'bz2_score':>11}{'lzma_score':>12}")
    for r in sorted(rgb_results, key=lambda r: r['score_lzma']):
        print(f"  {r['cfg']:<26}{r['sb_rgb']:>8}{r['sb_lzma']:>9}"
              f"{r['score_bz2']:>11.4f}{r['score_lzma']:>12.4f}")
    print(f"\nBEST: {best_config} → {best_score:.4f}")
    print(f"  vs prior best 0.2945 (e80 per-pair + LZMA): delta = {best_score - 0.2945:+.4f}")

    out_csv = OUTPUT_DIR / "v2_unified_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["cfg", "sb_rgb", "sb_total_bz2", "sb_lzma",
                     "score_bz2", "score_lzma", "seg_term", "pose_term"])
        for r in rgb_results:
            w.writerow([r['cfg'], r['sb_rgb'], r['sb_total_bz2'], r['sb_lzma'],
                        r['score_bz2'], r['score_lzma'], r['seg_term'], r['pose_term']])


if __name__ == "__main__":
    main()
