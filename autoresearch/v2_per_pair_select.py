"""Per-pair method selection.

Idea: each pair gets to pick the SUBSET of methods that best reduces its pose loss.
Methods evaluated:
  M0 = x2 (2x2 block flip from cache)
  M1 = cmaes (single-pixel from cache)
  M2 = pattern (variable-shape strip/3x3/etc — re-collected from S2)
  M3 = pose-delta (int8 dim 1/2/5 — re-collected from C3 grid)

For each pair we have between 0 and 4 candidate methods. We try all 2^k subsets.
The combo with the lowest pose loss WINS for that pair (subject to byte-cost
guard: each method costs ~5B/patch, and per-pair byte additions are tiny vs
score impact).

No metadata overhead: each method has its own bz2-compressed sidecar stream;
a pair just appears in the streams whose method was selected.

Then re-find RGB on the resulting frames + report score with and without LZMA2 wrap.
"""
import sys, os, pickle, time, struct, bz2, lzma
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import (State, batch_pose_loss_for_pattern_candidates,
                         gen_forward_with_oh_mask_batch,
                         compose_score, serialize_pattern_mask,
                         serialize_block_mask_v2)
from v2_s2_strip_cmaes import cma_es_pattern_for_pair, PATTERN_SIZES, N_PATTERNS
from v2_c3_pose_vector import (find_pose_deltas_gridsearch,
                                 serialize_pose_deltas, apply_pose_deltas_and_regen_full)
from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6)
import sidecar_explore as se
from sidecar_stack import fast_eval
from sidecar_mask_verified import (regenerate_frames_from_masks, mask_sidecar_size)
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from explore_x2_mask_blocks import block_mask_sidecar_size

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def collect_pattern_patches(s, masks_after_x2, top_n_pairs):
    """Re-run S2's variable-pattern CMA-ES to get per-pair patches."""
    out = {}
    t0 = time.time()
    for i, pi in enumerate(top_n_pairs):
        m = masks_after_x2[pi:pi+1].to(s.device).long()
        p = s.poses[pi:pi+1].to(s.device).float()
        gt_p = p.clone()
        configs = cma_es_pattern_for_pair(s.gen, m, p, gt_p, s.posenet, s.device,
                                            K=2, pop=12, gens=18)
        if configs:
            out[int(pi)] = configs
        if (i + 1) % 25 == 0:
            print(f"  ... pattern {i+1}/{len(top_n_pairs)} ({time.time()-t0:.0f}s)", flush=True)
    return out


def apply_combo_to_pair(masks_state_cpu, base_poses, pair_i, combo,
                          x2_p, cmaes_p, pattern_p, pose_p, device):
    """Build (mask_oh, pose) tensors for a pair under a method combo.
    combo: tuple of bools (use_x2, use_cmaes, use_pattern, use_pose)
    Returns (mask_oh (1, MH, MW, 5), pose (1, 6))
    """
    use_x2, use_cmaes, use_pattern, use_pose = combo
    m = masks_state_cpu[pair_i:pair_i+1].clone()
    p = base_poses[pair_i:pair_i+1].clone()
    if use_x2 and pair_i in x2_p:
        for (x, y, c) in x2_p[pair_i]:
            m[0, y:y+2, x:x+2] = c
    if use_pattern and pair_i in pattern_p:
        for (x, y, p_id, c) in pattern_p[pair_i]:
            ph, pw = PATTERN_SIZES[int(p_id)]
            yy_end = min(y + ph, MODEL_H); xx_end = min(x + pw, MODEL_W)
            m[0, y:yy_end, x:xx_end] = c
    if use_cmaes and pair_i in cmaes_p:
        for (x, y, c) in cmaes_p[pair_i]:
            m[0, y, x] = c
    if use_pose and pair_i in pose_p:
        d = pose_p[pair_i]
        scale_np = np.array([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], dtype=np.float32)
        for dim in (1, 2, 5):
            p[0, dim] = p[0, dim] + d[dim] * scale_np[dim]
    m_oh = F.one_hot(m.to(device).long(), num_classes=5).float()
    return m_oh, p.to(device).float()


def per_pair_select(s, masks_baseline_cpu, x2_p, cmaes_p, pattern_p, pose_p,
                     pair_indices, gt_poses):
    """For each pair, try every combo of available methods. Pick the lowest-loss."""
    selected = {}  # pair_i → tuple(use_x2, use_cmaes, use_pattern, use_pose)
    method_combos = []
    for use_x2 in (False, True):
        for use_cmaes in (False, True):
            for use_pattern in (False, True):
                for use_pose in (False, True):
                    method_combos.append((use_x2, use_cmaes, use_pattern, use_pose))
    print(f"  per-pair selection over {len(method_combos)} combos × {len(pair_indices)} pairs",
          flush=True)

    t0 = time.time()
    for i, pi in enumerate(pair_indices):
        # Filter combos to those where the pair has candidates for the active methods
        viable = []
        for combo in method_combos:
            ux, uc, up, upo = combo
            if ux and pi not in x2_p: continue
            if uc and pi not in cmaes_p: continue
            if up and pi not in pattern_p: continue
            if upo and pi not in pose_p: continue
            viable.append(combo)
        if not viable:
            continue

        # Batch all viable combos into a single gen forward
        m_ohs = []; ps = []
        for combo in viable:
            m_oh, p_combo = apply_combo_to_pair(masks_baseline_cpu, s.poses, pi, combo,
                                                  x2_p, cmaes_p, pattern_p, pose_p, s.device)
            m_ohs.append(m_oh); ps.append(p_combo)
        m_oh_batch = torch.cat(m_ohs, dim=0)  # (V, MH, MW, 5)
        p_batch = torch.cat(ps, dim=0)        # (V, 6)
        gt_batch = gt_poses[pi:pi+1].to(s.device).float().expand(len(viable), -1)

        with torch.no_grad():
            f1u, f2u = gen_forward_with_oh_mask_batch(s.gen, m_oh_batch, p_batch, s.device)
            pin = se.diff_posenet_input(f1u, f2u)
            fp = get_pose6(s.posenet, pin).float()
            losses = ((fp - gt_batch) ** 2).sum(dim=1).cpu().numpy()

        best_i_local = int(losses.argmin())
        selected[int(pi)] = viable[best_i_local]
        if (i + 1) % 100 == 0:
            print(f"    selection {i+1}/{len(pair_indices)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  selection done ({time.time()-t0:.0f}s)", flush=True)
    return selected


def main():
    import csv
    s = State()

    # Load cache
    print("\n=== Per-pair selector: loading cache ===", flush=True)
    if not (CACHE_DIR / "masks_x5.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder.py first.")
    masks_after_x2 = torch.load(CACHE_DIR / "masks_after_x2.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", 'rb') as f:
        x2_patches = pickle.load(f)
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", 'rb') as f:
        cmaes_patches = pickle.load(f)
    print(f"loaded: x2={len(x2_patches)} cmaes={len(cmaes_patches)}", flush=True)

    # Re-collect S2 pattern patches (top 100)
    print("\n=== Per-pair selector: collecting S2 pattern patches (top 100) ===", flush=True)
    pattern_patches = collect_pattern_patches(s, masks_after_x2,
                                                 [int(x) for x in s.rank[:100]])
    print(f"pattern: {len(pattern_patches)} pairs", flush=True)

    # Re-collect C3 pose deltas (top 200)
    print("\n=== Per-pair selector: collecting C3 pose deltas (top 200) ===", flush=True)
    masks_x5 = torch.load(CACHE_DIR / "masks_x5.pt", weights_only=False)
    pose_deltas = find_pose_deltas_gridsearch(
        s.gen, masks_x5, s.poses, s.posenet,
        [int(x) for x in s.rank[:200]], s.device,
        target_dims=(1, 2, 5))
    print(f"pose: {len(pose_deltas)} pairs", flush=True)

    # Per-pair selection over top 600 (X2 acts on top 600; others on top 100/200)
    print("\n=== Per-pair selector: choosing best subset per pair ===", flush=True)
    top_pairs = [int(x) for x in s.rank[:600]]
    selected = per_pair_select(s, s.masks_cpu, x2_patches, cmaes_patches,
                                  pattern_patches, pose_deltas, top_pairs, s.poses)

    # Method counts for reporting
    counts = {'x2':0, 'cmaes':0, 'pattern':0, 'pose':0, 'none':0}
    for combo in selected.values():
        ux, uc, up, upo = combo
        if not any(combo): counts['none'] += 1; continue
        if ux: counts['x2'] += 1
        if uc: counts['cmaes'] += 1
        if up: counts['pattern'] += 1
        if upo: counts['pose'] += 1
    print(f"  picked: x2={counts['x2']} cmaes={counts['cmaes']} "
          f"pattern={counts['pattern']} pose={counts['pose']} none={counts['none']}", flush=True)

    # Build final per-method dicts (only include each pair in the methods it picked)
    sel_x2 = {pi: x2_patches[pi] for pi, c in selected.items() if c[0] and pi in x2_patches}
    sel_cmaes = {pi: cmaes_patches[pi] for pi, c in selected.items() if c[1] and pi in cmaes_patches}
    sel_pattern = {pi: pattern_patches[pi] for pi, c in selected.items() if c[2] and pi in pattern_patches}
    sel_pose = {pi: pose_deltas[pi] for pi, c in selected.items() if c[3] and pi in pose_deltas}

    # Build final mask state (apply x2 + cmaes + pattern in order)
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

    # Apply pose deltas and regen full frames
    print("\n=== Per-pair selector: regen frames + RGB ===", flush=True)
    scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=s.device)
    if sel_pose:
        f1_new, f2_new = apply_pose_deltas_and_regen_full(
            s.gen, final_masks, s.poses, sel_pose, s.device, scale, (1, 2, 5))
    else:
        f1_new, f2_new = regenerate_frames_from_masks(s.gen, final_masks, s.poses, s.device)

    # Re-find RGB
    p_top = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                        [int(x) for x in s.rank[:250]], K=5, n_iter=80, device=s.device)
    p_tail = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                         [int(x) for x in s.rank[250:500]], K=2, n_iter=80, device=s.device)
    rgb_patches = {**p_top, **p_tail}

    # Compute sidecar bytes (per-stream bz2)
    sb_x2 = block_mask_sidecar_size(sel_x2)
    sb_cma = mask_sidecar_size(sel_cmaes)
    sb_pat = len(serialize_pattern_mask(sel_pattern))
    sb_pose = len(serialize_pose_deltas(sel_pose))
    sb_rgb = channel_sidecar_size(rgb_patches)
    sb_total = sb_x2 + sb_cma + sb_pat + sb_pose + sb_rgb

    # Score
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    seg, pose = fast_eval(f1_combined, f2_new, s.data["val_rgb"], s.device)
    full = compose_score(seg, pose, s.model_bytes, sb_total)
    delta = full['score'] - s.score_baseline

    # Also try LZMA2 wrap on the COMBINED raw stream (instead of per-stream bz2)
    # Build raw combined stream and try LZMA p6
    def build_combined_raw(x2_p, cma_p, pat_p, pose_p, rgb_p):
        parts = [b'V6\x00\x01']
        # x2 (2x2 blocks)
        parts.append(struct.pack("<I", len(x2_p)))
        for pi in sorted(x2_p.keys()):
            ps = x2_p[pi]
            parts.append(struct.pack("<HH", pi, len(ps)))
            for tup in ps:
                parts.append(struct.pack("<HHB", tup[0], tup[1], tup[2]))
        # cmaes (single-pixel)
        parts.append(struct.pack("<I", len(cma_p)))
        for pi in sorted(cma_p.keys()):
            ps = cma_p[pi]
            parts.append(struct.pack("<HH", pi, len(ps)))
            for (x, y, c) in ps:
                parts.append(struct.pack("<HHB", x, y, c))
        # pattern (variable shape)
        parts.append(struct.pack("<I", len(pat_p)))
        for pi in sorted(pat_p.keys()):
            ps = pat_p[pi]
            parts.append(struct.pack("<HH", pi, len(ps)))
            for (x, y, p_id, c) in ps:
                parts.append(struct.pack("<HHBB", x, y, p_id, c))
        # pose deltas
        parts.append(struct.pack("<I", len(pose_p)))
        for pi in sorted(pose_p.keys()):
            d = pose_p[pi]
            parts.append(struct.pack("<H", pi))
            parts.append(struct.pack("<bbb", int(d[1]), int(d[2]), int(d[5])))
        # rgb
        parts.append(struct.pack("<I", len(rgb_p)))
        for pi in sorted(rgb_p.keys()):
            ps = rgb_p[pi]
            parts.append(struct.pack("<HH", pi, len(ps)))
            for (x, y, c, d) in ps:
                parts.append(struct.pack("<HHBb", x, y, c, d))
        return b''.join(parts)

    raw = build_combined_raw(sel_x2, sel_cmaes, sel_pattern, sel_pose, rgb_patches)
    sb_lzma = len(lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6))
    full_lzma = compose_score(seg, pose, s.model_bytes, sb_lzma)
    delta_lzma = full_lzma['score'] - s.score_baseline

    print(f"\n=== Per-pair final ===")
    print(f"  per-stream bz2: sb_x2={sb_x2}B sb_cma={sb_cma}B sb_pat={sb_pat}B "
          f"sb_pose={sb_pose}B sb_rgb={sb_rgb}B total={sb_total}B")
    print(f"    score={full['score']:.4f} delta={delta:+.4f}")
    print(f"  LZMA2 combined: raw={len(raw)}B compressed={sb_lzma}B")
    print(f"    score={full_lzma['score']:.4f} delta={delta_lzma:+.4f}")

    out_csv = OUTPUT_DIR / "v2_per_pair_select_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "delta",
                     "n_x2", "n_cmaes", "n_pattern", "n_pose"])
        w.writerow(["per_pair_bz2", sb_total, full['score'], delta,
                    len(sel_x2), len(sel_cmaes), len(sel_pattern), len(sel_pose)])
        w.writerow(["per_pair_lzma", sb_lzma, full_lzma['score'], delta_lzma,
                    len(sel_x2), len(sel_cmaes), len(sel_pattern), len(sel_pose)])
    print(f"  results: {out_csv}")


if __name__ == "__main__":
    main()
