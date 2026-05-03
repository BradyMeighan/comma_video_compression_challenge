"""Add seg-targeted f2 patches on top of the e80 per-pair pipeline.

Pipeline: load cache → per-pair select (mask+pose) → regen frames → apply f1 RGB
patches → ALSO add f2 SEG patches → final eval with bz2 + LZMA2 wrap.

Tier allocation for seg patches: top 100 K=8, top 100-300 K=4 (by SEG distortion).
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
from v2_s2_strip_cmaes import cma_es_pattern_for_pair, PATTERN_SIZES
from v2_c3_pose_vector import (find_pose_deltas_gridsearch,
                                 serialize_pose_deltas, apply_pose_deltas_and_regen_full)
from v2_per_pair_select import (per_pair_select, apply_combo_to_pair,
                                  collect_pattern_patches)
from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_segnet)
import sidecar_explore as se
from sidecar_stack import fast_eval
from sidecar_mask_verified import (regenerate_frames_from_masks, mask_sidecar_size)
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from sidecar_seg_only import (find_seg_patches_f2, seg_patches_sidecar_size,
                                apply_seg_patches_f2)
from explore_x2_mask_blocks import block_mask_sidecar_size

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def main():
    import csv
    s = State()

    # ── Load cache + recollect per-pair candidates ─────────────────────
    print("\n=== seg-addon: loading cache ===", flush=True)
    if not (CACHE_DIR / "masks_x5.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder first.")
    masks_after_x2 = torch.load(CACHE_DIR / "masks_after_x2.pt", weights_only=False)
    masks_x5 = torch.load(CACHE_DIR / "masks_x5.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", 'rb') as f:
        x2_patches = pickle.load(f)
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", 'rb') as f:
        cmaes_patches = pickle.load(f)

    # Re-collect S2 patterns
    print("\n=== seg-addon: collecting S2 patterns ===", flush=True)
    pattern_patches = collect_pattern_patches(s, masks_after_x2,
                                                 [int(x) for x in s.rank[:100]])

    # Re-collect C3 pose deltas
    print("\n=== seg-addon: collecting C3 pose deltas ===", flush=True)
    pose_deltas = find_pose_deltas_gridsearch(
        s.gen, masks_x5, s.poses, s.posenet,
        [int(x) for x in s.rank[:200]], s.device, target_dims=(1, 2, 5))

    # ── Per-pair selection (mask+pose) ──────────────────────────────────
    print("\n=== seg-addon: per-pair selection ===", flush=True)
    top_pairs = [int(x) for x in s.rank[:600]]
    selected = per_pair_select(s, s.masks_cpu, x2_patches, cmaes_patches,
                                  pattern_patches, pose_deltas, top_pairs, s.poses)
    sel_x2 = {pi: x2_patches[pi] for pi, c in selected.items() if c[0] and pi in x2_patches}
    sel_cmaes = {pi: cmaes_patches[pi] for pi, c in selected.items() if c[1] and pi in cmaes_patches}
    sel_pattern = {pi: pattern_patches[pi] for pi, c in selected.items() if c[2] and pi in pattern_patches}
    sel_pose = {pi: pose_deltas[pi] for pi, c in selected.items() if c[3] and pi in pose_deltas}

    # Build final mask state + apply pose deltas + regen frames
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

    print("\n=== seg-addon: regen frames + RGB on f1 ===", flush=True)
    scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=s.device)
    if sel_pose:
        f1_new, f2_new = apply_pose_deltas_and_regen_full(
            s.gen, final_masks, s.poses, sel_pose, s.device, scale, (1, 2, 5))
    else:
        f1_new, f2_new = regenerate_frames_from_masks(s.gen, final_masks, s.poses, s.device)

    # f1 RGB patches (existing pose-targeted)
    p_top = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                        [int(x) for x in s.rank[:250]], K=5, n_iter=80, device=s.device)
    p_tail = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                         [int(x) for x in s.rank[250:500]], K=2, n_iter=80, device=s.device)
    rgb_f1_patches = {**p_top, **p_tail}
    f1_combined = apply_channel_patches(f1_new, rgb_f1_patches)

    # ── NEW: seg-targeted f2 patches ────────────────────────────────────
    print("\n=== seg-addon: f2 SEG patches ===", flush=True)
    segnet = load_segnet(s.device)

    # rank by per-pair seg distortion (recompute against current f2)
    print("  computing per-pair seg distortion ...", flush=True)
    seg_per_pair = compute_per_pair_seg(f2_new, s.data["val_rgb"], segnet, s.device)
    seg_rank = np.argsort(seg_per_pair)[::-1]
    print(f"  top-10 seg distortions: {seg_per_pair[seg_rank[:10]]}", flush=True)

    # Tier allocation: top 100 K=8, next 200 K=4
    seg_top100 = [int(x) for x in seg_rank[:100]]
    seg_tail200 = [int(x) for x in seg_rank[100:300]]

    print("  finding seg patches top 100 K=8 ...", flush=True)
    t0 = time.time()
    seg_p_top = find_seg_patches_f2(f2_new, s.data["val_rgb"][:, 1] if s.data["val_rgb"].dim() == 5 else s.data["val_rgb"],
                                       segnet, seg_top100, K=8, n_iter=80, device=s.device)
    print(f"    {len(seg_p_top)} pairs ({time.time()-t0:.0f}s)", flush=True)

    print("  finding seg patches 100-300 K=4 ...", flush=True)
    t1 = time.time()
    seg_p_tail = find_seg_patches_f2(f2_new, s.data["val_rgb"][:, 1] if s.data["val_rgb"].dim() == 5 else s.data["val_rgb"],
                                        segnet, seg_tail200, K=4, n_iter=80, device=s.device)
    print(f"    {len(seg_p_tail)} pairs ({time.time()-t1:.0f}s)", flush=True)

    seg_patches = {**seg_p_top, **seg_p_tail}

    # Apply seg patches to f2 (uint8 tensor). f2_new is (n, OUT_H, OUT_W, 3) uint8.
    f2_seg_patched = apply_seg_patches_f2(f2_new, seg_patches)

    # ── Bytes ───────────────────────────────────────────────────────────
    sb_x2 = block_mask_sidecar_size(sel_x2)
    sb_cma = mask_sidecar_size(sel_cmaes)
    sb_pat = len(serialize_pattern_mask(sel_pattern))
    sb_pose = len(serialize_pose_deltas(sel_pose))
    sb_rgb = channel_sidecar_size(rgb_f1_patches)
    sb_seg, _ = seg_patches_sidecar_size(seg_patches)
    sb_total_bz2 = sb_x2 + sb_cma + sb_pat + sb_pose + sb_rgb + sb_seg

    # ── Score (bz2 per-stream) ──────────────────────────────────────────
    seg_d, pose_d = fast_eval(f1_combined, f2_seg_patched, s.data["val_rgb"], s.device)
    full = compose_score(seg_d, pose_d, s.model_bytes, sb_total_bz2)
    delta = full['score'] - s.score_baseline

    # Also score WITHOUT seg patches (sanity / control)
    seg_d0, pose_d0 = fast_eval(f1_combined, f2_new, s.data["val_rgb"], s.device)
    sb_total_no_seg = sb_x2 + sb_cma + sb_pat + sb_pose + sb_rgb
    full_no_seg = compose_score(seg_d0, pose_d0, s.model_bytes, sb_total_no_seg)

    # ── LZMA2 combined ──────────────────────────────────────────────────
    # Build raw stream including seg patches as 6th section
    raw = build_combined_raw_v7(sel_x2, sel_cmaes, sel_pattern, sel_pose,
                                  rgb_f1_patches, seg_patches)
    sb_lzma = len(lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6))
    full_lzma = compose_score(seg_d, pose_d, s.model_bytes, sb_lzma)
    delta_lzma = full_lzma['score'] - s.score_baseline

    print(f"\n=== seg-addon final ===")
    print(f"  WITHOUT seg patches: sb={sb_total_no_seg}B score={full_no_seg['score']:.4f}")
    print(f"  per-stream bz2 + seg: sb_x2={sb_x2}B sb_cma={sb_cma}B sb_pat={sb_pat}B "
          f"sb_pose={sb_pose}B sb_rgb_f1={sb_rgb}B sb_seg_f2={sb_seg}B total={sb_total_bz2}B")
    print(f"    score={full['score']:.4f} delta={delta:+.4f} (seg_term={full['seg_term']:.4f} pose_term={full['pose_term']:.4f})")
    print(f"  LZMA2 combined: raw={len(raw)}B compressed={sb_lzma}B")
    print(f"    score={full_lzma['score']:.4f} delta={delta_lzma:+.4f}")

    out_csv = OUTPUT_DIR / "v2_seg_addon_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "delta", "seg_term", "pose_term"])
        w.writerow(["no_seg_bz2", sb_total_no_seg, full_no_seg['score'],
                    full_no_seg['score'] - s.score_baseline,
                    full_no_seg['seg_term'], full_no_seg['pose_term']])
        w.writerow(["with_seg_bz2", sb_total_bz2, full['score'], delta,
                    full['seg_term'], full['pose_term']])
        w.writerow(["with_seg_lzma", sb_lzma, full_lzma['score'], delta_lzma,
                    full_lzma['seg_term'], full_lzma['pose_term']])
    print(f"  results: {out_csv}")


def compute_per_pair_seg(f2_pred, val_rgb, segnet, device):
    """Compute per-pair seg argmax mismatch fraction for ranking."""
    n = f2_pred.shape[0]
    out = np.zeros(n, dtype=np.float32)
    bs = 8
    # val_rgb is (n, 2, OUT_H, OUT_W, 3) — pick frame 2
    if val_rgb.dim() == 5:
        f2_target = val_rgb[:, 1]
    else:
        f2_target = val_rgb
    with torch.no_grad():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            f2p = f2_pred[i:j].to(device).float().permute(0, 3, 1, 2)
            f2t = f2_target[i:j].to(device).float().permute(0, 3, 1, 2)
            from sidecar_seg_only import _segnet_logits
            log_p = _segnet_logits(f2p, segnet).argmax(dim=1)
            log_t = _segnet_logits(f2t, segnet).argmax(dim=1)
            mismatch = (log_p != log_t).float().mean(dim=(1, 2))
            out[i:j] = mismatch.cpu().numpy()
    return out


def build_combined_raw_v7(x2_p, cma_p, pat_p, pose_p, rgb_f1_p, seg_f2_p):
    """V7 raw stream — adds seg_f2 section to V6."""
    parts = [b'V7\x00\x01']
    # x2
    parts.append(struct.pack("<I", len(x2_p)))
    for pi in sorted(x2_p.keys()):
        ps = x2_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for tup in ps:
            parts.append(struct.pack("<HHB", tup[0], tup[1], tup[2]))
    # cmaes
    parts.append(struct.pack("<I", len(cma_p)))
    for pi in sorted(cma_p.keys()):
        ps = cma_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c) in ps:
            parts.append(struct.pack("<HHB", x, y, c))
    # pattern
    parts.append(struct.pack("<I", len(pat_p)))
    for pi in sorted(pat_p.keys()):
        ps = pat_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, p_id, c) in ps:
            parts.append(struct.pack("<HHBB", x, y, p_id, c))
    # pose
    parts.append(struct.pack("<I", len(pose_p)))
    for pi in sorted(pose_p.keys()):
        d = pose_p[pi]
        parts.append(struct.pack("<H", pi))
        parts.append(struct.pack("<bbb", int(d[1]), int(d[2]), int(d[5])))
    # rgb f1
    parts.append(struct.pack("<I", len(rgb_f1_p)))
    for pi in sorted(rgb_f1_p.keys()):
        ps = rgb_f1_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    # seg f2 (NEW)
    parts.append(struct.pack("<I", len(seg_f2_p)))
    for pi in sorted(seg_f2_p.keys()):
        ps = seg_f2_p[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    return b''.join(parts)


if __name__ == "__main__":
    main()
