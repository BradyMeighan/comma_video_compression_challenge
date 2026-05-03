"""S2: Variable-pattern CMA-ES (1x1, 2x2, 3x3, 1x4 horiz strip, 4x1 vert strip).

CMA-ES optimizes (x, y, pattern_id, class) for K=2 patches per pair.
Discrete pattern_id and class via floor() of continuous weights.

VECTORIZED: each generation evaluates pop=12 candidates IN A SINGLE BATCH (~10x faster).
Loads CACHED X2 base (masks_after_x2.pt) — replaces single-pixel CMA-ES.

Storage per pattern patch: u16 x, u16 y, u8 (pattern_id<<4 | class) = 5 bytes.
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import (State, batch_pose_loss_for_pattern_candidates,
                         compose_score, serialize_pattern_mask)
from prepare import MODEL_H, MODEL_W
from sidecar_stack import fast_eval
from sidecar_mask_verified import regenerate_frames_from_masks
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from explore_x2_mask_blocks import block_mask_sidecar_size

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"

N_PATTERNS = 5
PATTERN_SIZES = {0: (1,1), 1: (3,3), 2: (1,4), 3: (4,1), 4: (2,2)}


def cma_es_pattern_for_pair(gen, m_init, p, gt_p, posenet, device, K=2, pop=12, gens=18):
    """CMA-ES over (x, y, pattern_id, class) for K patches per pair.

    Genome: K × 4 floats = [x1, y1, p1, c1, x2, y2, p2, c2, ...]
    BATCHED: each generation's `pop` candidates evaluated in 1 gen forward.
    """
    dim = K * 4
    mu = np.zeros(dim)
    mu[0::4] = np.random.uniform(50, MODEL_W - 50, size=K)
    mu[1::4] = np.random.uniform(50, MODEL_H - 50, size=K)
    mu[2::4] = np.random.uniform(0, N_PATTERNS, size=K)
    mu[3::4] = np.random.uniform(0, 5, size=K)
    sigma = np.array([MODEL_W * 0.3, MODEL_H * 0.3, 1.5, 1.5] * K)

    from v2_shared import gen_forward_with_oh_mask_batch
    m_oh = F.one_hot(m_init, num_classes=5).float()
    with torch.no_grad():
        f1u, f2u = gen_forward_with_oh_mask_batch(gen, m_oh, p, device)
        from prepare import get_pose6
        import sidecar_explore as se
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        baseline_loss = ((fp - gt_p) ** 2).sum().item()

    best_loss = baseline_loss
    best_genome = None

    for _ in range(gens):
        samples = np.random.randn(pop, dim) * sigma + mu
        candidates = []
        for i in range(pop):
            configs = []
            for k in range(K):
                x = int(np.clip(samples[i, 4*k], 0, MODEL_W - 1))
                y = int(np.clip(samples[i, 4*k+1], 0, MODEL_H - 1))
                p_id = int(np.clip(samples[i, 4*k+2], 0, N_PATTERNS - 0.01))
                cls = int(np.clip(samples[i, 4*k+3], 0, 4.99))
                configs.append((x, y, p_id, cls))
            candidates.append(configs)

        losses = batch_pose_loss_for_pattern_candidates(
            gen, m_init, candidates, p, gt_p, posenet, device).cpu().numpy()

        best_i_local = losses.argmin()
        if losses[best_i_local] < best_loss:
            best_loss = losses[best_i_local]
            best_genome = candidates[best_i_local]

        sorted_idx = np.argsort(losses)
        elite = samples[sorted_idx[:pop // 2]]
        mu = elite.mean(axis=0)
        sigma = elite.std(axis=0) + 1e-6
        sigma *= 0.9

    return best_genome if best_genome else []


def main():
    import csv
    s = State()

    # Load X2 cache
    print("\n=== S2: loading X2 cache ===", flush=True)
    if not (CACHE_DIR / "masks_after_x2.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder.py first to populate {CACHE_DIR}")
    masks_after_x2 = torch.load(CACHE_DIR / "masks_after_x2.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", 'rb') as f:
        x2_patches = pickle.load(f)
    print(f"loaded: x2={len(x2_patches)}", flush=True)

    # Variable-pattern CMA-ES on top 100 (replaces O2 single-pixel CMA-ES)
    print(f"\n=== S2: variable-pattern CMA-ES K=2 on top 100 (BATCHED) ===", flush=True)
    t1 = time.time()
    pattern_patches = {}
    new_masks = masks_after_x2.clone()
    for i, pi in enumerate(s.rank[:100]):
        pi = int(pi)
        m = new_masks[pi:pi+1].to(s.device).long()
        p = s.poses[pi:pi+1].to(s.device).float()
        gt_p = p.clone()
        configs = cma_es_pattern_for_pair(s.gen, m, p, gt_p, s.posenet, s.device,
                                            K=2, pop=12, gens=18)
        if configs:
            pattern_patches[pi] = configs
            for (x, y, p_id, c) in configs:
                ph, pw = PATTERN_SIZES[p_id]
                yy_end = min(y + ph, MODEL_H); xx_end = min(x + pw, MODEL_W)
                new_masks[pi, y:yy_end, x:xx_end] = c
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/100 ({time.time()-t1:.0f}s)", flush=True)

    # Re-find RGB on new state
    print(f"\n=== S2: re-find channel-only RGB ===", flush=True)
    f1_new, f2_new = regenerate_frames_from_masks(s.gen, new_masks, s.poses, s.device)
    p_top = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                        [int(x) for x in s.rank[:250]], K=5, n_iter=80, device=s.device)
    p_tail = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                         [int(x) for x in s.rank[250:500]], K=2, n_iter=80, device=s.device)
    rgb_patches = {**p_top, **p_tail}

    sb_x2 = block_mask_sidecar_size(x2_patches)
    sb_pat = len(serialize_pattern_mask(pattern_patches))
    sb_rgb = channel_sidecar_size(rgb_patches)
    sb_total = sb_x2 + sb_pat + sb_rgb

    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    seg, pose = fast_eval(f1_combined, f2_new, s.data["val_rgb"], s.device)
    full = compose_score(seg, pose, s.model_bytes, sb_total)
    delta = full['score'] - s.score_baseline
    pat_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    for ps in pattern_patches.values():
        for (_, _, p_id, _) in ps:
            pat_counts[p_id] += 1
    print(f"\nS2 final: sb_x2={sb_x2}B sb_pat={sb_pat}B sb_rgb={sb_rgb}B "
          f"total={sb_total}B score={full['score']:.4f} delta={delta:+.4f}")
    print(f"  pattern usage: 1x1={pat_counts[0]} 3x3={pat_counts[1]} "
          f"1x4={pat_counts[2]} 4x1={pat_counts[3]} 2x2={pat_counts[4]}")
    print(f"  ({time.time()-t1:.0f}s)")

    out_csv = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_s2_strip_cmaes_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_x2", "sb_pat", "sb_rgb", "sb_total", "score", "delta",
                     "n_1x1", "n_3x3", "n_1x4", "n_4x1", "n_2x2"])
        w.writerow(["s2_x2+strip_cmaes_top100+rgb", sb_x2, sb_pat, sb_rgb, sb_total,
                    full['score'], delta, pat_counts[0], pat_counts[1],
                    pat_counts[2], pat_counts[3], pat_counts[4]])


if __name__ == "__main__":
    main()
