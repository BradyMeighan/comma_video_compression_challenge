"""C3: Pose-vector input sidecar with per-dim targeting.

Modify the input pose vector before generator (after X5 base applied).
Per-dim targeting: only optimize dims 1, 2, 5 (dominant residuals after RGB).
Quantize to int8 with per-dim scale factors.

VECTORIZED: process pairs in batches of 16; reuses CACHED X5 masks.
"""
import sys, os, pickle, time, struct, bz2, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import State, compose_score
from prepare import (OUT_H, OUT_W, get_pose6)
import sidecar_explore as se
from sidecar_stack import fast_eval
from sidecar_mask_verified import (regenerate_frames_from_masks, mask_sidecar_size)
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from explore_x2_mask_blocks import block_mask_sidecar_size

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"


def find_pose_deltas_gridsearch(gen, masks_state_cpu, base_poses, posenet, pair_indices,
                                  device, scale=None, target_dims=(1, 2, 5),
                                  delta_range=(-3, -2, -1, 0, 1, 2, 3)):
    """Gradient-FREE grid search over int8 pose deltas in target dims.

    For each pair, evaluate every (delta_dim1, delta_dim2, delta_dim5) tuple in 1
    batched gen forward, pick the lowest-loss combo. Avoids slow autograd through
    the FP4-quantized generator.

    7^3 = 343 candidates per pair × n_pairs. Batched per pair (343 in 1 forward).
    """
    from v2_shared import gen_forward_with_oh_mask_batch
    if scale is None:
        scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=device)
    target_mask = torch.zeros(6, device=device)
    for d in target_dims:
        target_mask[d] = 1.0

    # Build the candidate-delta grid: (N, 6) where only target_dims are non-zero
    n_per_dim = len(delta_range)
    n_dims = len(target_dims)
    grid_shape = tuple([n_per_dim] * n_dims)
    deltas_per_dim = torch.tensor(delta_range, dtype=torch.float32, device=device)
    grids = torch.meshgrid(*([deltas_per_dim] * n_dims), indexing='ij')
    grid_flat = torch.stack([g.reshape(-1) for g in grids], dim=1)  # (N, n_dims)
    N = grid_flat.shape[0]
    delta_grid = torch.zeros((N, 6), device=device)
    for j, d in enumerate(target_dims):
        delta_grid[:, d] = grid_flat[:, j]

    out = {}
    gen.eval()
    for pi in pair_indices:
        m = masks_state_cpu[pi:pi+1].to(device).long()
        p_base = base_poses[pi:pi+1].to(device).float()
        gt_p = p_base.clone()

        # Build N pose perturbations for this pair (chunk to avoid INT32 overflow on big feats)
        p_perturbed_all = p_base + (delta_grid * scale)  # (N, 6)
        m_oh = F.one_hot(m, num_classes=5).float()
        gt_batch_all = gt_p.expand(N, -1)
        chunk = 32
        losses = torch.zeros(N, device=device)
        with torch.no_grad():
            for cs in range(0, N, chunk):
                ce = min(cs + chunk, N)
                m_oh_batch = m_oh.expand(ce - cs, -1, -1, -1).contiguous()
                p_chunk = p_perturbed_all[cs:ce]
                f1u, f2u = gen_forward_with_oh_mask_batch(gen, m_oh_batch, p_chunk, device)
                pin = se.diff_posenet_input(f1u, f2u)
                fp = get_pose6(posenet, pin).float()
                losses[cs:ce] = ((fp - gt_batch_all[cs:ce]) ** 2).sum(dim=1)

        best_i = int(losses.argmin().item())
        # Compare to baseline (zero delta — index where all dims are 0)
        zero_idx_mask = (grid_flat == 0).all(dim=1)
        zero_idx = int(zero_idx_mask.nonzero(as_tuple=True)[0].item()) if zero_idx_mask.any() else -1
        if zero_idx >= 0 and losses[best_i] >= losses[zero_idx] - 1e-9:
            continue  # no improvement
        chosen_delta = grid_flat[best_i]  # (n_dims,)
        d_int = np.zeros(6, dtype=np.int8)
        for j, d in enumerate(target_dims):
            d_int[d] = int(chosen_delta[j].item())
        if (d_int != 0).any():
            out[int(pi)] = d_int
    return out


# Keep the old name as an alias for backwards compat
find_pose_deltas_batched = find_pose_deltas_gridsearch


def serialize_pose_deltas(pose_deltas, target_dims=(1, 2, 5)):
    """u16 num_pairs, then for each: u16 idx, K × i8 (only target dims). bz2."""
    if not pose_deltas:
        return b''
    parts = [struct.pack("<H", len(pose_deltas))]
    n_dims = len(target_dims)
    fmt = "<H" + "b" * n_dims
    for pi in sorted(pose_deltas.keys()):
        d = pose_deltas[pi]
        vals = [int(d[dim]) for dim in target_dims]
        parts.append(struct.pack(fmt, pi, *vals))
    return bz2.compress(b''.join(parts), compresslevel=9)


def apply_pose_deltas_and_regen_full(gen, masks_cpu, poses, pose_deltas, device,
                                        scale, target_dims, batch_size=8):
    n = poses.shape[0]
    f1_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    f2_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    poses_mod = poses.clone()
    for pi, d in pose_deltas.items():
        for dim in target_dims:
            poses_mod[pi, dim] = poses[pi, dim] + d[dim].item() * scale[dim].item()
    gen.eval()
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            m = masks_cpu[i:i+batch_size].to(device).long()
            p = poses_mod[i:i+batch_size].to(device).float()
            p1, p2 = gen(m, p)
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            f1_all[i:i+batch_size] = f1u.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
            f2_all[i:i+batch_size] = f2u.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
    return f1_all, f2_all


def main():
    import csv
    s = State()
    target_dims = (1, 2, 5)
    scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=s.device)

    # Load X5 cache (X2 + CMA-ES applied to masks)
    print("\n=== C3: loading X5 cache ===", flush=True)
    if not (CACHE_DIR / "masks_x5.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder.py first to populate {CACHE_DIR}")
    masks_x5 = torch.load(CACHE_DIR / "masks_x5.pt", weights_only=False)
    with open(CACHE_DIR / "x2_patches.pkl", 'rb') as f:
        x2_patches = pickle.load(f)
    with open(CACHE_DIR / "cmaes_top100_patches.pkl", 'rb') as f:
        cmaes_patches = pickle.load(f)
    print(f"loaded: x2={len(x2_patches)} cmaes={len(cmaes_patches)}", flush=True)

    # Pose-vector deltas (top 200 hardest) — gradient-free grid search
    print(f"\n=== C3: pose-vector deltas top 200 (dims {target_dims}, GRID 7^3=343/pair) ===", flush=True)
    t0 = time.time()
    top_n = [int(x) for x in s.rank[:200]]
    # Process in chunks for progress visibility
    pose_deltas = {}
    chunk = 50
    for i in range(0, len(top_n), chunk):
        sub = top_n[i:i+chunk]
        sub_out = find_pose_deltas_gridsearch(
            s.gen, masks_x5, s.poses, s.posenet, sub, s.device,
            scale=scale, target_dims=target_dims)
        pose_deltas.update(sub_out)
        print(f"  ... pose-deltas {min(i+chunk, len(top_n))}/{len(top_n)} "
              f"(found={len(pose_deltas)}, {time.time()-t0:.0f}s)", flush=True)
    print(f"Pose deltas: {len(pose_deltas)} pairs improved ({time.time()-t0:.0f}s)", flush=True)

    # Apply pose deltas + mask + regen full frame set
    f1_new, f2_new = apply_pose_deltas_and_regen_full(
        s.gen, masks_x5, s.poses, pose_deltas, s.device, scale, target_dims)

    # Re-find RGB on doubly-modified state
    print(f"\n=== C3: re-find channel-only RGB ===", flush=True)
    p_top = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                        [int(x) for x in s.rank[:250]], K=5, n_iter=80, device=s.device)
    p_tail = find_channel_only_patches(f1_new, f2_new, s.poses, s.posenet,
                                         [int(x) for x in s.rank[250:500]], K=2, n_iter=80, device=s.device)
    rgb_patches = {**p_top, **p_tail}

    sb_x2 = block_mask_sidecar_size(x2_patches)
    sb_cma = mask_sidecar_size(cmaes_patches)
    sb_pose = len(serialize_pose_deltas(pose_deltas, target_dims))
    sb_rgb = channel_sidecar_size(rgb_patches)
    sb_total = sb_x2 + sb_cma + sb_pose + sb_rgb

    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    seg, pose = fast_eval(f1_combined, f2_new, s.data["val_rgb"], s.device)
    full = compose_score(seg, pose, s.model_bytes, sb_total)
    delta = full['score'] - s.score_baseline
    print(f"\nC3 final: sb_x2={sb_x2}B sb_cma={sb_cma}B sb_pose={sb_pose}B sb_rgb={sb_rgb}B "
          f"total={sb_total}B score={full['score']:.4f} delta={delta:+.4f} ({time.time()-t0:.0f}s)")

    out_csv = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_c3_pose_vector_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_x2", "sb_cma", "sb_pose", "sb_rgb", "sb_total", "score", "delta"])
        w.writerow(["c3_x5+pose_dims125_top200", sb_x2, sb_cma, sb_pose, sb_rgb,
                    sb_total, full['score'], delta])


if __name__ == "__main__":
    main()
