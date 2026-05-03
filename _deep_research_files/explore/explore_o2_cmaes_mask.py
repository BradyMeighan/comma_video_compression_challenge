#!/usr/bin/env python
"""
O2: CMA-ES gradient-free optimization for mask flips.

The verified-greedy mask sidecar has 25% gradient direction accuracy. CMA-ES
samples populations of candidate flip CONFIGURATIONS and evaluates them
holistically — could find combinations the greedy misses.

For tractability:
- Restrict to top-50 candidate positions (from gradient) per pair
- CMA-ES optimizes which K=2 of these to flip + which class to flip to
- Population size 16, generations 30 → 480 evals per pair
- For top 100 hardest pairs only (compute budget)
"""
import sys, os, pickle, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      estimate_model_bytes)
from train import Generator, load_data_full, coords
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_mask_verified import (mask_sidecar_size, regenerate_frames_from_masks,
                                     gen_forward_with_oh_mask, pose_loss_for_pair)
from sidecar_channel_only import find_channel_only_patches, channel_sidecar_size, apply_channel_patches

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def cma_es_mask_for_pair(gen, m_init, p, gt_p, posenet, device, K=2, pop=12, gens=20, n_candidates=30):
    """For a single pair, CMA-ES over K mask flips."""
    # Get top-N candidate positions from gradient
    m_oh = F.one_hot(m_init, num_classes=5).float().requires_grad_(True)
    f1u, f2u = gen_forward_with_oh_mask(gen, m_oh, p, device)
    pin = se.diff_posenet_input(f1u, f2u)
    fp = get_pose6(posenet, pin).float()
    loss = ((fp - gt_p) ** 2).sum()
    grad = torch.autograd.grad(loss, m_oh)[0]
    cur_class = m_init
    grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
    candidate_delta = grad - grad_cur.unsqueeze(-1)
    for cls in range(5):
        candidate_delta[..., cls][cur_class == cls] = float('inf')
    best_delta, best_class = candidate_delta.min(dim=-1)
    flat = best_delta.contiguous().reshape(-1)
    _, top_idx = torch.topk(-flat, n_candidates)
    cand_ys = (top_idx // MODEL_W).long().cpu().numpy()
    cand_xs = (top_idx % MODEL_W).long().cpu().numpy()
    cand_classes = best_class.contiguous().reshape(-1)[top_idx].cpu().numpy()

    # Baseline loss
    with torch.no_grad():
        base_loss = pose_loss_for_pair(gen, F.one_hot(m_init, num_classes=5).float(), p, gt_p, posenet, device)

    # CMA-ES over [0, n_candidates)^K (which K candidates to use)
    # Use simple Gaussian-perturbation ES (lighter than full CMA-ES, no extra deps)
    mu = np.random.uniform(0, n_candidates, K)
    sigma = n_candidates * 0.3
    best_choice = None
    best_loss = base_loss

    for gen_iter in range(gens):
        # Sample population
        samples = []
        losses = []
        for _ in range(pop):
            x = mu + sigma * np.random.randn(K)
            x = np.clip(x.round().astype(int), 0, n_candidates - 1)
            x = np.unique(x)  # dedupe
            if len(x) == 0:
                continue
            # Apply these flips
            m_test = m_init.clone()
            flips = []
            for idx in x:
                yy = int(cand_ys[idx]); xx = int(cand_xs[idx])
                cc = int(cand_classes[idx])
                m_test[0, yy, xx] = cc
                flips.append((xx, yy, cc))
            with torch.no_grad():
                test_oh = F.one_hot(m_test, num_classes=5).float()
                new_loss = pose_loss_for_pair(gen, test_oh, p, gt_p, posenet, device)
            samples.append((x, flips, new_loss))
            losses.append(new_loss)
            if new_loss < best_loss:
                best_loss = new_loss
                best_choice = flips

        # Evolutionary update: keep top half, recompute mean/sigma
        if not samples:
            break
        samples.sort(key=lambda s: s[2])
        elite = samples[:max(1, pop // 2)]
        mu_x = np.zeros(K)
        cnt = 0
        for x, _, _ in elite:
            for v in x[:K]:
                mu_x[cnt % K] += v
                cnt += 1
        mu = mu_x / max(1, len(elite))
        sigma *= 0.85  # shrink

    return best_choice if best_choice else []


def main():
    device = torch.device("cuda")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
    new_masks = bf['new_masks']
    f1_new_bl, f2_new_bl = bf['f1_new'], bf['f2_new']
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    score_bl = bp['score']
    sb_mask_bl = bp['sb_mask_bz2']

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}, sb_mask={sb_mask_bl}B")

    # Run CMA-ES on top 100 hardest pairs
    print("\n=== O2: CMA-ES mask flips on top 100 ===")
    t0 = time.time()
    new_mask_patches = dict(bp['mask_patches'])  # start from baseline mask sidecar
    n_added = 0
    for i, pi in enumerate(rank[:100]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        # Apply existing mask flips first
        if pi in new_mask_patches:
            for (x, y, c) in new_mask_patches[pi]:
                m[0, y, x] = c
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        # CMA-ES for K=2 additional flips
        flips = cma_es_mask_for_pair(gen, m, p, gt_p, posenet, device, K=2, pop=10, gens=15)
        if flips:
            if pi in new_mask_patches:
                new_mask_patches[pi].extend(flips)
            else:
                new_mask_patches[pi] = flips
            n_added += len(flips)
        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/100 (added {n_added}) ({time.time()-t0:.0f}s)", flush=True)

    sb_mask_new = mask_sidecar_size(new_mask_patches)
    print(f"Mask: original {len(bp['mask_patches'])} pairs / {sb_mask_bl}B → "
          f"new {len(new_mask_patches)} pairs / {sb_mask_new}B (+{n_added} flips)")

    # Apply new mask + regenerate
    new_masks_full = masks_cpu.clone()
    for pi, patches in new_mask_patches.items():
        for (x, y, c) in patches:
            new_masks_full[pi, y, x] = c
    f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks_full, poses, device)

    # Re-find RGB on new frames
    print("\nRe-finding RGB on new frames (250_K5+250_K2)...")
    p_top = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_channel_only_patches(f1_new, f2_new, poses, posenet,
                                         [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = channel_sidecar_size(rgb_patches)
    f1_combined = apply_channel_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask_new + sb_rgb)
    print(f"O2: sb_mask={sb_mask_new}B sb_rgb={sb_rgb}B sb_total={sb_mask_new+sb_rgb}B "
          f"score={full['score']:.4f} delta={full['score']-score_bl:+.4f} ({time.time()-t0:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "o2_cmaes_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "n_added_flips", "sb_total", "score", "delta"])
        w.writerow(["o2_cmaes_top100_K2", n_added, sb_mask_new + sb_rgb, full['score'],
                    full['score']-score_bl])


if __name__ == "__main__":
    main()
