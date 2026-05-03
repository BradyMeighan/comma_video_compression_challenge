#!/usr/bin/env python
"""
Verified greedy mask sidecar:
  For each pair, find top-N candidate flips via gradient, then VERIFY each
  by actually flipping and recomputing pose. Keep only flips that reduce loss.
  Iterative: apply best, then re-search for next best.

Test K=1, 3, 5 on hardest 100 pairs.
"""
import sys, os, time, csv, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6, load_posenet,
                      pack_pair_yuv6, estimate_model_bytes)
from train import Generator, load_data_full, coords
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs, per_pair_pose_mse)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def gen_forward_with_oh_mask(gen, m_oh, p, device):
    co = coords(m_oh.shape[0], MODEL_H, MODEL_W, device)
    e_emb = m_oh @ gen.trunk.emb.weight
    e = F.interpolate(e_emb.permute(0, 3, 1, 2), co.shape[-2:], mode='bilinear', align_corners=False)
    s = gen.trunk.s1(gen.trunk.stem(torch.cat([e, co], 1)))
    z = gen.trunk.up(gen.trunk.d1(gen.trunk.down(s)))
    feat = gen.trunk.f1(gen.trunk.fuse(torch.cat([z, s], 1)))
    cond = gen.pose_mlp(p)
    f1u = F.interpolate(gen.h1(feat, cond), (OUT_H, OUT_W), mode='bilinear', align_corners=False)
    f2u = F.interpolate(gen.h2(feat), (OUT_H, OUT_W), mode='bilinear', align_corners=False)
    return f1u, f2u


def pose_loss_for_pair(gen, m_oh, p, gt_p, posenet, device):
    f1u, f2u = gen_forward_with_oh_mask(gen, m_oh, p, device)
    pin = se.diff_posenet_input(f1u, f2u)
    fp = get_pose6(posenet, pin).float()
    return ((fp - gt_p) ** 2).sum().item()


def verified_greedy_mask(gen, m, p, gt_p, posenet, device, K, n_candidates=20):
    """Iterative greedy: K times, find top-N candidates, verify, pick best."""
    cur_m = m.clone()
    accepted = []  # list of (x, y, new_class)

    for k_iter in range(K):
        # Compute gradient at current state
        m_oh = F.one_hot(cur_m, num_classes=5).float()
        m_oh_g = m_oh.clone().requires_grad_(True)
        f1u, f2u = gen_forward_with_oh_mask(gen, m_oh_g, p, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        baseline_loss = loss.item()
        grad = torch.autograd.grad(loss, m_oh_g)[0]

        # Score candidates
        cur_class = cur_m
        grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
        candidate_delta = grad - grad_cur.unsqueeze(-1)
        for cls in range(5):
            mask_cur = (cur_class == cls)
            candidate_delta[..., cls][mask_cur] = float('inf')
        # Also exclude already-accepted positions
        for (x, y, _) in accepted:
            candidate_delta[0, y, x, :] = float('inf')
        best_delta, best_class = candidate_delta.min(dim=-1)

        # Top-N candidates by predicted delta
        flat = best_delta.contiguous().reshape(-1)
        topk_vals, topk_idx = torch.topk(-flat, n_candidates)

        # Verify each
        best_actual = float('inf')
        best_choice = None
        for k in range(n_candidates):
            idx = topk_idx[k].item()
            y = idx // MODEL_W; x = idx % MODEL_W
            new_cls = best_class[0, y, x].item()

            test_m = cur_m.clone()
            test_m[0, y, x] = new_cls
            test_oh = F.one_hot(test_m, num_classes=5).float()
            with torch.no_grad():
                new_loss = pose_loss_for_pair(gen, test_oh, p, gt_p, posenet, device)
            actual_delta = new_loss - baseline_loss
            if actual_delta < best_actual:
                best_actual = actual_delta
                best_choice = (x, y, new_cls, new_loss)

        if best_choice is None or best_actual >= 0:
            break  # no improvement possible
        x, y, new_cls, new_loss = best_choice
        cur_m[0, y, x] = new_cls
        accepted.append((x, y, new_cls))

    return accepted, cur_m


def mask_sidecar_size(mask_patches):
    if not mask_patches:
        return 0
    parts = [struct.pack("<H", len(mask_patches))]
    for pair_i, patches in sorted(mask_patches.items()):
        parts.append(struct.pack("<HH", pair_i, len(patches)))
        for (x, y, c) in patches:
            parts.append(struct.pack("<HHB", x, y, c))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def regenerate_frames_from_masks(gen, masks, poses, device, batch_size=8):
    n = masks.shape[0]
    f1_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    f2_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    gen.eval()
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            m = masks[i:i+batch_size].to(device).long()
            p = poses[i:i+batch_size].to(device).float()
            p1, p2 = gen(m, p)
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            f1_all[i:i+batch_size] = f1u.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
            f2_all[i:i+batch_size] = f2u.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
    return f1_all, f2_all


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
    seg, pose = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg, pose, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    csv_path = OUTPUT_DIR / "mask_verified_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_pairs", "K_max", "n_accepted",
                                 "sidecar_bytes", "score", "pose_term", "delta", "elapsed"])

    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    for K_max, n_top in [(1, 100), (3, 100), (1, 200), (3, 200)]:
        spec = f"verified_K{K_max}_top{n_top}"
        print(f"\n=== {spec} ===", flush=True)
        t0 = time.time()
        mask_patches = {}
        n_accepted = 0
        for pi in rank[:n_top]:
            pi = int(pi)
            m = masks_cpu[pi:pi+1].to(device).long()
            p = poses[pi:pi+1].to(device).float()
            gt_p = p.clone()
            accepted, _ = verified_greedy_mask(
                gen, m, p, gt_p, posenet, device, K=K_max, n_candidates=15)
            if accepted:
                mask_patches[pi] = accepted
                n_accepted += len(accepted)
        elapsed = time.time() - t0

        # Apply
        new_masks = masks_cpu.clone()
        for pi, patches in mask_patches.items():
            for (x, y, c) in patches:
                new_masks[pi, y, x] = c

        # Regenerate
        f1_p, f2_p = regenerate_frames_from_masks(gen, new_masks, poses, device)
        sb = mask_sidecar_size(mask_patches)
        seg, pose = fast_eval(f1_p, f2_p, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb)
        delta = full['score'] - base['score']
        print(f"  >> {spec}: pairs_with_patches={len(mask_patches)} K_total={n_accepted} "
              f"sb={sb}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} elapsed={elapsed:.1f}s", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, n_top, K_max, n_accepted, sb,
                                     full['score'], full['pose_term'], delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
