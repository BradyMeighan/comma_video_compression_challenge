#!/usr/bin/env python
"""
ITERATIVE mask <-> RGB alternation.

Round 1: mask K=1 top600
Apply + regenerate
Round 2: RGB on new frames
Apply RGB
Round 3: mask K=1 on RGB-patched output (compute mask grad on patched frames)
Apply mask + regenerate (now both old + new mask flips)
Round 4: RGB again on doubly-patched output
... continue while score improves

Hypothesis: each round absorbs DIFFERENT errors. Mask catches global structure,
RGB catches local pixels, mask round 2 catches new global gaps from RGB
introducing residual structure changes.
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
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_mask_verified import (verified_greedy_mask, mask_sidecar_size,
                                     regenerate_frames_from_masks,
                                     gen_forward_with_oh_mask, pose_loss_for_pair)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def verified_greedy_mask_from_state(gen, m_init, p, gt_p, posenet, device, K, n_candidates=10):
    """Same as verified_greedy_mask but starts from already-modified mask state.
    Returns (accepted_flips, final_mask)."""
    cur_m = m_init.clone()
    accepted = []
    for k_iter in range(K):
        m_oh = F.one_hot(cur_m, num_classes=5).float()
        m_oh_g = m_oh.clone().requires_grad_(True)
        f1u, f2u = gen_forward_with_oh_mask(gen, m_oh_g, p, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        baseline_loss = loss.item()
        grad = torch.autograd.grad(loss, m_oh_g)[0]
        cur_class = cur_m
        grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
        candidate_delta = grad - grad_cur.unsqueeze(-1)
        for cls in range(5):
            mask_cur = (cur_class == cls)
            candidate_delta[..., cls][mask_cur] = float('inf')
        for (x, y, _) in accepted:
            candidate_delta[0, y, x, :] = float('inf')
        best_delta, best_class = candidate_delta.min(dim=-1)
        flat = best_delta.contiguous().reshape(-1)
        topk_vals, topk_idx = torch.topk(-flat, n_candidates)

        best_actual = float('inf'); best_choice = None
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
                best_choice = (x, y, new_cls)
        if best_choice is None or best_actual >= 0:
            break
        x, y, new_cls = best_choice
        cur_m[0, y, x] = new_cls
        accepted.append((x, y, new_cls))
    return accepted, cur_m


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
    print(f"Baseline: score={base['score']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    masks_cpu = data["val_masks"].cpu()
    poses = data["val_poses"]

    csv_path = OUTPUT_DIR / "iterative_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["round", "n_mask", "K_mask", "sb_mask",
                                 "sb_rgb", "sb_total", "score", "pose_term", "delta", "elapsed"])

    BEST_KNOWN = -0.0287
    print(f"Best known: {BEST_KNOWN:+.4f}", flush=True)

    # ─── Round 1: mask K=1 top600 ───
    print("\n=== Round 1: mask K=1 top600 ===", flush=True)
    t0 = time.time()
    accumulated_mask = {}
    for i, pi in enumerate(rank[:600]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        accepted, _ = verified_greedy_mask(gen, m, p, gt_p, posenet, device, K=1, n_candidates=10)
        if accepted:
            accumulated_mask[pi] = accepted
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/600 ({time.time()-t0:.0f}s)", flush=True)
    sb_mask = mask_sidecar_size(accumulated_mask)
    print(f"  Round 1 mask: {len(accumulated_mask)} pairs, sb={sb_mask}B ({time.time()-t0:.0f}s)", flush=True)

    # Apply mask + regenerate
    new_masks = masks_cpu.clone()
    for pi, patches in accumulated_mask.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_cur, f2_cur = regenerate_frames_from_masks(gen, new_masks, poses, device)

    # ─── Round 2: RGB on new frames (winner: 250_K5+250_K2) ───
    print("\n=== Round 2: RGB 250_K5+250_K2 on regenerated frames ===", flush=True)
    t1 = time.time()
    p_top = find_pose_patches_for_pairs(
        f1_cur, f2_cur, poses, posenet,
        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_cur, f2_cur, poses, posenet,
        [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = sparse_sidecar_size(rgb_patches)
    f1_combined = apply_sparse_patches(f1_cur, rgb_patches)
    s, p = fast_eval(f1_combined, f2_cur, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"  Round 2: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
          f"delta={full['score']-base['score']:+.4f} ({time.time()-t1:.0f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([2, len(accumulated_mask), 1, sb_mask, sb_rgb,
                                 sb_mask + sb_rgb, full['score'], full['pose_term'],
                                 full['score']-base['score'], time.time()-t0])

    # ─── Round 3: mask K=1 on RGB-PATCHED output ───
    # Now find mask flips that reduce pose loss measured on f1_combined (RGB-patched)
    # Tricky: gen forward must produce frames that get RGB-patched. We need to compute
    # gradient through a chain: mask -> gen -> apply_RGB -> posenet
    # But RGB patches are non-differentiable on coords. Apply them at fixed positions/values.
    print("\n=== Round 3: mask K=1 on RGB-patched frames ===", flush=True)
    t2 = time.time()
    additional_mask = {}
    # We need to verify against state where MASK is applied + RGB is applied
    # Simplification: use the new_masks (already mask-patched), find more mask flips
    # that AFTER regeneration + RGB-patching reduce pose loss.
    # This requires custom forward + verify. Use the regenerated cur_f1 and add
    # RGB on top, then check if mask flip changes the eval.
    rgb_apply_dict = rgb_patches  # use same RGB patches

    def eval_pair_with_rgb(pi, m_modified):
        """Forward gen with given mask, apply RGB patches, eval pose."""
        m = m_modified.to(device).long()
        p = poses[pi:pi+1].to(device).float()
        with torch.no_grad():
            p1, p2 = gen(m, p)
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode='bilinear', align_corners=False).clamp(0, 255).round()
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode='bilinear', align_corners=False).clamp(0, 255).round()
            # Apply RGB
            if pi in rgb_apply_dict:
                xy, d = rgb_apply_dict[pi]
                arr = f1u[0].cpu().permute(1, 2, 0).float().numpy()
                for j in range(xy.shape[0]):
                    x, y = int(xy[j, 0]), int(xy[j, 1])
                    arr[y, x] += d[j].astype(np.float32)
                arr = np.clip(arr, 0, 255)
                f1u = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
            # PoseNet eval
            pin = se.diff_posenet_input(f1u, f2u)
            fp = get_pose6(posenet, pin).float()
            return ((fp - p) ** 2).sum().item()

    # Iterate over residual hardest pairs after current state
    # For speed, only test top 200 residual pairs
    print("  Computing residual pose error...", flush=True)
    # Get current per-pair pose error
    from sidecar_stack import per_pair_pose_mse
    res_pose = per_pair_pose_mse(f1_combined, f2_cur, poses, posenet, device)
    res_rank = np.argsort(res_pose)[::-1]
    targets = [int(x) for x in res_rank[:200]]
    print(f"  Targeting top 200 residual pairs (max residual: {res_pose[res_rank[0]]:.5f})", flush=True)

    n_added = 0
    for i, pi in enumerate(targets):
        pi = int(pi)
        m_init = new_masks[pi:pi+1].to(device).long()  # already-mask-patched
        p = poses[pi:pi+1].to(device).float()
        gt_p = p.clone()
        # Run verified greedy starting from mask-patched state, but verify against pose loss
        # of f1 = gen(new_mask).then RGB-patched.
        # The verified_greedy_mask doesn't account for RGB. So just find candidates via gradient
        # and verify the FULL pipeline (not just gen-pose).
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
            mask_cur = (cur_class == cls)
            candidate_delta[..., cls][mask_cur] = float('inf')
        # Exclude already-applied flips
        if pi in accumulated_mask:
            for (x, y, _) in accumulated_mask[pi]:
                candidate_delta[0, y, x, :] = float('inf')
        best_delta, best_class = candidate_delta.min(dim=-1)
        flat = best_delta.contiguous().reshape(-1)
        topk_vals, topk_idx = torch.topk(-flat, 5)

        # Verify with FULL pipeline
        baseline_loss = eval_pair_with_rgb(pi, m_init)
        best_actual = baseline_loss; best_choice = None
        for k in range(5):
            idx = topk_idx[k].item()
            y = idx // MODEL_W; x = idx % MODEL_W
            new_cls = best_class[0, y, x].item()
            test_m = m_init.clone()
            test_m[0, y, x] = new_cls
            new_loss = eval_pair_with_rgb(pi, test_m)
            if new_loss < best_actual:
                best_actual = new_loss
                best_choice = (x, y, new_cls)
        if best_choice is not None:
            x, y, new_cls = best_choice
            if pi in accumulated_mask:
                accumulated_mask[pi].append((x, y, new_cls))
            else:
                accumulated_mask[pi] = [(x, y, new_cls)]
            new_masks[pi, y, x] = new_cls
            n_added += 1
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/200 (added {n_added}) ({time.time()-t2:.0f}s)", flush=True)

    sb_mask3 = mask_sidecar_size(accumulated_mask)
    print(f"  Round 3: added {n_added} mask flips, sb_mask={sb_mask3}B (was {sb_mask}B)", flush=True)

    # Apply ALL accumulated mask + regen
    new_masks_final = masks_cpu.clone()
    for pi, patches in accumulated_mask.items():
        for (x, y, c) in patches:
            new_masks_final[pi, y, x] = c
    f1_new3, f2_new3 = regenerate_frames_from_masks(gen, new_masks_final, poses, device)

    # ─── Round 4: NEW RGB on doubly-patched frames ───
    print("\n=== Round 4: RGB on doubly-mask-patched frames ===", flush=True)
    t3 = time.time()
    p_top4 = find_pose_patches_for_pairs(
        f1_new3, f2_new3, poses, posenet,
        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail4 = find_pose_patches_for_pairs(
        f1_new3, f2_new3, poses, posenet,
        [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb4 = {**p_top4, **p_tail4}
    sb_rgb4 = sparse_sidecar_size(rgb4)
    f1_combined4 = apply_sparse_patches(f1_new3, rgb4)
    s, p = fast_eval(f1_combined4, f2_new3, data["val_rgb"], device)
    full4 = fast_compose(s, p, model_bytes, sb_mask3 + sb_rgb4)
    print(f"  Round 4: sb_mask={sb_mask3}B sb_rgb={sb_rgb4}B sb_total={sb_mask3+sb_rgb4}B "
          f"score={full4['score']:.4f} pose={full4['pose_term']:.4f} "
          f"delta={full4['score']-base['score']:+.4f} d_vs_best={full4['score']-base['score']-BEST_KNOWN:+.4f} "
          f"({time.time()-t3:.0f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow([4, len(accumulated_mask),
                                 sum(len(v) for v in accumulated_mask.values()),
                                 sb_mask3, sb_rgb4, sb_mask3 + sb_rgb4,
                                 full4['score'], full4['pose_term'],
                                 full4['score']-base['score'], time.time()-t0])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
