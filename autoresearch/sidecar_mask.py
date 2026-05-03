#!/usr/bin/env python
"""
MASK SIDECAR — patch the input mask, not the output frames.

Why: The mask drives the generator's Trunk feature map (via embedding +
DSConvs). One mask pixel flip propagates through the receptive field to
change MANY output pixels. So 1 mask flip ≈ many RGB pixel patches.

Encoding: 5 bytes per patch (u16 x, u16 y, u8 new_class).

Method per pair:
  1. Compute baseline forward via gen(mask, pose) → frames → eval pose loss.
  2. Compute gradient of loss w.r.t. one-hot mask via straight-through.
  3. Score each pixel by (loss reduction if flipped to best alternative class).
  4. Pick top-K pixels and write down their (x, y, best_class).
  5. Apply patches: modify mask at those positions; re-run generator.

Tests:
  A. Mask-only sidecar (no RGB patches) at K=10, 20, 50
  B. Mask sidecar STACKED on top of best RGB sidecar (350_K7+250_K2)
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
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs, per_pair_pose_mse)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_mask_patches_for_pairs(gen, masks, poses, gt_poses_for_loss, posenet,
                                  pair_indices, K, device):
    """For each pair, find K mask flips that reduce pose loss most.

    masks: (n, MH, MW) long tensor on CPU.
    Returns dict pair_i -> list of (x, y, new_class) tuples.
    """
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        m = masks[sel].to(device).long()  # (b, MH, MW)
        p = poses[sel].to(device).float()  # (b, 6)
        gt_p = gt_poses_for_loss[sel].to(device).float()

        # Build a continuous one-hot mask tensor and route through gen
        m_oh = F.one_hot(m, num_classes=5).float()  # (b, MH, MW, 5)
        m_oh = m_oh.requires_grad_(True)

        # We need to call gen but it expects integer mask for the embedding.
        # Trick: use the embedding weights directly:
        # gen.trunk.emb is a QEmb(5, EMB_DIM). Its weight is (5, EMB_DIM).
        # The line in trunk.forward is: e = F.interpolate(self.emb(mask.long()).permute(0,3,1,2), ...)
        # We can replace this with: e_emb = m_oh @ emb_weight → (b, MH, MW, EMB_DIM) → interpolate
        # But we'd need to monkey-patch the trunk. Easier:
        # Build a custom forward that replaces self.emb(mask.long()) with one_hot @ embedding.weight

        emb_weight = gen.trunk.emb.weight  # (5, EMB_DIM)
        # Replicate trunk forward
        with torch.enable_grad():
            from prepare import MODEL_H, MODEL_W
            from train import coords
            co = coords(b, MODEL_H, MODEL_W, device)
            # Original: self.emb(mask.long()) - shape (b, MH, MW, EMB_DIM)
            e_emb = m_oh @ emb_weight  # (b, MH, MW, EMB_DIM)
            e = F.interpolate(e_emb.permute(0, 3, 1, 2), co.shape[-2:], mode='bilinear', align_corners=False)
            s = gen.trunk.s1(gen.trunk.stem(torch.cat([e, co], 1)))
            z = gen.trunk.up(gen.trunk.d1(gen.trunk.down(s)))
            feat = gen.trunk.f1(gen.trunk.fuse(torch.cat([z, s], 1)))
            cond = gen.pose_mlp(p)
            f1_out = gen.h1(feat, cond)
            f2_out = gen.h2(feat)
            f1u = F.interpolate(f1_out, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            f2u = F.interpolate(f2_out, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            pin = se.diff_posenet_input(f1u, f2u)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()

        # Gradient w.r.t. one-hot mask
        grad = torch.autograd.grad(loss, m_oh)[0]  # (b, MH, MW, 5)

        # For each pixel, score each alternative class by linear approx:
        # delta_loss ≈ grad . delta_oh, where delta_oh = (target_oh - current_oh)
        # We want delta_loss to be NEGATIVE (reduce loss).
        # Best alternative class for a pixel: argmin over c of grad[c] - grad[current_c]
        cur_class = m  # (b, MH, MW)
        # Gather grad of current class: grad_at_cur[b, h, w] = grad[b, h, w, cur_class[b, h, w]]
        grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)  # (b, MH, MW)
        # delta for each candidate class: grad[b, h, w, c] - grad_cur[b, h, w]
        candidate_delta = grad - grad_cur.unsqueeze(-1)  # (b, MH, MW, 5)
        # Mask out current class (delta would be 0)
        for cls in range(5):
            mask_cur = (cur_class == cls)
            candidate_delta[..., cls][mask_cur] = float('inf')  # don't pick same class
        # Best (most negative delta) per pixel
        best_delta, best_class = candidate_delta.min(dim=-1)  # (b, MH, MW)

        # Top-K pixels with most negative delta (biggest expected loss reduction)
        flat = best_delta.contiguous().reshape(b, -1)
        # We want most negative — use topk on -flat
        _, topk = torch.topk(-flat, K, dim=1)
        ys_t = (topk // MODEL_W).long()
        xs_t = (topk % MODEL_W).long()
        cls_t = torch.gather(best_class.contiguous().reshape(b, -1), 1, topk)  # (b, K)

        xs_np = xs_t.cpu().numpy().astype(np.uint16)
        ys_np = ys_t.cpu().numpy().astype(np.uint16)
        cls_np = cls_t.cpu().numpy().astype(np.uint8)
        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(), cls_np[bi].tolist()))
            out[pair_i] = patches
    return out


def apply_mask_patches(masks_orig, mask_patches):
    """Apply mask patches: dict pair_i -> [(x, y, new_class), ...]. Returns new mask tensor."""
    out = masks_orig.clone()
    for pair_i, patches in mask_patches.items():
        m = out[pair_i]
        for (x, y, c) in patches:
            m[y, x] = c
    return out


def mask_sidecar_size(mask_patches):
    """5 bytes per patch (u16 x, u16 y, u8 class). Plus per-pair header (u16 idx, u16 K)."""
    if not mask_patches:
        return 0
    parts = [struct.pack("<H", len(mask_patches))]
    for pair_i, patches in sorted(mask_patches.items()):
        parts.append(struct.pack("<HH", pair_i, len(patches)))
        for (x, y, c) in patches:
            parts.append(struct.pack("<HHB", x, y, c))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def regenerate_frames_from_masks(gen, new_masks, poses, device, batch_size=8):
    """Re-run generator with new masks to get patched frames at OUT resolution."""
    n = new_masks.shape[0]
    f1_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    f2_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    gen.eval()
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            m = new_masks[i:i+batch_size].to(device).long()
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

    masks = data["val_masks"]  # (n, MH, MW) on device
    poses = data["val_poses"]
    val_poses = data["val_poses"]

    # Baseline (no patches)
    f1_all, f2_all = se.generate_all_frames(gen, data, device)
    seg, pose = fast_eval(f1_all, f2_all, data["val_rgb"], device)
    base = fast_compose(seg, pose, model_bytes, 0)
    print(f"Baseline: score={base['score']:.4f} pose={base['pose_term']:.4f}", flush=True)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    csv_path = OUTPUT_DIR / "mask_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "pairs", "K_total", "sidecar_bytes",
                                 "score", "seg_term", "pose_term", "delta", "elapsed"])

    # ─── Test A: mask-only sidecar (no RGB patches) ───
    print("\n=== A: mask-only sidecar tests ===", flush=True)
    masks_cpu = masks.cpu()
    for K, n_top in [(5, 600), (10, 600), (20, 600), (5, 300), (10, 300), (20, 300)]:
        spec = f"mask_K{K}_top{n_top}"
        print(f"\n  Searching: {spec}", flush=True)
        t0 = time.time()
        pair_idxs = [int(x) for x in rank[:n_top]]
        mask_patches = find_mask_patches_for_pairs(
            gen, masks_cpu, poses, val_poses, posenet, pair_idxs, K=K, device=device)
        # Apply
        new_masks = apply_mask_patches(masks_cpu, mask_patches)
        # Regenerate frames from new masks
        f1_p, f2_p = regenerate_frames_from_masks(gen, new_masks, poses, device)
        elapsed = time.time() - t0
        sb = mask_sidecar_size(mask_patches)
        seg, pose = fast_eval(f1_p, f2_p, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb)
        delta = full['score'] - base['score']
        K_total = sum(len(v) for v in mask_patches.values())
        print(f"  >> {spec}: K_total={K_total} sb={sb}B "
              f"score={full['score']:.4f} seg={full['seg_term']:.4f} "
              f"pose={full['pose_term']:.4f} delta={delta:+.4f} elapsed={elapsed:.1f}s",
              flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, n_top, K_total, sb,
                                     full['score'], full['seg_term'], full['pose_term'],
                                     delta, elapsed])

    # ─── Test B: mask sidecar STACKED on top of best RGB sidecar ───
    print("\n=== B: mask + RGB stacked ===", flush=True)
    print("  Building best RGB base (350_K7+250_K2)...", flush=True)
    p_top = find_pose_patches_for_pairs(
        f1_all, f2_all, val_poses, posenet,
        [int(x) for x in rank[:350]], K=7, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_all, f2_all, val_poses, posenet,
        [int(x) for x in rank[350:600]], K=2, n_iter=80, device=device)
    best_rgb_patches = {**p_top, **p_tail}
    sb_rgb = sparse_sidecar_size(best_rgb_patches)
    f1_rgb = apply_sparse_patches(f1_all, best_rgb_patches)
    seg_rgb, pose_rgb = fast_eval(f1_rgb, f2_all, data["val_rgb"], device)
    base_rgb = fast_compose(seg_rgb, pose_rgb, model_bytes, sb_rgb)
    print(f"  RGB base: score={base_rgb['score']:.4f} sb={sb_rgb}B (delta {base_rgb['score']-base['score']:+.4f})",
          flush=True)

    # Now find mask patches w.r.t. the RGB-patched output
    # NOTE: this is tricky — applying both at the same time means the generator
    # output (modified by mask) gets RGB patches on top. We need to find mask
    # patches that, AFTER the RGB patches are applied, still reduce pose loss.
    # Simplified: apply mask patches → regenerate → apply RGB patches → eval.

    # Identify residual hardest pairs after RGB patches
    res_pose = per_pair_pose_mse(f1_rgb, f2_all, val_poses, posenet, device)
    res_rank = np.argsort(res_pose)[::-1]
    res_top = [int(x) for x in res_rank[:200]]
    print(f"  Searching mask patches on residual top 200...", flush=True)

    for K in [5, 10, 20]:
        spec = f"mask_K{K}_res200_+rgb"
        t0 = time.time()
        # Find mask patches based on baseline frames (we use the original gen forward)
        mask_patches = find_mask_patches_for_pairs(
            gen, masks_cpu, poses, val_poses, posenet, res_top, K=K, device=device)
        # Apply mask patches
        new_masks = apply_mask_patches(masks_cpu, mask_patches)
        # Regenerate frames from new masks
        f1_new, f2_new = regenerate_frames_from_masks(gen, new_masks, poses, device)
        # Apply the SAME RGB patches on top (but regenerated f1 differs from f1_all)
        f1_combined = apply_sparse_patches(f1_new, best_rgb_patches)
        elapsed = time.time() - t0

        sb_mask = mask_sidecar_size(mask_patches)
        sb_total = sb_rgb + sb_mask
        seg, pose = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
        full = fast_compose(seg, pose, model_bytes, sb_total)
        delta = full['score'] - base['score']
        delta_vs_rgb = full['score'] - base_rgb['score']
        K_total = sum(len(v) for v in mask_patches.values())
        print(f"  >> {spec}: K_total={K_total} sb_mask={sb_mask}B sb_total={sb_total}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"d_baseline={delta:+.4f} d_rgb_only={delta_vs_rgb:+.4f} elapsed={elapsed:.1f}s",
              flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, len(res_top), K_total, sb_total,
                                     full['score'], full['seg_term'], full['pose_term'],
                                     delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
