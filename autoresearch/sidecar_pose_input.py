#!/usr/bin/env python
"""
POSE INPUT patches: modify the 6D pose vector that feeds the generator.

Format: u16 pair_idx, 6 × i8 delta = 8 bytes per patched pair.
Way more compact than pixel patches.

Method: gradient through gen + posenet w.r.t. pose vector. Quantize to int8 deltas.
Apply: at decode time, add quantized delta back to pose before gen forward.

Hypothesis: Generator was trained on continuous pose. A small pose delta could
shift the gen output toward ground truth more efficiently than RGB patches.
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
from sidecar_stack import (get_dist_net, fast_eval, fast_compose,
                            find_pose_patches_for_pairs)
from sidecar_adaptive import sparse_sidecar_size, apply_sparse_patches
from sidecar_mask_verified import (verified_greedy_mask, mask_sidecar_size,
                                     regenerate_frames_from_masks)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_pose_input_deltas(gen, masks, poses, posenet, pair_indices, n_iter, device,
                             scale=0.01):
    """For each pair, find a pose delta (6 floats, quantized to int8) that
    reduces pose loss when applied to the gen input pose.

    scale: quantization scale (delta_int8 * scale -> actual pose delta).
    Final stored: int8 vector × 6.
    """
    out = {}  # pair_i -> int8 delta vector (6,)
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        m = masks[sel].to(device).long()
        p = poses[sel].to(device).float()
        gt_p = p.clone()  # use input pose as gt for posenet output

        # Initialize delta as zeros
        delta = torch.zeros((b, 6), device=device, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=0.005)

        for _ in range(n_iter):
            opt.zero_grad()
            p_mod = p + delta * scale  # add quantized delta back
            with torch.enable_grad():
                p1, p2 = gen(m, p_mod)
                f1u = F.interpolate(p1, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
                f2u = F.interpolate(p2, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
                pin = se.diff_posenet_input(f1u, f2u)
                fp = get_pose6(posenet, pin).float()
                # We want PoseNet output to match GT pose
                loss = ((fp - gt_p) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.clamp_(-127, 127)

        d_np = delta.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            if d_np[bi].any():
                out[pair_i] = d_np[bi]
    return out


def pose_sidecar_size(pose_deltas):
    if not pose_deltas:
        return 0
    parts = [struct.pack("<H", len(pose_deltas))]
    for pair_i, d in sorted(pose_deltas.items()):
        parts.append(struct.pack("<H6b", pair_i, *d.tolist()))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def apply_pose_deltas_and_regen(gen, masks_cpu, poses, pose_deltas, device,
                                  scale=0.01, batch_size=8):
    """Apply pose deltas, run gen, return new f1, f2."""
    n = poses.shape[0]
    f1_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    f2_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    poses_mod = poses.clone()
    for pi, d in pose_deltas.items():
        d_t = torch.from_numpy(d.astype(np.float32) * scale)
        poses_mod[pi] = poses[pi] + d_t.to(poses.device)
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

    csv_path = OUTPUT_DIR / "pose_input_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_pairs", "sb_pose", "score",
                                 "pose_term", "delta", "elapsed"])

    BEST_KNOWN = -0.0287

    # Test pose-input deltas alone (no other sidecar)
    for spec, n_top, scale, n_iter in [
        ("pose_in_top600_s0.01", 600, 0.01, 50),
        ("pose_in_top600_s0.005", 600, 0.005, 50),
        ("pose_in_top300_s0.01", 300, 0.01, 80),
    ]:
        print(f"\n=== {spec} ===", flush=True)
        t0 = time.time()
        pair_indices = [int(x) for x in rank[:n_top]]
        deltas = find_pose_input_deltas(gen, masks_cpu, poses, posenet,
                                          pair_indices, n_iter=n_iter, device=device, scale=scale)
        sb = pose_sidecar_size(deltas)
        f1_p, f2_p = apply_pose_deltas_and_regen(gen, masks_cpu, poses, deltas, device, scale=scale)
        s, p = fast_eval(f1_p, f2_p, data["val_rgb"], device)
        full = fast_compose(s, p, model_bytes, sb)
        delta = full['score'] - base['score']
        elapsed = time.time() - t0
        print(f"  >> {spec}: n_pairs={len(deltas)} sb={sb}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={delta:+.4f} ({elapsed:.0f}s)", flush=True)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([spec, len(deltas), sb, full['score'],
                                     full['pose_term'], delta, elapsed])

    # Combined: pose_input + mask K=1 + RGB lean tier
    print("\n=== Combined: pose_input + mask K=1 top600 + RGB 250_K5+250_K2 ===", flush=True)
    t0 = time.time()
    # 1. Find pose deltas
    deltas = find_pose_input_deltas(gen, masks_cpu, poses, posenet,
                                       [int(x) for x in rank[:600]],
                                       n_iter=50, device=device, scale=0.01)
    sb_pose = pose_sidecar_size(deltas)
    print(f"  Pose deltas: {len(deltas)} pairs, sb={sb_pose}B", flush=True)

    # 2. Apply pose deltas, regen frames
    poses_mod = poses.clone()
    for pi, d in deltas.items():
        d_t = torch.from_numpy(d.astype(np.float32) * 0.01)
        poses_mod[pi] = poses[pi] + d_t.to(poses.device)

    # 3. Now build mask sidecar on the modified-pose state
    # (mask is independent of pose for the gradient computation)
    print("  Building mask K=1 top600 with modified pose input...", flush=True)
    t1 = time.time()
    mask_patches = {}
    for i, pi in enumerate(rank[:600]):
        pi = int(pi)
        m = masks_cpu[pi:pi+1].to(device).long()
        p_use = poses_mod[pi:pi+1].to(device).float()
        gt_p = poses[pi:pi+1].to(device).float()  # gt is original pose
        accepted, _ = verified_greedy_mask(gen, m, p_use, gt_p, posenet, device,
                                             K=1, n_candidates=10)
        if accepted:
            mask_patches[pi] = accepted
        if (i + 1) % 200 == 0:
            print(f"    ... {i+1}/600 ({time.time()-t1:.0f}s)", flush=True)
    sb_mask = mask_sidecar_size(mask_patches)
    print(f"  Mask: {len(mask_patches)} pairs, sb={sb_mask}B", flush=True)

    # 4. Apply mask + regenerate
    new_masks = masks_cpu.clone()
    for pi, patches in mask_patches.items():
        for (x, y, c) in patches:
            new_masks[pi, y, x] = c
    f1_new, f2_new = apply_pose_deltas_and_regen(gen, new_masks, poses, deltas, device, scale=0.01)

    # 5. RGB on these frames
    print("  RGB 250_K5+250_K2 on combined frames...", flush=True)
    t2 = time.time()
    p_top = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_pose_patches_for_pairs(
        f1_new, f2_new, poses, posenet,
        [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    sb_rgb = sparse_sidecar_size(rgb_patches)
    f1_combined = apply_sparse_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    sb_total = sb_pose + sb_mask + sb_rgb
    full = fast_compose(s, p, model_bytes, sb_total)
    delta = full['score'] - base['score']
    elapsed = time.time() - t0
    print(f"  >> Combined: sb_pose={sb_pose}B sb_mask={sb_mask}B sb_rgb={sb_rgb}B "
          f"sb_total={sb_total}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
          f"delta={delta:+.4f} d_vs_best={delta-BEST_KNOWN:+.4f} ({elapsed:.0f}s)", flush=True)
    with open(csv_path, 'a', newline='') as f:
        csv.writer(f).writerow(["combined_pose+mask+rgb", len(mask_patches),
                                 sb_total, full['score'], full['pose_term'], delta, elapsed])

    print(f"\nDone. {csv_path}", flush=True)


if __name__ == "__main__":
    main()
