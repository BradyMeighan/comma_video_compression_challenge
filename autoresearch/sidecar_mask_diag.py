#!/usr/bin/env python
"""
Mask sidecar DIAGNOSTIC: does the gradient point the right direction
for K=1? If so, try iterative greedy.

For 5 hardest residual pairs:
  - Compute baseline pose loss
  - Find top-1 mask flip via gradient
  - Actually flip it, regenerate, recompute loss
  - Did loss go down?
"""
import sys, os, time
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
from sidecar_stack import fast_eval, fast_compose

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def gen_forward_with_oh_mask(gen, m_oh, p, device):
    """Forward gen using one-hot mask (b, MH, MW, 5)."""
    co = coords(m_oh.shape[0], MODEL_H, MODEL_W, device)
    e_emb = m_oh @ gen.trunk.emb.weight
    e = F.interpolate(e_emb.permute(0, 3, 1, 2), co.shape[-2:], mode='bilinear', align_corners=False)
    s = gen.trunk.s1(gen.trunk.stem(torch.cat([e, co], 1)))
    z = gen.trunk.up(gen.trunk.d1(gen.trunk.down(s)))
    feat = gen.trunk.f1(gen.trunk.fuse(torch.cat([z, s], 1)))
    cond = gen.pose_mlp(p)
    f1_out = gen.h1(feat, cond)
    f2_out = gen.h2(feat)
    f1u = F.interpolate(f1_out, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
    f2u = F.interpolate(f2_out, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
    return f1u, f2u


def compute_pose_loss(gen, m_oh, p, gt_p, posenet, device):
    f1u, f2u = gen_forward_with_oh_mask(gen, m_oh, p, device)
    pin = se.diff_posenet_input(f1u, f2u)
    fp = get_pose6(posenet, pin).float()
    return ((fp - gt_p) ** 2).sum()


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print("Loading...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]

    print("\n=== K=1 mask flip diagnostic on 5 hardest pairs ===\n", flush=True)
    for pi in rank[:5]:
        pi = int(pi)
        m = data["val_masks"][pi:pi+1].to(device).long()
        p = data["val_poses"][pi:pi+1].to(device).float()
        gt_p = p.clone()  # in this dataset val_poses serve as both input and gt

        # Baseline loss
        m_oh = F.one_hot(m, num_classes=5).float()
        with torch.no_grad():
            base_loss = compute_pose_loss(gen, m_oh, p, gt_p, posenet, device).item()
        print(f"Pair {pi}: baseline loss = {base_loss:.6f}", flush=True)

        # Compute gradient w.r.t. one-hot mask
        m_oh_g = m_oh.clone().requires_grad_(True)
        loss = compute_pose_loss(gen, m_oh_g, p, gt_p, posenet, device)
        grad = torch.autograd.grad(loss, m_oh_g)[0]  # (1, MH, MW, 5)

        # For each pixel, score each alternative class
        cur_class = m  # (1, MH, MW)
        grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
        candidate_delta = grad - grad_cur.unsqueeze(-1)
        for cls in range(5):
            mask_cur = (cur_class == cls)
            candidate_delta[..., cls][mask_cur] = float('inf')
        best_delta, best_class = candidate_delta.min(dim=-1)

        # Find top-5 candidate flips
        flat = best_delta.contiguous().reshape(-1)
        topk_vals, topk_idx = torch.topk(-flat, 5)

        # For each candidate, actually flip and check loss
        for k in range(5):
            idx = topk_idx[k].item()
            y = idx // MODEL_W; x = idx % MODEL_W
            new_cls = best_class[0, y, x].item()
            old_cls = m[0, y, x].item()
            predicted_delta = -topk_vals[k].item()  # negated back

            # Apply this single flip
            m_test = m.clone()
            m_test[0, y, x] = new_cls
            m_test_oh = F.one_hot(m_test, num_classes=5).float()
            with torch.no_grad():
                new_loss = compute_pose_loss(gen, m_test_oh, p, gt_p, posenet, device).item()
            actual_delta = new_loss - base_loss
            verdict = "OK" if actual_delta < -0.01 * abs(base_loss) else ("flat" if abs(actual_delta) < 0.01 * abs(base_loss) else "WRONG")
            print(f"  candidate {k+1}: ({x:>3d},{y:>3d}) {old_cls}->{new_cls}  "
                  f"predicted_delta={predicted_delta:+.5f}  actual_delta={actual_delta:+.5f}  [{verdict}]",
                  flush=True)
        print()


if __name__ == "__main__":
    main()
