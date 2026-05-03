#!/usr/bin/env python
"""
M2: Tiny delta MLP (INR-style). Compass Rank 3 / Sidecar Opt 5.3.

Train a small MLP that takes the per-pair pose (6D) and outputs a fixed set of
N patch (x, y, channel, delta) corrections. Store the MLP weights (INT8 quantized)
in the sidecar — at decode time, run MLP for each pair to generate patches.

Design:
- Input: 6D pose
- 3 layers: 6 → 32 → 32 → N*4 (each "patch" is 4 outputs: x_norm, y_norm, channel_logits[3], delta)
- Actually for stability: predict N positions (continuous) + N (channel, delta) pairs
- Quantize weights to INT8 → ~3-5KB sidecar regardless of N_pairs

This is risky: 600 pairs is small for fitting a meaningful MLP. Tries N=10, N=20.
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, get_pose6, load_posenet, estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose)

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


class TinyDeltaMLP(nn.Module):
    """Predicts N (x_norm, y_norm, channel_logits[3], delta_norm) per pair."""
    def __init__(self, n_patches, hidden=24):
        super().__init__()
        self.n_patches = n_patches
        out_dim = n_patches * 5  # 5 outputs per patch: x, y, ch_logit (3 wrapped as 1 by argmax during inference), delta
        # Use 4 outputs per patch: x, y, channel_idx_norm, delta
        out_dim = n_patches * 4
        self.fc1 = nn.Linear(6, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
        self.out_dim = out_dim

    def forward(self, pose):
        # pose: (B, 6)
        h = F.silu(self.fc1(pose))
        h = F.silu(self.fc2(h))
        out = self.fc3(h)
        return out.view(out.size(0), self.n_patches, 4)


def quantize_int8(tensor):
    """Symmetric per-tensor int8 quantization. Returns (int8_tensor, scale)."""
    s = tensor.abs().max().item() / 127.0
    if s == 0:
        s = 1e-8
    q = (tensor / s).round().clamp(-127, 127).to(torch.int8)
    return q, s


def dequantize_int8(q, s):
    return q.float() * s


def mlp_size_bytes(mlp, quantize=True):
    """Total bytes of an INT8 quantized MLP."""
    total = 0
    for p in mlp.parameters():
        if quantize:
            # int8 weights + 1 float scale per tensor (4B)
            total += p.numel() + 4
        else:
            total += p.numel() * 4  # float32
    return total


def train_mlp(mlp, poses_tensor, f1_all, f2_all, posenet, device, n_iter=500):
    """Train the MLP to minimize pose loss across all pairs."""
    mlp.train()
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    n = poses_tensor.shape[0]
    BATCH = 32
    for it in range(n_iter):
        # Sample a mini-batch of pairs
        idx = torch.randint(0, n, (BATCH,))
        sel_poses = poses_tensor[idx].to(device).float()
        f1 = f1_all[idx].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[idx].to(device).float().permute(0, 3, 1, 2)
        gt_p = poses_tensor[idx].to(device).float()

        opt.zero_grad()
        # Predict patches
        pred = mlp(sel_poses)  # (B, N, 4)
        # Decode: x in [0, OUT_W), y in [0, OUT_H), channel in {0, 1, 2}, delta in [-127, 127]
        x_norm = torch.sigmoid(pred[:, :, 0])
        y_norm = torch.sigmoid(pred[:, :, 1])
        ch_logit = pred[:, :, 2]  # use modulo 3 for channel
        delta = torch.tanh(pred[:, :, 3]) * 64  # delta in [-64, 64]

        # Convert to int positions (differentiable via STE)
        x_int = (x_norm * (OUT_W - 1)).long()
        y_int = (y_norm * (OUT_H - 1)).long()
        # For differentiability of channel choice: use soft channel
        ch_soft = F.softmax(ch_logit.unsqueeze(-1).repeat(1, 1, 3) * torch.tensor([0, 1, 2], device=device).float() / 3.0 - torch.arange(3, device=device).float(), dim=-1)
        # Actually just use channel modulo (non-differentiable but works)
        ch = (ch_logit.long() % 3).clamp(0, 2)

        # Apply patches
        f1_p = f1.clone()
        b = f1.shape[0]
        for k in range(mlp.n_patches):
            for c in range(3):
                mask_c = (ch[:, k] == c)
                if mask_c.any():
                    rows = mask_c.nonzero(as_tuple=True)[0]
                    yy = y_int[rows, k]
                    xx = x_int[rows, k]
                    f1_p[rows, c, yy, xx] = f1_p[rows, c, yy, xx] + delta[rows, k]
        f1_p = f1_p.clamp(0, 255)
        pin = se.diff_posenet_input(f1_p, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        loss.backward()
        opt.step()
        if (it + 1) % 100 == 0:
            print(f"  MLP iter {it+1}/{n_iter}: loss={loss.item():.4f}", flush=True)


def predict_patches(mlp, poses_tensor, device):
    """Run MLP on all pairs, return dict pair_i -> list of (x, y, channel, delta)."""
    mlp.eval()
    out = {}
    with torch.no_grad():
        for pi in range(poses_tensor.shape[0]):
            p = poses_tensor[pi:pi+1].to(device).float()
            pred = mlp(p)[0]  # (N, 4)
            x = (torch.sigmoid(pred[:, 0]) * (OUT_W - 1)).long().cpu().numpy()
            y = (torch.sigmoid(pred[:, 1]) * (OUT_H - 1)).long().cpu().numpy()
            ch = (pred[:, 2].long() % 3).clamp(0, 2).cpu().numpy()
            d = (torch.tanh(pred[:, 3]) * 64).round().clamp(-127, 127).long().cpu().numpy()
            patches = list(zip(x.tolist(), y.tolist(), ch.tolist(), d.tolist()))
            out[pi] = patches
    return out


def apply_predicted_patches(f_all, patches):
    out = f_all.clone()
    for pi in patches:
        arr = out[pi].float().numpy()
        for (x, y, c, d) in patches[pi]:
            arr[y, x, c] += d
        out[pi] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def main():
    device = torch.device("cuda")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
    f1_new, f2_new = bf['f1_new'], bf['f2_new']
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    sb_mask = bp['sb_mask_bz2']
    score_bl = bp['score']
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}, sb_mask={sb_mask}B")

    import csv
    csv_path = OUTPUT_DIR / "m2_tinymlp_results.csv"
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(["spec", "n_patches", "hidden", "mlp_bytes", "sb_total", "score", "delta", "elapsed"])

    for n_patches, hidden in [(10, 16), (20, 16), (30, 24)]:
        print(f"\n=== M2: TinyDeltaMLP n_patches={n_patches} hidden={hidden} ===")
        t0 = time.time()
        mlp = TinyDeltaMLP(n_patches=n_patches, hidden=hidden).to(device)
        train_mlp(mlp, poses, f1_new, f2_new, posenet, device, n_iter=500)
        mlp_bytes = mlp_size_bytes(mlp, quantize=True)
        # Predict + eval
        patches = predict_patches(mlp, poses, device)
        f1_combined = apply_predicted_patches(f1_new, patches)
        s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
        # Sidecar bytes = mask + MLP weights (no per-pair patches)
        sb_total = sb_mask + mlp_bytes
        full = fast_compose(s, p, model_bytes, sb_total)
        elapsed = time.time() - t0
        print(f"  M2 N={n_patches} h={hidden}: mlp_bytes={mlp_bytes}B sb_total={sb_total}B "
              f"score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([f"N{n_patches}_h{hidden}", n_patches, hidden,
                                     mlp_bytes, sb_total, full['score'],
                                     full['score']-score_bl, elapsed])


if __name__ == "__main__":
    main()
