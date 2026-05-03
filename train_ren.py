#!/usr/bin/env python
"""
Train Residual Enhancement Network (REN) with task-aware loss.
Adapted from PR #49's approach. Trains on compressed/decoded frames
to learn per-pixel corrections that minimize SegNet/PoseNet distortion.

Usage:
  python train_ren.py [--epochs 80] [--batch-size 2] [--lr 1e-3]
"""
import os, sys, struct, bz2, io, argparse, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class REN(nn.Module):
    """Residual Enhancement Network. ~25K params.
    PixelUnshuffle(2) -> 4 Conv layers -> PixelShuffle(2)."""
    def __init__(self, features=32):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)
        # Init last layer to zero -> identity at start
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0


def decode_all_frames(video_path, target_w=None, target_h=None, lanczos=False):
    fmt = 'hevc' if str(video_path).endswith('.hevc') else None
    container = av.open(str(video_path), format=fmt)
    frames = []
    for frame in container.decode(container.streams.video[0]):
        t = yuv420_to_rgb(frame)
        if target_w and target_h and (t.shape[0] != target_h or t.shape[1] != target_w):
            if lanczos:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                t = torch.from_numpy(np.array(pil))
            else:
                t = F.interpolate(
                    t.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(target_h, target_w), mode='bicubic', align_corners=False
                ).clamp(0, 255).squeeze(0).permute(1, 2, 0).round().to(torch.uint8)
        frames.append(t)
    container.close()
    return frames


class ConsecutivePairDataset(Dataset):
    def __init__(self, comp_frames, gt_frames):
        assert len(comp_frames) == len(gt_frames)
        self.comp = comp_frames
        self.gt = gt_frames

    def __len__(self):
        return len(self.comp) - 1

    def __getitem__(self, idx):
        ca = self.comp[idx].permute(2, 0, 1).float()
        cb = self.comp[idx + 1].permute(2, 0, 1).float()
        ga = self.gt[idx].permute(2, 0, 1).float()
        gb = self.gt[idx + 1].permute(2, 0, 1).float()
        return ca, cb, ga, gb


def compute_loss(model, posenet, segnet, comp_a, comp_b, gt_a, gt_b, w_seg, w_temp):
    inf_a = model(comp_a)
    inf_b = model(comp_b)

    # Build frame pairs for model evaluation: (B, 2, H, W, 3)
    pair_inf = torch.stack([inf_a.permute(0, 2, 3, 1),
                            inf_b.permute(0, 2, 3, 1)], dim=1)
    pair_gt = torch.stack([gt_a.permute(0, 2, 3, 1),
                           gt_b.permute(0, 2, 3, 1)], dim=1)

    # PoseNet loss
    posenet_in_inf = posenet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        posenet_in_gt = posenet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
        posenet_out_gt = posenet(posenet_in_gt)
    posenet_out_inf = posenet(posenet_in_inf)
    loss_pose = sum(
        F.mse_loss(posenet_out_inf[h.name][..., :h.out // 2],
                    posenet_out_gt[h.name][..., :h.out // 2])
        for h in posenet.hydra.heads
    )

    # SegNet loss (KL divergence on logits)
    segnet_in_inf = segnet.preprocess_input(pair_inf.permute(0, 1, 4, 2, 3))
    with torch.no_grad():
        segnet_in_gt = segnet.preprocess_input(pair_gt.permute(0, 1, 4, 2, 3))
        logits_gt = segnet(segnet_in_gt)
    logits_inf = segnet(segnet_in_inf)
    loss_seg = F.kl_div(
        F.log_softmax(logits_inf, dim=1),
        F.softmax(logits_gt, dim=1),
        reduction='batchmean'
    )

    # Temporal consistency: corrections should be smooth across frames
    corr_a = (inf_a - comp_a) / 255.0
    corr_b = (inf_b - comp_b) / 255.0
    loss_temp = F.l1_loss(corr_a, corr_b)

    loss = loss_pose + w_seg * loss_seg + w_temp * loss_temp
    return loss, loss_pose.item(), loss_seg.item(), loss_temp.item()


def save_int8_bz2(model, path):
    """Save model weights as int8 quantized + bz2 compressed."""
    sd = model.state_dict()
    buf = io.BytesIO()
    buf.write(struct.pack('<I', len(sd)))
    for name, tensor in sd.items():
        name_bytes = name.encode('utf-8')
        buf.write(struct.pack('<I', len(name_bytes)))
        buf.write(name_bytes)
        buf.write(struct.pack('<I', tensor.ndim))
        for s in tensor.shape:
            buf.write(struct.pack('<I', s))
        # Quantize to int8
        t = tensor.float().flatten()
        scale = t.abs().max().item() / 127.0 if t.abs().max().item() > 0 else 1.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8)
        buf.write(struct.pack('<f', scale))
        q_bytes = q.cpu().numpy().tobytes()
        buf.write(struct.pack('<I', len(q_bytes)))
        buf.write(q_bytes)

    compressed = bz2.compress(buf.getvalue(), compresslevel=9)
    Path(path).write_bytes(compressed)
    return len(compressed)


def train(args):
    print(f"Device: {DEVICE}")
    torch.manual_seed(1234)
    np.random.seed(1234)

    W, H = camera_size
    archive_mkv = ROOT / 'submissions' / 'av1_repro' / 'archive' / '0.mkv'
    gt_path = ROOT / 'videos' / '0.mkv'

    if not archive_mkv.exists():
        print("ERROR: No compressed archive. Run compress.py first.")
        sys.exit(1)

    print(f"Loading compressed frames from {archive_mkv}...")
    comp_frames = decode_all_frames(archive_mkv, target_w=W, target_h=H, lanczos=True)
    print(f"  {len(comp_frames)} frames")

    print(f"Loading GT frames from {gt_path}...")
    gt_frames = decode_all_frames(gt_path)
    print(f"  {len(gt_frames)} frames")

    assert len(comp_frames) == len(gt_frames)

    split = min(1000, len(comp_frames) - 2)
    train_ds = ConsecutivePairDataset(comp_frames[:split], gt_frames[:split])
    val_ds = ConsecutivePairDataset(comp_frames[split:], gt_frames[split:])
    print(f"  Train: {len(train_ds)} pairs, Val: {len(val_ds)} pairs")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = REN(features=args.features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  REN: {n_params:,} params ({n_params/1024:.1f} KB raw)")

    print("Loading DistortionNet (SegNet + PoseNet)...")
    distortion_net = DistortionNet().to(DEVICE).eval()
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, DEVICE)
    for p in distortion_net.parameters():
        p.requires_grad_(False)
    posenet = distortion_net.posenet
    segnet = distortion_net.segnet

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Calibrate loss weights
    print("\nCalibrating loss weights...")
    model.eval()
    ca, cb, ga, gb = train_ds[0]
    ca = ca.unsqueeze(0).to(DEVICE)
    cb = cb.unsqueeze(0).to(DEVICE)
    ga = ga.unsqueeze(0).to(DEVICE)
    gb = gb.unsqueeze(0).to(DEVICE)
    model.train()
    _, lp0, ls0, lt0 = compute_loss(model, posenet, segnet, ca, cb, ga, gb, 0.1, 0.005)
    print(f"  Identity baseline — pose: {lp0:.6f}, seg: {ls0:.6f}, temp: {lt0:.6f}")

    # Score formula: 100*seg + sqrt(10*pose) + 25*rate
    # seg has 100x weight, so boost seg loss significantly
    w_seg = max(0.1, min(100.0, 10.0 * lp0 / ls0)) if ls0 > 0 else 1.0
    w_temp = 0.005
    print(f"  Weights: w_seg={w_seg:.4f}, w_temp={w_temp:.4f}")
    del ca, cb, ga, gb
    torch.cuda.empty_cache()

    save_dir = ROOT / 'submissions' / 'av1_repro' / 'archive'
    save_path = save_dir / 'ren_model.pt'
    best_val = float('inf')

    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size})...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, train_lp, train_ls = 0, 0, 0
        n_batches = 0

        for comp_a, comp_b, gt_a, gt_b in train_loader:
            comp_a = comp_a.to(DEVICE)
            comp_b = comp_b.to(DEVICE)
            gt_a = gt_a.to(DEVICE)
            gt_b = gt_b.to(DEVICE)

            optimizer.zero_grad()
            loss, lp, ls, lt = compute_loss(model, posenet, segnet,
                                            comp_a, comp_b, gt_a, gt_b, w_seg, w_temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_lp += lp
            train_ls += ls
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)
        train_lp /= max(n_batches, 1)
        train_ls /= max(n_batches, 1)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            val_loss, val_lp, val_ls = 0, 0, 0
            n_val = 0
            with torch.no_grad():
                for comp_a, comp_b, gt_a, gt_b in val_loader:
                    comp_a = comp_a.to(DEVICE)
                    comp_b = comp_b.to(DEVICE)
                    gt_a = gt_a.to(DEVICE)
                    gt_b = gt_b.to(DEVICE)
                    loss, lp, ls, lt = compute_loss(model, posenet, segnet,
                                                    comp_a, comp_b, gt_a, gt_b, w_seg, w_temp)
                    val_loss += loss.item()
                    val_lp += lp
                    val_ls += ls
                    n_val += 1

            val_loss /= max(n_val, 1)
            val_lp /= max(n_val, 1)
            val_ls /= max(n_val, 1)

            marker = ''
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), save_path)
                marker = '  <- saved'

            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.6f} (pose={train_lp:.6f} seg={train_ls:.4f})  "
                  f"val={val_loss:.6f} (pose={val_lp:.6f} seg={val_ls:.4f})  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}")
        else:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={train_loss:.6f} (pose={train_lp:.6f} seg={train_ls:.4f})")

    # Load best and quantize
    print("\nQuantizing best model to int8 + bz2...")
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    int8_path = save_dir / 'ren_model.int8.bz2'
    size = save_int8_bz2(model, int8_path)
    print(f"  Saved: {int8_path} ({size/1024:.1f} KB)")
    print(f"  Best val_loss: {best_val:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--features', type=int, default=32)
    args = parser.parse_args()
    train(args)
