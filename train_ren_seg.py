#!/usr/bin/env python
"""
Train REN v2 for SegNet boundary correction.
Uses Carlini-Wagner logit margin loss, boundary weighting, hard example mining.
Architecture: PixelUnshuffle(3) + depthwise separable convs.

Usage:
  python train_ren_seg.py [--epochs 500] [--batch-size 4] [--lr 2e-3]
"""
import os, sys, struct, bz2, io, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, cv2, numpy as np
from PIL import Image
from pathlib import Path
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    # Fallback: simple boundary distance using cv2
    def distance_transform_edt(mask):
        return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import SegNet, segnet_sd_path
from safetensors.torch import load_file

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W_CAM, H_CAM = camera_size
MODEL_W, MODEL_H = segnet_model_input_size


class DepthwiseSeparable(nn.Module):
    def __init__(self, ch, kernel=3):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, kernel, padding=kernel//2, groups=ch, bias=False)
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.dw(x)))


class RENv2(nn.Module):
    """Residual Enhancement Network v2 for SegNet correction.
    PixelUnshuffle(3) + depthwise separable convs + PixelShuffle(3).
    ~25K params."""
    def __init__(self, ch=40):
        super().__init__()
        in_ch = 3 * 9  # 27 channels after PixelUnshuffle(3)
        self.down = nn.PixelUnshuffle(3)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            DepthwiseSeparable(ch),
            DepthwiseSeparable(ch),
            DepthwiseSeparable(ch),
            DepthwiseSeparable(ch),
        )
        self.tail = nn.Conv2d(ch, in_ch, 3, padding=1, bias=True)
        self.up = nn.PixelShuffle(3)
        # Init tail to zero -> identity at start
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        x_norm = x / 255.0
        feat = self.down(x_norm)
        residual = self.tail(self.body(self.head(feat)))
        return (feat + residual)  # return at unshuffled resolution for training
        # For inference, caller does PixelShuffle

    def forward_full(self, x):
        """Full forward: input and output at original resolution.
        Pads to multiple of 3, processes, crops back."""
        _, _, H, W = x.shape
        # Pad to multiple of 3
        pad_h = (3 - H % 3) % 3
        pad_w = (3 - W % 3) % 3
        x_norm = x / 255.0
        if pad_h > 0 or pad_w > 0:
            x_norm = F.pad(x_norm, (0, pad_w, 0, pad_h), mode='reflect')
        feat = self.down(x_norm)
        residual = self.tail(self.body(self.head(feat)))
        out = self.up(feat + residual)
        # Crop back
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]
        return out.clamp(0, 1) * 255.0


def decode_frames(video_path, target_w=None, target_h=None, lanczos=False):
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        t = yuv420_to_rgb(frame)
        if target_w and target_h and (t.shape[0] != target_h or t.shape[1] != target_w):
            if lanczos:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                t = torch.from_numpy(np.array(pil))
        frames.append(t)
    container.close()
    return frames


class OddFrameDataset(Dataset):
    """Returns (compressed_odd_frame, gt_segmap, boundary_weight_map) for each pair."""
    def __init__(self, comp_frames, gt_segmaps, boundary_weights):
        self.comp = comp_frames  # list of (H,W,3) uint8 tensors
        self.gt = gt_segmaps     # (N, MODEL_H, MODEL_W) long
        self.bw = boundary_weights  # (N, MODEL_H, MODEL_W) float

    def __len__(self):
        return self.gt.shape[0]

    def __getitem__(self, idx):
        # Odd frame = index 2*idx + 1
        frame = self.comp[2 * idx + 1].permute(2, 0, 1).float()  # (3, H, W)
        return frame, self.gt[idx], self.bw[idx]


def compute_boundary_weights(gt_segmaps, sigma=4.0, base_weight=1.0, boundary_weight=10.0):
    """Compute distance-transform-based boundary weights."""
    N = gt_segmaps.shape[0]
    weights = torch.zeros(N, gt_segmaps.shape[1], gt_segmaps.shape[2])
    for i in range(N):
        seg = gt_segmaps[i].numpy()
        # Find boundaries: pixels adjacent to different class
        boundary = np.zeros_like(seg, dtype=bool)
        boundary[:-1, :] |= (seg[:-1, :] != seg[1:, :])
        boundary[1:, :] |= (seg[:-1, :] != seg[1:, :])
        boundary[:, :-1] |= (seg[:, :-1] != seg[:, 1:])
        boundary[:, 1:] |= (seg[:, :-1] != seg[:, 1:])

        if boundary.any():
            dist = distance_transform_edt(~boundary)
            w = base_weight + (boundary_weight - base_weight) * np.exp(-dist / sigma)
        else:
            w = np.full_like(seg, base_weight, dtype=np.float32)
        weights[i] = torch.from_numpy(w.astype(np.float32))
    return weights


def logit_margin_loss(logits, target, weight_map, margin=0.5):
    """Carlini-Wagner style margin loss. Only penalizes pixels where
    correct class doesn't win by `margin`. Correct pixels produce zero gradient."""
    B, C, H, W = logits.shape
    # Get correct class logit
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)  # (B, H, W)
    # Get max competing logit
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target.unsqueeze(1), False)
    competitor_logits = logits.clone()
    competitor_logits[~mask] = float('-inf')
    max_other = competitor_logits.max(dim=1).values  # (B, H, W)
    # Hinge: penalize when max_other > target - margin
    loss_per_pixel = F.relu(max_other - target_logit + margin)
    # Weight by boundary proximity
    weighted = loss_per_pixel * weight_map
    return weighted.mean()


def train(args):
    print(f"Device: {DEVICE}")
    torch.manual_seed(42)

    # Load compressed frames
    archive_mkv = ROOT / 'submissions' / 'evenframe_meta_v4_crf34' / 'archive' / '0.mkv'
    if not archive_mkv.exists():
        archive_mkv = ROOT / 'submissions' / 'av1_repro' / 'archive' / '0.mkv'
    print(f"Loading compressed frames from {archive_mkv}...")
    comp_frames = decode_frames(archive_mkv, target_w=W_CAM, target_h=H_CAM, lanczos=True)
    print(f"  {len(comp_frames)} frames")

    # Load GT frames and compute SegNet targets
    print("Loading GT frames and computing SegNet targets...")
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)

    gt_path = ROOT / 'videos' / '0.mkv'
    gt_frames = decode_frames(gt_path)
    n_pairs = len(gt_frames) // 2

    # Compute golden target segmaps from original odd frames
    gt_segmaps = torch.zeros(n_pairs, MODEL_H, MODEL_W, dtype=torch.long)
    with torch.inference_mode():
        for i in range(0, n_pairs, 16):
            end = min(i + 16, n_pairs)
            batch = torch.stack([gt_frames[2*j+1].permute(2,0,1).float() for j in range(i, end)]).to(DEVICE)
            batch_small = F.interpolate(batch, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            preds = segnet(batch_small).argmax(dim=1).cpu()
            gt_segmaps[i:end] = preds
    print(f"  {n_pairs} target segmaps computed")

    # Compute baseline errors
    baseline_errors = 0
    with torch.inference_mode():
        for i in range(0, n_pairs, 16):
            end = min(i + 16, n_pairs)
            batch = torch.stack([comp_frames[2*j+1].permute(2,0,1).float() for j in range(i, end)]).to(DEVICE)
            batch_small = F.interpolate(batch, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            preds = segnet(batch_small).argmax(dim=1).cpu()
            baseline_errors += (preds != gt_segmaps[i:end]).sum().item()
    baseline_seg = baseline_errors / (n_pairs * MODEL_H * MODEL_W)
    print(f"  Baseline: {baseline_errors:,} errors, seg_dist={baseline_seg:.6f}, 100*seg={100*baseline_seg:.4f}")

    # Compute boundary weights
    print("Computing boundary weights...")
    boundary_weights = compute_boundary_weights(gt_segmaps)

    # Create dataset
    ds = OddFrameDataset(comp_frames, gt_segmaps, boundary_weights)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # Create model
    model = RENv2(ch=args.channels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  RENv2: {n_params:,} params, ch={args.channels}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)

    # Stage boundaries
    stage2_start = args.epochs * 3 // 10  # 30% soft, 70% hard
    save_dir = ROOT / 'submissions' / 'evenframe_meta_v4_crf34' / 'archive'
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / 'ren_seg_v2.pt'
    best_fixed = 0

    print(f"\nTraining {args.epochs} epochs (stage1: 1-{stage2_start}, stage2: {stage2_start+1}-{args.epochs})\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        is_stage2 = epoch > stage2_start
        margin = 0.7 if is_stage2 else 0.2

        # Cosine LR
        if is_stage2:
            progress = (epoch - stage2_start) / (args.epochs - stage2_start)
            lr = 5e-4 * (1 + math.cos(math.pi * progress)) / 2 + 1e-5
        else:
            lr = args.lr
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        epoch_loss = 0
        n_batches = 0
        for comp_frame, gt_seg, bw in loader:
            comp_frame = comp_frame.to(DEVICE)
            gt_seg = gt_seg.to(DEVICE)
            bw = bw.to(DEVICE)

            optimizer.zero_grad()

            # Forward through REN at full res, then bilinear resize to model res
            enhanced = model.forward_full(comp_frame)
            enhanced_small = F.interpolate(enhanced, size=(MODEL_H, MODEL_W),
                                           mode='bilinear', align_corners=False)

            # Forward through frozen SegNet
            logits = segnet(enhanced_small)

            # Loss
            loss = logit_margin_loss(logits, gt_seg, bw, margin=margin)

            if not is_stage2:
                # Stage 1: add soft KL loss for general alignment
                with torch.no_grad():
                    orig_small = F.interpolate(comp_frame, size=(MODEL_H, MODEL_W),
                                               mode='bilinear', align_corners=False)
                    orig_logits = segnet(orig_small)
                kl = F.kl_div(F.log_softmax(logits, dim=1),
                              F.softmax(orig_logits, dim=1),
                              reduction='batchmean')
                loss = loss + 0.1 * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            total_fixed = 0
            with torch.inference_mode():
                for i in range(0, n_pairs, 16):
                    end = min(i + 16, n_pairs)
                    batch = torch.stack([comp_frames[2*j+1].permute(2,0,1).float()
                                        for j in range(i, end)]).to(DEVICE)
                    enhanced = model.forward_full(batch)
                    enhanced_small = F.interpolate(enhanced, size=(MODEL_H, MODEL_W),
                                                   mode='bilinear', align_corners=False)
                    preds = segnet(enhanced_small).argmax(dim=1).cpu()
                    new_errors = (preds != gt_segmaps[i:end]).sum().item()
                    # Count for this batch
                    orig_errors = (gt_segmaps[i:end] !=
                                   segnet(F.interpolate(batch, size=(MODEL_H, MODEL_W),
                                          mode='bilinear', align_corners=False)).argmax(dim=1).cpu()
                                  ).sum().item()
                    total_fixed += orig_errors - new_errors

            marker = ''
            if total_fixed > best_fixed:
                best_fixed = total_fixed
                torch.save(model.state_dict(), save_path)
                marker = ' <- saved'

            new_errors_total = baseline_errors - total_fixed
            new_seg = new_errors_total / (n_pairs * MODEL_H * MODEL_W)
            stage = "S2" if is_stage2 else "S1"
            print(f"  [{stage}] Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.5f}  "
                  f"fixed={total_fixed:+d}  seg={100*new_seg:.4f}  "
                  f"lr={lr:.1e}{marker}", flush=True)
        else:
            stage = "S2" if is_stage2 else "S1"
            print(f"  [{stage}] Epoch {epoch:3d}/{args.epochs}  loss={epoch_loss:.5f}  lr={lr:.1e}", flush=True)

    # Quantize best model
    print(f"\nBest: fixed {best_fixed} pixels")
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))

    # Int8 + bz2
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
        t = tensor.float().flatten()
        scale = t.abs().max().item() / 127.0 if t.abs().max().item() > 0 else 1.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8)
        buf.write(struct.pack('<f', scale))
        buf.write(struct.pack('<I', len(q)))
        buf.write(q.numpy().tobytes())

    compressed = bz2.compress(buf.getvalue(), compresslevel=9)
    int8_path = save_dir / 'ren_seg_v2.int8.bz2'
    int8_path.write_bytes(compressed)

    new_seg = (baseline_errors - best_fixed) / (n_pairs * MODEL_H * MODEL_W)
    seg_gain = 100 * (baseline_seg - new_seg)
    rate_cost = 25 * len(compressed) / 37_545_489

    print(f"\nResults:")
    print(f"  Baseline 100*seg: {100*baseline_seg:.4f}")
    print(f"  New 100*seg:      {100*new_seg:.4f}")
    print(f"  Seg gain:         {seg_gain:.4f}")
    print(f"  Model size:       {len(compressed)} bytes ({len(compressed)/1024:.1f} KB)")
    print(f"  Rate cost:        {rate_cost:.4f}")
    print(f"  Net score:        {-seg_gain + rate_cost:+.4f} ({'WINS' if seg_gain > rate_cost else 'LOSES'})")
    print(f"  Time: {time.time()-t0:.0f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--channels', type=int, default=40)
    train(ap.parse_args())
