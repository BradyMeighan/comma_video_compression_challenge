#!/usr/bin/env python
"""
REN v3: Train at MODEL RESOLUTION (384x512) directly.
No resolution mismatch — corrections directly affect SegNet's pixel grid.

Two-stage:
  Stage 1: KL divergence against ORIGINAL logits + margin loss, top-5000 hardest pixels
  Stage 2: Pure margin loss, strict error-pixel mask, QAT-aware

Usage: python train_ren_seg_v3.py --epochs 400 --batch-size 8
"""
import os, sys, struct, bz2, io, argparse, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import av, cv2, numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import SegNet, segnet_sd_path
from safetensors.torch import load_file

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
W_CAM, H_CAM = camera_size
MODEL_W, MODEL_H = segnet_model_input_size  # 512, 384


class RENv3(nn.Module):
    """Operates at model resolution (384x512). Simple residual CNN.
    No PixelUnshuffle — just direct convolutions at SegNet's eval grid."""
    def __init__(self, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))
        # Zero-init last layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """x: (B, 3, 384, 512) in [0, 255]"""
        return x + self.scale * self.net(x)


def decode_frames(video_path):
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return frames


class SegFrameDataset(Dataset):
    """Returns compressed odd frame at model res, original logits, original argmax, error mask."""
    def __init__(self, comp_model, orig_logits, orig_argmax, error_masks):
        self.comp = comp_model       # (N, 3, MODEL_H, MODEL_W) float
        self.logits = orig_logits    # (N, 5, MODEL_H, MODEL_W) float
        self.argmax = orig_argmax    # (N, MODEL_H, MODEL_W) long
        self.errors = error_masks    # (N, MODEL_H, MODEL_W) bool

    def __len__(self):
        return self.comp.shape[0]

    def __getitem__(self, idx):
        return self.comp[idx], self.logits[idx], self.argmax[idx], self.errors[idx]


def train(args):
    print(f"Device: {DEVICE}", flush=True)
    torch.manual_seed(42)

    # Load SegNet
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)

    # Load compressed frames
    archive_mkv = ROOT / 'submissions' / 'evenframe_meta_v4_crf34' / 'archive' / '0.mkv'
    if not archive_mkv.exists():
        archive_mkv = ROOT / 'submissions' / 'av1_repro' / 'archive' / '0.mkv'
    print(f"Loading compressed frames from {archive_mkv}...", flush=True)
    comp_raw = decode_frames(archive_mkv)

    # Load original frames
    print("Loading original frames...", flush=True)
    orig_raw = decode_frames(ROOT / 'videos' / '0.mkv')
    n_pairs = len(orig_raw) // 2

    # Match the REAL pipeline: Lanczos up to 1164x874 -> unsharp -> bilinear down to model res
    # This is what SegNet actually sees during evaluation
    print("Preprocessing frames to match real eval pipeline...", flush=True)
    _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    UNSHARP_KERNEL = (torch.outer(_r, _r) / (_r.sum()**2)).expand(3, 1, 9, 9).to(DEVICE)
    UNSHARP_STRENGTH = 0.45

    def pipeline_to_model_res(frame_np):
        """Replicate inflate.py pipeline: Lanczos up -> unsharp -> bilinear down."""
        pil = Image.fromarray(frame_np)
        pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
        x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), UNSHARP_KERNEL, padding=0, groups=3)
        x = (x + UNSHARP_STRENGTH * (x - blur)).clamp(0, 255)
        x_small = F.interpolate(x, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
        return x_small.squeeze(0).cpu()

    comp_model = torch.zeros(n_pairs, 3, MODEL_H, MODEL_W)
    orig_logits = torch.zeros(n_pairs, 5, MODEL_H, MODEL_W)
    orig_argmax = torch.zeros(n_pairs, MODEL_H, MODEL_W, dtype=torch.long)
    comp_argmax = torch.zeros(n_pairs, MODEL_H, MODEL_W, dtype=torch.long)

    with torch.inference_mode():
        for i in range(n_pairs):
            # Compressed odd frame through real pipeline
            c_t = pipeline_to_model_res(comp_raw[2*i+1])
            comp_model[i] = c_t

            # Original odd frame -> SegNet logits (golden targets)
            o_t = pipeline_to_model_res(orig_raw[2*i+1])
            o_logits = segnet(o_t.unsqueeze(0).to(DEVICE))
            orig_logits[i] = o_logits[0].cpu()
            orig_argmax[i] = o_logits.argmax(dim=1)[0].cpu()

            # Compressed -> SegNet prediction
            c_logits = segnet(c_t.unsqueeze(0).to(DEVICE))
            comp_argmax[i] = c_logits.argmax(dim=1)[0].cpu()

            if i % 100 == 0:
                print(f"  {i}/{n_pairs}...", flush=True)

    # Error masks: where compressed prediction != original prediction
    error_masks = (comp_argmax != orig_argmax)
    baseline_errors = error_masks.sum().item()
    baseline_seg = baseline_errors / (n_pairs * MODEL_H * MODEL_W)
    print(f"  Baseline: {baseline_errors:,} errors ({baseline_errors/n_pairs:.0f}/frame), "
          f"seg_dist={baseline_seg:.6f}, 100*seg={100*baseline_seg:.4f}", flush=True)

    # Dataset
    ds = SegFrameDataset(comp_model, orig_logits, orig_argmax, error_masks.float())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)

    # Model
    model = RENv3(ch=args.channels).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  RENv3: {n_params:,} params, ch={args.channels}", flush=True)

    if args.resume:
        print(f"  Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=DEVICE, weights_only=True))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    stage2_start = args.epochs + 1  # stage 1 only — stage 2 was destructive
    save_dir = ROOT / 'submissions' / 'evenframe_meta_v4_crf34' / 'archive'
    save_dir.mkdir(exist_ok=True)
    tag = args.tag if args.tag else ''
    save_path = save_dir / f'ren_seg_v3{tag}.pt'
    best_fixed = -999999
    epochs_without_improvement = 0

    print(f"\nTraining {args.epochs} epochs (S1: 1-{stage2_start}, S2: {stage2_start+1}-{args.epochs})")
    print(f"  S1: KL(orig_logits) + margin(0.2), top-5000 pixels")
    print(f"  S2: pure margin(0.5), strict error mask\n", flush=True)
    t0 = time.time()

    for epoch in range(args.start_epoch, args.epochs + 1):
        model.train()
        is_stage2 = epoch > stage2_start
        margin = 0.5 if is_stage2 else 0.2

        # LR schedule
        if is_stage2:
            progress = (epoch - stage2_start) / max(args.epochs - stage2_start, 1)
            lr = 5e-4 * (1 + math.cos(math.pi * progress)) / 2 + 1e-5
        else:
            progress = epoch / stage2_start
            lr = args.lr * (1 + math.cos(math.pi * progress)) / 2 + args.lr * 0.1
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        epoch_loss = 0
        n_batches = 0

        for comp, o_logits, o_argmax, err_mask in loader:
            comp = comp.to(DEVICE)
            o_logits = o_logits.to(DEVICE)
            o_argmax = o_argmax.to(DEVICE)
            err_mask = err_mask.to(DEVICE)

            optimizer.zero_grad()

            # Forward through REN (at model resolution — no resize needed!)
            enhanced = model(comp)

            # Forward through frozen SegNet
            logits = segnet(enhanced)
            B, C, H, W = logits.shape

            # === Margin loss ===
            target_logit = logits.gather(1, o_argmax.unsqueeze(1)).squeeze(1)
            mask_for_max = torch.ones_like(logits, dtype=torch.bool)
            mask_for_max.scatter_(1, o_argmax.unsqueeze(1), False)
            competitor = logits.clone()
            competitor[~mask_for_max] = float('-inf')
            max_other = competitor.max(dim=1).values
            margin_loss_map = F.relu(max_other - target_logit + margin)

            if is_stage2:
                # Asymmetric: heavy weight on errors, light weight on correct to preserve them
                weights = torch.where(err_mask > 0.5, 10.0, 0.1)
                weighted_margin = (margin_loss_map * weights).mean()
                loss = weighted_margin
            else:
                # Top-K hardest pixels per frame
                k = min(5000, H * W)
                flat = margin_loss_map.view(B, -1)
                topk_vals, _ = flat.topk(k, dim=1)
                margin_part = topk_vals.mean()

                # KL divergence against ORIGINAL logits (not compressed!)
                T = 2.0  # temperature
                kl = F.kl_div(
                    F.log_softmax(logits / T, dim=1),
                    F.softmax(o_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)

                loss = margin_part + 0.5 * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Evaluate
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            total_new_errors = 0
            with torch.inference_mode():
                for i in range(0, n_pairs, 32):
                    end = min(i + 32, n_pairs)
                    batch = comp_model[i:end].to(DEVICE)
                    enhanced = model(batch)
                    preds = segnet(enhanced).argmax(dim=1).cpu()
                    total_new_errors += (preds != orig_argmax[i:end]).sum().item()

            total_fixed = baseline_errors - total_new_errors
            new_seg = total_new_errors / (n_pairs * MODEL_H * MODEL_W)

            marker = ''
            if total_fixed > best_fixed:
                best_fixed = total_fixed
                torch.save(model.state_dict(), save_path)
                marker = ' <- saved'
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 10  # we only eval every 10 epochs

            stage = "S2" if is_stage2 else "S1"
            elapsed = time.time() - t0
            print(f"  [{stage}] Ep {epoch:3d}/{args.epochs}  loss={epoch_loss:.5f}  "
                  f"fixed={total_fixed:+d} ({total_fixed/n_pairs:+.1f}/fr)  "
                  f"100*seg={100*new_seg:.4f}  lr={lr:.1e}  "
                  f"[{elapsed:.0f}s]{marker}", flush=True)

            # Early stopping
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"  Early stopping: no improvement for {epochs_without_improvement} epochs")
                break
        else:
            stage = "S2" if is_stage2 else "S1"
            if epoch % 5 == 0:
                print(f"  [{stage}] Ep {epoch:3d}/{args.epochs}  loss={epoch_loss:.5f}  "
                      f"lr={lr:.1e}", flush=True)

    # Quantize
    print(f"\nBest: fixed {best_fixed} pixels ({best_fixed/n_pairs:.1f}/frame)")
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=True))

    sd = model.state_dict()
    buf = io.BytesIO()
    buf.write(struct.pack('<I', len(sd)))
    for name, tensor in sd.items():
        nb = name.encode('utf-8')
        buf.write(struct.pack('<I', len(nb)))
        buf.write(nb)
        buf.write(struct.pack('<I', tensor.ndim))
        for s in tensor.shape:
            buf.write(struct.pack('<I', s))
        t = tensor.float().flatten()
        scale = t.abs().max().item() / 127.0 if t.abs().max().item() > 0 else 1.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8)
        buf.write(struct.pack('<f', scale))
        buf.write(struct.pack('<I', len(q)))
        buf.write(q.cpu().numpy().tobytes())

    compressed = bz2.compress(buf.getvalue(), compresslevel=9)
    int8_path = save_dir / f'ren_seg_v3{tag}.int8.bz2'
    int8_path.write_bytes(compressed)

    new_seg = (baseline_errors - best_fixed) / (n_pairs * MODEL_H * MODEL_W)
    seg_gain = 100 * (baseline_seg - new_seg)
    rate_cost = 25 * len(compressed) / 37_545_489

    print(f"\n{'='*60}")
    print(f"  Baseline 100*seg: {100*baseline_seg:.4f}")
    print(f"  New 100*seg:      {100*new_seg:.4f}")
    print(f"  Seg gain:         {seg_gain:.4f}")
    print(f"  Model:            {len(compressed)} bytes ({len(compressed)/1024:.1f} KB)")
    print(f"  Rate cost:        {rate_cost:.4f}")
    print(f"  Net:              {-seg_gain + rate_cost:+.4f} ({'WINS' if seg_gain > rate_cost else 'LOSES'})")
    print(f"  Time:             {time.time()-t0:.0f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=400)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--channels', type=int, default=32)
    ap.add_argument('--resume', type=str, default='', help='Resume from checkpoint path')
    ap.add_argument('--start-epoch', type=int, default=1)
    ap.add_argument('--tag', type=str, default='', help='Suffix for saved files (e.g. "_ch64")')
    ap.add_argument('--patience', type=int, default=100, help='Early stop after N epochs without improvement (0=off)')
    train(ap.parse_args())
