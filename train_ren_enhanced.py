#!/usr/bin/env python
"""
Enhanced REN training: resume from checkpoint with experimental objectives.

Toggleable techniques (each can be enabled independently):
  --qat              Quantization-aware training (simulate int8 forward pass)
  --margin-anneal    Anneal margin from 0.2 -> target over training
  --margin-target    Final margin value (default 0.5)
  --hard-oversample  Sample pairs proportional to baseline error count
  --ema              Track EMA of weights, save EMA model
  --ema-decay        EMA decay rate (default 0.999)
  --input-noise      Add gaussian noise to input during training (sigma)
  --margin-only      Skip KL loss, pure margin loss only

Usage examples:
  # QAT only
  python train_ren_enhanced.py --resume submissions/evenframe_meta_v4_crf34/archive/ren_seg_v3_ch48.pt \
      --channels 48 --epochs 500 --tag _ch48_qat --qat

  # All enhancements
  python train_ren_enhanced.py --resume <ckpt> --channels 48 --epochs 1000 \
      --tag _ch48_all --qat --margin-anneal --hard-oversample --ema --input-noise 1.0
"""
import os, sys, struct, bz2, io, argparse, math, time, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
MODEL_W, MODEL_H = segnet_model_input_size


class FakeQuant(torch.autograd.Function):
    """Straight-through int8 fake quantization."""
    @staticmethod
    def forward(ctx, x, scale):
        # Quantize then dequantize
        q = (x / scale).round().clamp(-127, 127)
        return q * scale

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def fake_quant(x):
    """Quantize x to int8 with per-tensor scale."""
    scale = x.abs().max().detach() / 127.0 + 1e-8
    return FakeQuant.apply(x, scale)


class RENv3(nn.Module):
    def __init__(self, ch=32, qat=False):
        super().__init__()
        self.qat = qat
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, 3, 3, padding=1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _qat_forward(self, x):
        # Fake-quantize each layer's weights and activations
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                # Quantize weights
                w_q = fake_quant(layer.weight)
                b = layer.bias
                x = F.conv2d(x, w_q, b, padding=layer.padding, stride=layer.stride)
            else:
                x = layer(x)
        return x

    def forward(self, x):
        if self.qat:
            res = self._qat_forward(x)
        else:
            res = self.net(x)
        scale_val = fake_quant(self.scale) if self.qat else self.scale
        return x + scale_val * res


def decode_frames(video_path):
    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return frames


def pipeline_to_model_res(frame_np, kernel, strength=0.45):
    pil = Image.fromarray(frame_np)
    pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
    x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), kernel, padding=0, groups=3)
    x = (x + strength * (x - blur)).clamp(0, 255)
    return F.interpolate(x, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False).squeeze(0).cpu()


class FrameDataset(Dataset):
    def __init__(self, comp, gt_logits, gt_argmax, err_count):
        self.comp = comp
        self.logits = gt_logits
        self.argmax = gt_argmax
        self.err_count = err_count

    def __len__(self):
        return self.comp.shape[0]

    def __getitem__(self, idx):
        return self.comp[idx], self.logits[idx], self.argmax[idx]


def margin_loss(logits, target, margin):
    target_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target.unsqueeze(1), False)
    competitor = logits.clone()
    competitor[~mask] = float('-inf')
    max_other = competitor.max(dim=1).values
    return F.relu(max_other - target_logit + margin)


def evaluate(model, comp, gt_argmax, segnet, n_pairs):
    model.eval()
    total_errors = 0
    with torch.inference_mode():
        for i in range(0, n_pairs, 32):
            end = min(i + 32, n_pairs)
            batch = comp[i:end].to(DEVICE)
            enhanced = model(batch)
            preds = segnet(enhanced).argmax(dim=1).cpu()
            total_errors += (preds != gt_argmax[i:end]).sum().item()
    return total_errors


def train(args):
    print(f"Device: {DEVICE}")
    torch.manual_seed(42)

    # Load SegNet
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)

    # Load frames
    archive_mkv = ROOT / 'submissions' / 'evenframe_meta_v4_crf34' / 'archive' / '0.mkv'
    print(f"Loading from {archive_mkv}...", flush=True)
    comp_raw = decode_frames(archive_mkv)
    orig_raw = decode_frames(ROOT / 'videos' / '0.mkv')
    n_pairs = len(orig_raw) // 2

    # Preprocess
    print("Preprocessing frames...", flush=True)
    _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    KERNEL = (torch.outer(_r, _r) / (_r.sum()**2)).expand(3, 1, 9, 9).to(DEVICE)

    comp_model = torch.zeros(n_pairs, 3, MODEL_H, MODEL_W)
    orig_logits = torch.zeros(n_pairs, 5, MODEL_H, MODEL_W)
    orig_argmax = torch.zeros(n_pairs, MODEL_H, MODEL_W, dtype=torch.long)
    err_count = torch.zeros(n_pairs)

    with torch.inference_mode():
        for i in range(n_pairs):
            comp_model[i] = pipeline_to_model_res(comp_raw[2*i+1], KERNEL)
            o_t = pipeline_to_model_res(orig_raw[2*i+1], KERNEL)
            o_logits = segnet(o_t.unsqueeze(0).to(DEVICE))
            orig_logits[i] = o_logits[0].cpu()
            orig_argmax[i] = o_logits.argmax(dim=1)[0].cpu()

            c_logits = segnet(comp_model[i].unsqueeze(0).to(DEVICE))
            c_pred = c_logits.argmax(dim=1)[0].cpu()
            err_count[i] = (c_pred != orig_argmax[i]).sum().item()
            if i % 100 == 0:
                print(f"  {i}/{n_pairs}...", flush=True)

    baseline_errors = int(err_count.sum().item())
    baseline_seg = baseline_errors / (n_pairs * MODEL_H * MODEL_W)
    print(f"Baseline: {baseline_errors:,} errors, 100*seg={100*baseline_seg:.4f}", flush=True)

    # Dataset
    ds = FrameDataset(comp_model, orig_logits, orig_argmax, err_count)
    if args.hard_oversample:
        # Weight pairs by error count (more errors = more samples)
        weights = err_count + 1.0
        sampler = WeightedRandomSampler(weights.tolist(), len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            num_workers=0, pin_memory=True, drop_last=True)
        print(f"  Hard oversampling enabled")
    else:
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)

    # Model
    model = RENv3(ch=args.channels, qat=args.qat).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  RENv3: {n_params:,} params, ch={args.channels}, qat={args.qat}", flush=True)

    if args.resume:
        print(f"  Resuming from {args.resume}")
        sd = torch.load(args.resume, map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd, strict=True)

    # EMA
    ema_model = None
    if args.ema:
        ema_model = copy.deepcopy(model).eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        print(f"  EMA enabled (decay={args.ema_decay})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = ROOT / 'submissions' / 'evenframe_meta_v4_crf34' / 'archive'
    save_dir.mkdir(exist_ok=True)
    tag = args.tag if args.tag else f'_enh_ch{args.channels}'
    save_path = save_dir / f'ren_seg_v3{tag}.pt'
    save_path_ema = save_dir / f'ren_seg_v3{tag}_ema.pt'

    # Initial eval
    initial_errors = evaluate(model, comp_model, orig_argmax, segnet, n_pairs)
    initial_fixed = baseline_errors - initial_errors
    print(f"  Initial (resumed): fixed={initial_fixed:+d}, "
          f"100*seg={100*initial_errors/(n_pairs*MODEL_H*MODEL_W):.4f}\n", flush=True)
    best_fixed = initial_fixed
    best_ema_fixed = initial_fixed

    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Margin schedule
        if args.margin_anneal:
            progress = min(1.0, epoch / (args.epochs * 0.5))
            margin = 0.2 + (args.margin_target - 0.2) * progress
        else:
            margin = 0.2

        # LR cosine schedule
        lr = args.lr * (1 + math.cos(math.pi * epoch / args.epochs)) / 2 + 1e-5
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        epoch_loss = 0
        n_batches = 0

        for comp, o_logits, o_argmax in loader:
            comp = comp.to(DEVICE)
            o_logits = o_logits.to(DEVICE)
            o_argmax = o_argmax.to(DEVICE)

            # Input noise
            if args.input_noise > 0:
                noise = torch.randn_like(comp) * args.input_noise
                comp = (comp + noise).clamp(0, 255)

            optimizer.zero_grad()
            enhanced = model(comp)
            logits = segnet(enhanced)

            # Margin loss with top-K hard mining
            margin_map = margin_loss(logits, o_argmax, margin)
            B = margin_map.shape[0]
            k = min(5000, MODEL_H * MODEL_W)
            flat = margin_map.view(B, -1)
            topk_vals, _ = flat.topk(k, dim=1)
            margin_part = topk_vals.mean()

            if args.margin_only:
                loss = margin_part
            else:
                # Add KL against original logits
                T = 2.0
                kl = F.kl_div(
                    F.log_softmax(logits / T, dim=1),
                    F.softmax(o_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)
                loss = margin_part + 0.5 * kl

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update EMA
            if ema_model is not None:
                with torch.no_grad():
                    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                        ema_p.data.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)

            epoch_loss += loss.item()
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            new_errors = evaluate(model, comp_model, orig_argmax, segnet, n_pairs)
            total_fixed = baseline_errors - new_errors
            new_seg = new_errors / (n_pairs * MODEL_H * MODEL_W)

            marker = ''
            if total_fixed > best_fixed:
                best_fixed = total_fixed
                torch.save(model.state_dict(), save_path)
                marker = ' <- saved'
                epochs_no_improve = 0
            else:
                epochs_no_improve += 10

            ema_marker = ''
            if ema_model is not None:
                ema_errors = evaluate(ema_model, comp_model, orig_argmax, segnet, n_pairs)
                ema_fixed = baseline_errors - ema_errors
                if ema_fixed > best_ema_fixed:
                    best_ema_fixed = ema_fixed
                    torch.save(ema_model.state_dict(), save_path_ema)
                    ema_marker = f' EMA={ema_fixed:+d} <- saved'
                else:
                    ema_marker = f' EMA={ema_fixed:+d}'

            elapsed = time.time() - t0
            print(f"  Ep {epoch:4d}/{args.epochs}  loss={epoch_loss:.4f}  "
                  f"fixed={total_fixed:+d} ({total_fixed/n_pairs:+.0f}/fr)  "
                  f"seg={100*new_seg:.4f}  m={margin:.2f}  lr={lr:.1e}  "
                  f"[{elapsed:.0f}s]{marker}{ema_marker}", flush=True)

            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"  Early stop: no improvement for {epochs_no_improve} epochs")
                break

    # Final report
    final_seg = (baseline_errors - best_fixed) / (n_pairs * MODEL_H * MODEL_W)
    seg_gain = 100 * (baseline_seg - final_seg)
    initial_seg_gain = 100 * (baseline_seg - initial_errors / (n_pairs * MODEL_H * MODEL_W))

    print(f"\n{'='*60}")
    print(f"Initial gain (from resume): {initial_seg_gain:.4f}")
    print(f"Final gain:                 {seg_gain:.4f}")
    print(f"Improvement:                {seg_gain - initial_seg_gain:+.4f}")
    if ema_model is not None:
        ema_final_seg = (baseline_errors - best_ema_fixed) / (n_pairs * MODEL_H * MODEL_W)
        ema_gain = 100 * (baseline_seg - ema_final_seg)
        print(f"EMA gain:                   {ema_gain:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', type=str, required=True)
    ap.add_argument('--channels', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)  # lower than fresh training
    ap.add_argument('--tag', type=str, default='')
    ap.add_argument('--patience', type=int, default=200)

    # Enhancement toggles
    ap.add_argument('--qat', action='store_true', help='Quantization-aware training')
    ap.add_argument('--margin-anneal', action='store_true', help='Anneal margin from 0.2 to target')
    ap.add_argument('--margin-target', type=float, default=0.5)
    ap.add_argument('--hard-oversample', action='store_true', help='Sample by error count')
    ap.add_argument('--ema', action='store_true', help='Track EMA weights')
    ap.add_argument('--ema-decay', type=float, default=0.999)
    ap.add_argument('--input-noise', type=float, default=0.0, help='Gaussian noise sigma on input')
    ap.add_argument('--margin-only', action='store_true', help='Skip KL loss')

    train(ap.parse_args())
