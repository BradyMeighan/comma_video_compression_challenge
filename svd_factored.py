#!/usr/bin/env python
"""
Factored SVD fine-tuning: replace Conv2d with two smaller convs, train directly.

Instead of: Conv2d(C_in, C_out, k) with weight (C_out, C_in, k, k)
We use:     Conv2d(C_in, r, k, bias=False) → Conv2d(r, C_out, 1, bias=True)

Initialized from SVD of original teacher weights. Fine-tuned on 600 frames
+ trajectory data. The factors (U, V) are the only parameters stored.

Usage:
  python svd_factored.py --rank-ratio 0.2 --epochs 60
"""
import sys, os, time, math, argparse, bz2, pickle, copy
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss

DISTILL_DIR = Path('distill_data')
TRAJ_DIR = DISTILL_DIR / 'trajectory'
OUT_DIR = Path('compressed_models')
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
#  FACTORED CONV2D
# ═══════════════════════════════════════════════════════════════════════

class FactoredConv2d(nn.Module):
    """Two-conv replacement: spatial V conv → 1x1 U projection."""
    def __init__(self, in_channels, out_channels, kernel_size, rank,
                 stride=1, padding=0, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.conv_v = nn.Conv2d(in_channels, rank, kernel_size, stride, padding, bias=False)
        self.conv_u = nn.Conv2d(rank, out_channels, 1, 1, 0, bias=bias)
        self.rank = rank

    def forward(self, x):
        return self.conv_u(self.conv_v(x))


def factor_model(model, rank_ratio=0.3, min_channels=16, verbose=True):
    """Replace standard Conv2d layers with FactoredConv2d, initialized from SVD.

    Skips:
    - Depthwise convolutions (groups > 1)
    - Tiny layers (in_channels < min_channels)
    - 1x1 convolutions (no spatial kernel to factor, treated as linear)
    - The final segmentation head (must keep 5 output classes)
    """
    total_orig = 0
    total_factored = 0
    replaced = 0

    def _replace(parent, name, child):
        nonlocal total_orig, total_factored, replaced

        # Skip depthwise, tiny, or 1x1 convs
        if not isinstance(child, nn.Conv2d):
            return False
        if child.groups > 1:
            return False
        if child.in_channels < min_channels:
            return False
        k = child.kernel_size[0] * child.kernel_size[1]
        if k <= 1:
            # 1x1 conv: factor as matrix (no spatial kernel)
            return _replace_1x1(parent, name, child)
        if child.out_channels == 5:  # segmentation head
            return False

        W = child.weight.data
        C_out, C_in, kH, kW = W.shape
        W_flat = W.reshape(C_out, C_in * kH * kW)

        max_rank = min(C_out, C_in * kH * kW)
        rank = max(1, int(max_rank * rank_ratio))

        U, S, Vh = torch.linalg.svd(W_flat, full_matrices=False)
        U_r = U[:, :rank]                    # (C_out, rank)
        SV_r = torch.diag(S[:rank]) @ Vh[:rank, :]  # (rank, C_in*kH*kW)

        factored = FactoredConv2d(
            C_in, C_out, child.kernel_size, rank,
            child.stride, child.padding, child.bias is not None
        ).to(W.device)

        factored.conv_v.weight.data = SV_r.reshape(rank, C_in, kH, kW)
        factored.conv_u.weight.data = U_r.reshape(C_out, rank, 1, 1)
        if child.bias is not None:
            factored.conv_u.bias.data = child.bias.data.clone()

        setattr(parent, name, factored)

        orig_params = C_out * C_in * kH * kW
        new_params = rank * C_in * kH * kW + C_out * rank
        total_orig += orig_params
        total_factored += new_params
        replaced += 1

        if verbose and orig_params > 5000:
            energy = (S[:rank] ** 2).sum() / (S ** 2).sum()
            print(f"  {name:<30} ({C_out},{C_in},{kH},{kW}) rank={rank}/{max_rank} "
                  f"energy={energy*100:.1f}% {orig_params:,}->{new_params:,}")
        return True

    def _replace_1x1(parent, name, child):
        nonlocal total_orig, total_factored, replaced

        W = child.weight.data.squeeze()  # (C_out, C_in)
        if W.dim() != 2:
            return False

        C_out, C_in = W.shape
        if C_in < min_channels:
            return False

        max_rank = min(C_out, C_in)
        rank = max(1, int(max_rank * rank_ratio))

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        U_r = U[:, :rank]
        SV_r = torch.diag(S[:rank]) @ Vh[:rank, :]

        # Replace with two 1x1 convs
        factored = nn.Sequential(
            nn.Conv2d(C_in, rank, 1, child.stride, 0, bias=False),
            nn.Conv2d(rank, C_out, 1, 1, 0, bias=child.bias is not None),
        ).to(W.device)

        factored[0].weight.data = SV_r.reshape(rank, C_in, 1, 1)
        factored[1].weight.data = U_r.reshape(C_out, rank, 1, 1)
        if child.bias is not None:
            factored[1].bias.data = child.bias.data.clone()

        setattr(parent, name, factored)

        orig_params = C_out * C_in
        new_params = rank * C_in + C_out * rank
        total_orig += orig_params
        total_factored += new_params
        replaced += 1

        if verbose and orig_params > 5000:
            energy = (S[:rank] ** 2).sum() / (S ** 2).sum()
            print(f"  {name:<30} 1x1 ({C_out},{C_in}) rank={rank}/{max_rank} "
                  f"energy={energy*100:.1f}% {orig_params:,}->{new_params:,}")
        return True

    # Recursively replace
    def _recurse(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                _replace(module, name, child)
            else:
                _recurse(child)

    _recurse(model)

    if verbose:
        print(f"\n  Replaced {replaced} layers")
        print(f"  Params: {total_orig:,} -> {total_factored:,} ({total_factored/max(total_orig,1):.1%})")

    return total_orig, total_factored


def check_seg_accuracy(model, seg_inputs, teacher_argmax, device, bs=32):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(seg_inputs), bs):
            x = seg_inputs[i:i+bs].to(device)
            ta = teacher_argmax[i:i+bs].to(device)
            correct += (model(x).argmax(1) == ta).sum().item()
            total += ta.numel()
    return correct / total


# ═══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank-ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--skip-finetune', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load teacher and factor it ──
    print("=" * 70)
    print(f"STEP 1: FACTOR SEGNET (rank_ratio={args.rank_ratio:.0%})")
    print("=" * 70)
    print()

    model = SegNet().to(device)
    model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    before_params = sum(p.numel() for p in model.parameters())

    print("Replacing Conv2d layers with FactoredConv2d...")
    print()
    orig_p, factored_p = factor_model(model, rank_ratio=args.rank_ratio)
    after_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Total model params: {before_params:,} -> {after_params:,}")
    print(f"  FP32 size: {after_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  FP16 size: {after_params * 2 / 1024:.0f} KB")
    print(f"  INT8 size: {after_params / 1024:.0f} KB")
    print(f"  INT5 size: {after_params * 5 / 8 / 1024:.0f} KB (raw, before bz2)")
    print()

    # Check post-factoring accuracy
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)

    acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    print(f"  Post-factoring accuracy: {acc*100:.2f}% (will recover with fine-tuning)")
    print()

    if args.skip_finetune:
        torch.save(model.state_dict(), OUT_DIR / 'segnet_factored.pt')
        print(f"  Saved to {OUT_DIR / 'segnet_factored.pt'}")
        return

    # ── Step 2: Fine-tune in factored form ──
    print("=" * 70)
    print(f"STEP 2: FINE-TUNE FACTORED MODEL ({args.epochs} epochs)")
    print("=" * 70)
    print()

    # Load trajectory data
    traj_frames = traj_logits = traj_argmax = None
    N_traj = 0
    if TRAJ_DIR.exists():
        traj_frames = torch.load(TRAJ_DIR / 'traj_frames.pt', weights_only=True)
        traj_logits = torch.load(TRAJ_DIR / 'traj_logits.pt', weights_only=True)
        traj_argmax = traj_logits.float().argmax(1)
        N_traj = traj_frames.shape[0]
        print(f"  Trajectory data: {N_traj} frames")

    N = seg_inputs.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 0.01)
    T_kd = 6.0
    best_acc = 0.0
    best_state = None
    t0 = time.time()

    print(f"  {'Ep':>3} {'Loss':>8} {'Acc':>8} {'Best':>8} {'Time':>5}")
    print("  " + "-" * 40)

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0
        n_b = 0

        # Original frames
        perm = torch.randperm(N)
        for i in range(0, N, args.batch_size):
            idx = perm[i:i+args.batch_size]
            x = seg_inputs[idx].to(device)
            tl = seg_logits[idx].to(device)
            ta = teacher_argmax[idx].to(device)

            out = model(x)
            kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                          F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
            ce = F.cross_entropy(out, ta)
            mse = F.mse_loss(out, tl)
            loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        # Trajectory frames
        if traj_frames is not None:
            traj_perm = torch.randperm(N_traj)[:N]
            for i in range(0, len(traj_perm), args.batch_size):
                idx = traj_perm[i:i+args.batch_size]
                x = traj_frames[idx].float().to(device)
                tl = traj_logits[idx].float().to(device)
                ta = traj_argmax[idx].to(device)

                out = model(x)
                kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                              F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
                ce = F.cross_entropy(out, ta)
                mse = F.mse_loss(out, tl)
                loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                n_b += 1

        sched.step()

        model.eval()
        acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT_DIR / 'segnet_factored.pt')

        elapsed = time.time() - t0
        saved = "*" if acc >= best_acc else " "
        print(f" {saved}{epoch:3d} {ep_loss/n_b:8.1f} {acc*100:7.3f}% {best_acc*100:7.3f}% {elapsed:4.0f}s",
              flush=True)

    print(f"\n  Best accuracy: {best_acc*100:.3f}%")

    # ── Step 3: Compress ──
    print()
    print("=" * 70)
    print("STEP 3: COMPRESS FACTORED MODEL")
    print("=" * 70)
    print()

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    sd = model.state_dict()
    for bits in [5, 6, 8]:
        max_val = 2**(bits-1) - 1
        min_val = -2**(bits-1)
        packed = {}
        for name, tensor in sd.items():
            t = tensor.cpu().float()
            if t.dim() >= 2 and t.numel() > 10 and 'running' not in name and 'num_batches' not in name:
                out_ch = t.shape[0]
                flat = t.view(out_ch, -1)
                amax = flat.abs().amax(dim=1).clamp(min=1e-12)
                scales = amax / max_val
                q = (flat / scales.unsqueeze(1)).round().clamp(min_val, max_val)
                packed[name] = {'q': q.to(torch.int8).numpy(),
                                's': scales.half().numpy(), 'shape': list(t.shape)}
            else:
                packed[name] = {'f16': t.half().numpy() if t.numel() > 1 else t.numpy()}

        compressed = bz2.compress(pickle.dumps(packed), 9)
        path = OUT_DIR / f'segnet_factored_int{bits}.bin'
        with open(path, 'wb') as f:
            f.write(compressed)
        print(f"  INT{bits}: {len(compressed)/1024:.0f} KB -> {path}")

    # ── Step 4: Quick adversarial decode test ──
    print()
    print("=" * 70)
    print("STEP 4: ADVERSARIAL DECODE TRANSFER TEST")
    print("=" * 70)
    print()

    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    seg_logits_all = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    ta_all = seg_logits_all.argmax(1).to(device)
    pose_all = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    del seg_logits_all

    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    iters = 150
    starts = [0, 100, 200, 300, 450]
    results = {'ts': [], 'tp': []}

    for bi, st in enumerate(starts):
        end = min(st + bs, 600)
        tgt_s = ta_all[st:end]
        tgt_p = pose_all[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt_adv = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters - 1, 1))) for i in range(iters)]

        print(f"  Batch {bi+1}/5 (pairs {st}-{end-1}): ", end='', flush=True)
        for it in range(iters):
            for pg in opt_adv.param_groups: pg['lr'] = lr_s[it]
            opt_adv.zero_grad(set_to_none=True)
            progress = it / max(iters - 1, 1)
            seg_l = margin_loss(model(f1), tgt_s, 0.1 if progress < 0.5 else 5.0)
            if progress >= 0.3:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:, :6], tgt_p)
                total = 120.0 * seg_l + 0.2 * pose_l
            else:
                total = 120.0 * seg_l
            total.backward()
            opt_adv.step()
            with torch.no_grad(): f0.data.clamp_(0, 255); f1.data.clamp_(0, 255)
            if (it + 1) % 50 == 0: print(f"{it+1}", end=' ', flush=True)
        print("done", flush=True)

        with torch.no_grad():
            f1u = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                align_corners=False).clamp(0, 255).round().byte().float()
            f0u = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                align_corners=False).clamp(0, 255).round().byte().float()
            ts_in = F.interpolate(f1u, (mH, mW), mode='bilinear')
            ts = (t_seg(ts_in).argmax(1) != tgt_s).float().mean((1, 2))
            results['ts'].extend(ts.cpu().tolist())
            tp_pair = F.interpolate(
                torch.stack([f0u, f1u], 1).reshape(-1, 3, Hc, Wc),
                (mH, mW), mode='bilinear').reshape(B, 2, 3, mH, mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:, :6]
            results['tp'].extend((tpo - tgt_p).pow(2).mean(1).cpu().tolist())

        print(f"    Teacher: seg={np.mean(results['ts'][-B:]):.6f} "
              f"pose={np.mean(results['tp'][-B:]):.6f}")
        del f0, f1, opt_adv
        torch.cuda.empty_cache()

    avg_ts = np.mean(results['ts'])
    avg_tp = np.mean(results['tp'])
    s_seg = 100 * avg_ts
    s_pose = math.sqrt(10 * avg_tp)
    dist = s_seg + s_pose

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  seg_dist={avg_ts:.6f} ({(1-avg_ts)*100:.2f}% agreement)")
    print(f"  pose_mse={avg_tp:.6f}")
    print(f"  100*seg={s_seg:.4f}  sqrt(10*p)={s_pose:.4f}  distortion={dist:.4f}")
    print()

    for bits in [5, 6, 8]:
        bp = OUT_DIR / f'segnet_factored_int{bits}.bin'
        if bp.exists():
            seg_kb = os.path.getsize(bp) / 1024
            archive_kb = seg_kb + 300  # targets
            rate = (archive_kb * 1024) / 37_545_489
            total = dist + 25 * rate
            print(f"  INT{bits}: {seg_kb:.0f}KB  25*rate={25*rate:.3f}  TOTAL={total:.3f}")

    print(f"\n  Leader: 1.95 | Baseline: 4.39")


if __name__ == '__main__':
    main()
