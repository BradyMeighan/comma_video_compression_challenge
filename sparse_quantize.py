#!/usr/bin/env python
"""
Sparse fine-tuning + extreme quantization of teacher SegNet & PoseNet.

Strategy:
1. Start from the exact teacher weights (perfect accuracy)
2. Fine-tune with L1 penalty on weights → drives individual weights to zero
3. Gradually increase L1 pressure while monitoring accuracy
4. Hard-threshold near-zero weights to exactly zero
5. Quantize remaining non-zero weights to INT4/INT5
6. Compress with bz2 (zeros compress to nearly nothing)

The model architecture stays IDENTICAL to the teacher. Same SegNet, same PoseNet.
Just many weights are zero and the rest are low-bit quantized.
At inflate time: decompress → dequantize → load into SegNet()/PoseNet() → adversarial decode.

Usage:
  python sparse_quantize.py                    # Full pipeline
  python sparse_quantize.py --seg-only         # Just SegNet
  python sparse_quantize.py --lambda-l1 1e-4   # Adjust sparsity pressure
"""
import sys, os, time, math, argparse, bz2, pickle
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss

DISTILL_DIR = Path('distill_data')
OUT_DIR = Path('compressed_models')


def count_sparsity(model):
    """Count fraction of exactly-zero weights in conv/linear layers."""
    total = zeros = 0
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            total += p.numel()
            zeros += (p.data == 0).sum().item()
    return zeros / total if total > 0 else 0


def apply_l1_grad(model, lambda_l1):
    """Add L1 gradient manually (subgradient: sign of weight)."""
    for name, p in model.named_parameters():
        if p.dim() >= 2 and p.grad is not None:
            p.grad.data.add_(lambda_l1 * p.data.sign())


def hard_threshold(model, threshold):
    """Set weights with |w| < threshold to exactly zero."""
    zeroed = total = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:
                mask = p.data.abs() < threshold
                p.data[mask] = 0.0
                zeroed += mask.sum().item()
                total += p.numel()
    return zeroed / total if total > 0 else 0


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


def check_pose_mse(model, pose_inputs, pose_outputs, device, bs=32):
    model.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(pose_inputs), bs):
            x = pose_inputs[i:i+bs].to(device)
            t = pose_outputs[i:i+bs].to(device)
            pred = model(x)['pose'][:, :6]
            total_mse += (pred - t).pow(2).mean(1).sum().item()
            n += len(x)
    return total_mse / n


def quantize_and_compress(state_dict, bits=4):
    """Quantize weights to N bits per-channel, keep BN/bias as FP16, compress."""
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    packed = {}

    for name, tensor in state_dict.items():
        t = tensor.cpu().float()
        is_weight = (t.dim() >= 2 and 'running' not in name
                     and 'bn' not in name.lower() and 'norm' not in name.lower()
                     and t.numel() > 10)

        if is_weight:
            out_ch = t.shape[0]
            flat = t.view(out_ch, -1)
            amax = flat.abs().amax(dim=1).clamp(min=1e-12)
            scales = amax / max_val
            q = (flat / scales.unsqueeze(1)).round().clamp(min_val, max_val)
            packed[name] = {
                'q': q.to(torch.int8).numpy(),
                's': scales.half().numpy(),
                'shape': list(t.shape),
            }
        else:
            packed[name] = {'f16': t.half().numpy()}

    raw = pickle.dumps(packed)
    compressed = bz2.compress(raw, 9)
    return compressed


def dequantize_compressed(compressed_bytes, device):
    """Decompress + dequantize back to FP32 state dict."""
    packed = pickle.loads(bz2.decompress(compressed_bytes))
    sd = {}
    for name, entry in packed.items():
        if 'q' in entry:
            q = torch.from_numpy(entry['q']).float()
            scales = torch.from_numpy(entry['s']).float()
            shape = entry['shape']
            flat = q * scales.unsqueeze(1)
            sd[name] = flat.view(shape).to(device)
        else:
            sd[name] = torch.from_numpy(entry['f16']).float().to(device)
    return sd


# ─── Sparse fine-tuning for SegNet ──────────────────────────────────────────

def sparse_finetune_segnet(device, epochs=80, lambda_l1_start=1e-5,
                            lambda_l1_end=5e-4, target_sparsity=0.90,
                            lr=1e-4, use_trajectory=True):
    print("=" * 70)
    print("SEGNET SPARSE FINE-TUNING" + (" + TRAJECTORY" if use_trajectory else ""))
    print("=" * 70)
    print()
    print(f"Strategy: start from teacher, add L1 penalty to drive weights to zero.")
    print(f"  L1 ramps from {lambda_l1_start} to {lambda_l1_end} over {epochs} epochs.")
    print(f"  Target sparsity: {target_sparsity:.0%} of weights = 0")
    if use_trajectory:
        print(f"  Training on BOTH original frames AND trajectory data.")
        print(f"  This preserves gradient quality on adversarial decode inputs.")
    print()

    # Load teacher (the model we're sparsifying)
    print("Loading teacher SegNet...")
    model = SegNet().to(device)
    model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    n_params = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    print(f"  {n_params:,} weight params")

    # Frozen reference teacher for KDIGA (gradient alignment)
    print("Loading frozen reference teacher for KDIGA...")
    ref_teacher = SegNet().eval().to(device)
    ref_teacher.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    for p in ref_teacher.parameters():
        p.requires_grad_(False)
    print("  KDIGA: will align input gradients with reference teacher")

    # Load original data
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    N_orig = seg_inputs.shape[0]

    # Load trajectory data if available
    traj_frames = traj_logits = traj_argmax = None
    N_traj = 0
    if use_trajectory:
        traj_path = DISTILL_DIR / 'trajectory' / 'traj_frames.pt'
        if traj_path.exists():
            traj_frames = torch.load(traj_path, weights_only=True)  # half
            traj_logits = torch.load(DISTILL_DIR / 'trajectory' / 'traj_logits.pt', weights_only=True)
            traj_argmax = traj_logits.float().argmax(1)
            N_traj = traj_frames.shape[0]
            print(f"  Loaded {N_traj} trajectory frames")
        else:
            print(f"  WARNING: No trajectory data found at {traj_path}")
            print(f"  Run: python train_onpolicy.py --phase 0")
    print(f"  Total training data: {N_orig} original + {N_traj} trajectory")

    # Baseline
    acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    sparsity = count_sparsity(model)
    print(f"  Baseline: acc={acc*100:.3f}%, sparsity={sparsity:.1%}")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr*0.01)
    T_kd = 6.0
    batch_size = 8
    kdiga_weight = 0.1  # weight for gradient alignment loss
    kdiga_every = 5  # run KDIGA every Nth batch (create_graph=True triples VRAM)
    best_acc = acc
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    # Weight masks: True = keep, False = permanently frozen at 0
    masks = {}
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            masks[name] = torch.ones_like(p.data, dtype=torch.bool)

    print(f"  {'Ep':>3} {'Loss':>8} {'L1':>9} {'Acc':>8} {'Sparsity':>9} {'Size4':>7} {'Size5':>7} {'Time':>5}")
    print("  " + "-" * 70)

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(epochs - 1, 1)
        lambda_l1 = lambda_l1_start + (lambda_l1_end - lambda_l1_start) * progress

        ep_loss = 0.0
        n_b = 0

        # ── Train on both original + trajectory frames with KDIGA ──
        # Build combined batches: alternate original and trajectory
        perm_orig = torch.randperm(N_orig)
        if traj_frames is not None:
            perm_traj = torch.randperm(N_traj)
            traj_per_epoch = min(N_traj, N_orig)
        else:
            traj_per_epoch = 0

        # Run KDIGA every 3rd batch (expensive: needs 2 forward + 2 backward)
        use_kdiga = (epoch >= 5)  # skip KDIGA for first 5 epochs (let model stabilize)

        # Original frames
        for i in range(0, N_orig, batch_size):
            idx = perm_orig[i:i+batch_size]
            x = seg_inputs[idx].to(device)
            tl = seg_logits[idx].to(device)
            ta = teacher_argmax[idx].to(device)

            out = model(x)
            kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                          F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
            ce = F.cross_entropy(out, ta)
            mse = F.mse_loss(out, tl)
            loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            # KDIGA: align input gradients with reference teacher (every 3rd batch)
            if use_kdiga and n_b % kdiga_every == 0:
                x_grad = x[:2].detach().requires_grad_(True)  # tiny sub-batch for VRAM
                s_out = model(x_grad)
                s_loss = F.cross_entropy(s_out, ta[:2])
                s_grad = torch.autograd.grad(s_loss, x_grad, create_graph=True)[0]
                t_out = ref_teacher(x_grad)
                t_loss = F.cross_entropy(t_out, ta[:4])
                t_grad = torch.autograd.grad(t_loss, x_grad)[0]
                grad_loss = F.mse_loss(s_grad, t_grad.detach())
                loss = loss + kdiga_weight * grad_loss

            opt.zero_grad()
            loss.backward()
            apply_l1_grad(model, lambda_l1)
            opt.step()

            with torch.no_grad():
                for name, p in model.named_parameters():
                    if name in masks:
                        p.data.mul_(masks[name])

            ep_loss += loss.item()
            n_b += 1

        # Trajectory frames
        if traj_frames is not None:
            for i in range(0, traj_per_epoch, batch_size):
                idx = perm_traj[i:i+batch_size]
                x = traj_frames[idx].float().to(device)
                tl = traj_logits[idx].float().to(device)
                ta = traj_argmax[idx].to(device)

                out = model(x)
                kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                              F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
                ce = F.cross_entropy(out, ta)
                mse = F.mse_loss(out, tl)
                loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

                # No KDIGA on trajectory frames (saves VRAM, originals are enough)
                opt.zero_grad()
                loss.backward()
                apply_l1_grad(model, lambda_l1)
                opt.step()

                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if name in masks:
                            p.data.mul_(masks[name])

                ep_loss += loss.item()
                n_b += 1

        sched.step()

        # Hard-threshold tiny weights every 5 epochs AND freeze them permanently
        if (epoch + 1) % 5 == 0:
            all_weights = []
            for name, p in model.named_parameters():
                if p.dim() >= 2:
                    all_weights.append(p.data.abs().flatten())
            all_w = torch.cat(all_weights)
            current_target = min(target_sparsity, 0.5 + 0.5 * progress)
            thresh_val = torch.quantile(all_w, current_target).item()
            hard_threshold(model, max(thresh_val, 1e-7))
            # Update masks to PERMANENTLY freeze zeroed weights
            for name, p in model.named_parameters():
                if p.dim() >= 2 and name in masks:
                    masks[name] = masks[name] & (p.data != 0)  # can only go 0->0, never 0->1

        # Evaluate
        model.eval()
        acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
        sparsity = count_sparsity(model)

        # Quick compressed size estimate
        sd = {k: v for k, v in model.state_dict().items()}
        size4 = len(quantize_and_compress(sd, bits=4))
        size5 = len(quantize_and_compress(sd, bits=5))

        if acc > best_acc * 0.999 and sparsity > count_sparsity(
                nn.Module()):  # save if acc is close to best AND more sparse
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Always save the most recent good checkpoint
        if acc > 0.99:
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_acc = acc

        elapsed = time.time() - t0
        print(f"  {epoch:3d} {ep_loss/n_b:8.1f} {lambda_l1:.2e} {acc*100:7.3f}% "
              f"{sparsity:8.1%} {size4/1024:6.0f}KB {size5/1024:6.0f}KB {elapsed:4.0f}s",
              flush=True)

    # Final results
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    final_acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    final_sparsity = count_sparsity(model)

    print()
    print(f"  Final: acc={final_acc*100:.3f}%, sparsity={final_sparsity:.1%}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for bits in [4, 5, 6]:
        compressed = quantize_and_compress(model.state_dict(), bits=bits)
        path = OUT_DIR / f'segnet_sparse_int{bits}.bin'
        with open(path, 'wb') as f:
            f.write(compressed)
        print(f"  INT{bits}: {len(compressed)/1024:.0f} KB → {path}")

    torch.save(best_state, OUT_DIR / 'segnet_sparse_fp32.pt')
    del seg_inputs, seg_logits, ref_teacher
    if traj_frames is not None:
        del traj_frames, traj_logits
    torch.cuda.empty_cache()
    return model


# ─── Sparse fine-tuning for PoseNet ─────────────────────────────────────────

def sparse_finetune_posenet(device, epochs=60, lambda_l1_start=1e-5,
                             lambda_l1_end=3e-4, target_sparsity=0.90,
                             lr=5e-5):
    print()
    print("=" * 70)
    print("POSENET SPARSE FINE-TUNING")
    print("=" * 70)
    print()

    model = PoseNet().to(device)
    model.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    n_params = sum(p.numel() for p in model.parameters() if p.dim() >= 2)
    print(f"  {n_params:,} weight params")

    pose_inputs = torch.load(DISTILL_DIR / 'pose_inputs.pt', weights_only=True)
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True)
    N = pose_inputs.shape[0]

    mse = check_pose_mse(model, pose_inputs, pose_outputs, device)
    sparsity = count_sparsity(model)
    print(f"  Baseline: mse={mse:.8f}, sparsity={sparsity:.1%}")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr*0.01)
    batch_size = 8
    best_mse = mse
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    t0 = time.time()

    print(f"  {'Ep':>3} {'Loss':>10} {'L1':>9} {'MSE':>12} {'Sparsity':>9} {'Size4':>7} {'Time':>5}")
    print("  " + "-" * 65)

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(epochs - 1, 1)
        lambda_l1 = lambda_l1_start + (lambda_l1_end - lambda_l1_start) * progress

        perm = torch.randperm(N)
        ep_loss = 0.0
        n_b = 0

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = pose_inputs[idx].to(device)
            t = pose_outputs[idx].to(device)

            pred = model(x)['pose'][:, :6]
            loss = F.mse_loss(pred, t)

            opt.zero_grad()
            loss.backward()
            apply_l1_grad(model, lambda_l1)
            opt.step()

            ep_loss += loss.item()
            n_b += 1

        sched.step()

        if (epoch + 1) % 5 == 0:
            all_weights = []
            for name, p in model.named_parameters():
                if p.dim() >= 2:
                    all_weights.append(p.data.abs().flatten())
            all_w = torch.cat(all_weights)
            current_target = min(target_sparsity, 0.5 + 0.5 * progress)
            thresh_val = torch.quantile(all_w, current_target).item()
            hard_threshold(model, max(thresh_val, 1e-7))

        model.eval()
        mse = check_pose_mse(model, pose_inputs, pose_outputs, device)
        sparsity = count_sparsity(model)

        sd = model.state_dict()
        size4 = len(quantize_and_compress(sd, bits=4))

        if mse < best_mse * 1.5:  # allow some MSE increase for sparsity
            best_state = {k: v.cpu().clone() for k, v in sd.items()}
            best_mse = min(best_mse, mse)

        elapsed = time.time() - t0
        print(f"  {epoch:3d} {ep_loss/n_b:10.6f} {lambda_l1:.2e} {mse:12.8f} "
              f"{sparsity:8.1%} {size4/1024:6.0f}KB {elapsed:4.0f}s", flush=True)

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    final_mse = check_pose_mse(model, pose_inputs, pose_outputs, device)
    final_sparsity = count_sparsity(model)
    print(f"\n  Final: mse={final_mse:.8f}, sparsity={final_sparsity:.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bits in [4, 5, 6]:
        compressed = quantize_and_compress(model.state_dict(), bits=bits)
        path = OUT_DIR / f'posenet_sparse_int{bits}.bin'
        with open(path, 'wb') as f:
            f.write(compressed)
        print(f"  INT{bits}: {len(compressed)/1024:.0f} KB -> {path}")

    torch.save(best_state, OUT_DIR / 'posenet_sparse_fp32.pt')
    del pose_inputs, pose_outputs
    torch.cuda.empty_cache()
    return model


# ─── Summary ─────────────────────────────────────────────────────────────────

def print_summary():
    print()
    print("=" * 70)
    print("COMPRESSED MODEL SIZES")
    print("=" * 70)
    print()

    target_size = 300  # KB estimate for targets
    for bits in [4, 5, 6]:
        seg_path = OUT_DIR / f'segnet_sparse_int{bits}.bin'
        pose_path = OUT_DIR / f'posenet_sparse_int{bits}.bin'
        if seg_path.exists() and pose_path.exists():
            seg_kb = os.path.getsize(seg_path) / 1024
            pose_kb = os.path.getsize(pose_path) / 1024
            total_kb = seg_kb + pose_kb + target_size
            rate = (total_kb * 1024) / 37_545_489
            print(f"  INT{bits}: seg={seg_kb:.0f}KB + pose={pose_kb:.0f}KB + "
                  f"tgt={target_size}KB = {total_kb:.0f}KB")
            print(f"        25*rate = {25*rate:.3f}")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg-only', action='store_true')
    parser.add_argument('--pose-only', action='store_true')
    parser.add_argument('--seg-epochs', type=int, default=80)
    parser.add_argument('--pose-epochs', type=int, default=60)
    parser.add_argument('--seg-l1-end', type=float, default=5e-4)
    parser.add_argument('--pose-l1-end', type=float, default=3e-4)
    parser.add_argument('--target-sparsity', type=float, default=0.90)
    parser.add_argument('--use-trajectory', action='store_true', default=True,
                        help='Train on trajectory data too (preserves gradient quality)')
    parser.add_argument('--no-trajectory', action='store_false', dest='use_trajectory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    if not args.pose_only:
        sparse_finetune_segnet(device, epochs=args.seg_epochs,
                               lambda_l1_end=args.seg_l1_end,
                               target_sparsity=args.target_sparsity,
                               use_trajectory=args.use_trajectory)

    if not args.seg_only:
        sparse_finetune_posenet(device, epochs=args.pose_epochs,
                                lambda_l1_end=args.pose_l1_end,
                                target_sparsity=args.target_sparsity)

    print_summary()


if __name__ == '__main__':
    main()
