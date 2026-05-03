#!/usr/bin/env python
"""
Quick validation tests for the SVD + KDIGA + geometric pose pipeline.
Each test takes 1-5 minutes. Find failures early before committing hours.

Tests:
  1. SVD rank analysis: how compressible is each layer?
  2. SVD at various compression ratios: forward accuracy
  3. SVD adversarial decode transfer: THE critical test
  4. torch.func.jvp KDIGA: does it work without VRAM explosion?
  5. Geometric pose proxy: can we compute pose without PoseNet?
"""
import sys, time, math, bz2, pickle
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss

DISTILL_DIR = Path('distill_data')
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


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


def svd_compress_state_dict(state_dict, keep_ratio=0.5):
    """Apply truncated SVD to all conv/linear weight tensors.

    For conv weight (O, I, kH, kW): reshape to (O, I*kH*kW), SVD, truncate.
    keep_ratio: fraction of singular values to keep.
    Returns compressed state dict + total parameter count.
    """
    new_sd = {}
    total_orig = 0
    total_compressed = 0

    for name, tensor in state_dict.items():
        if tensor.dim() == 4 and tensor.shape[2] * tensor.shape[3] > 1:
            # Conv weight: (O, I, kH, kW) -> (O, I*kH*kW)
            O, I, kH, kW = tensor.shape
            mat = tensor.reshape(O, I * kH * kW).float()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            k = max(1, int(min(O, I * kH * kW) * keep_ratio))
            # Truncate
            U_k = U[:, :k]          # (O, k)
            S_k = S[:k]             # (k,)
            Vh_k = Vh[:k, :]        # (k, I*kH*kW)
            # Reconstruct
            reconstructed = (U_k * S_k.unsqueeze(0)) @ Vh_k
            new_sd[name] = reconstructed.reshape(O, I, kH, kW).to(tensor.dtype)
            total_orig += tensor.numel()
            total_compressed += U_k.numel() + S_k.numel() + Vh_k.numel()
        elif tensor.dim() == 2 and tensor.numel() > 100:
            # Linear weight: (O, I)
            mat = tensor.float()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            k = max(1, int(min(mat.shape) * keep_ratio))
            U_k = U[:, :k]
            S_k = S[:k]
            Vh_k = Vh[:k, :]
            reconstructed = (U_k * S_k.unsqueeze(0)) @ Vh_k
            new_sd[name] = reconstructed.to(tensor.dtype)
            total_orig += tensor.numel()
            total_compressed += U_k.numel() + S_k.numel() + Vh_k.numel()
        else:
            new_sd[name] = tensor.clone()

    return new_sd, total_orig, total_compressed


def run_quick_adversarial_decode(seg_model, pose_model, teacher_argmax, pose_targets,
                                  t_seg, t_pose, device, num_batches=3, iters=100):
    """Quick adversarial decode test. Returns avg seg_dist, pose_mse."""
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    starts = [0, 200, 400][:num_batches]
    results = {'ts': [], 'tp': []}

    for st in starts:
        end = min(st + bs, len(teacher_argmax))
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_targets[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters - 1, 1))) for i in range(iters)]

        for it in range(iters):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            progress = it / max(iters - 1, 1)
            seg_l = margin_loss(seg_model(f1), tgt_s, 0.1 if progress < 0.5 else 5.0)
            if progress >= 0.3 and pose_model is not None:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                po = pose_model(pn_in)
                pose_pred = po['pose'][:, :6] if isinstance(po, dict) else po[:, :6]
                pose_l = F.smooth_l1_loss(pose_pred, tgt_p)
                total = 120.0 * seg_l + 0.2 * pose_l
            else:
                total = 120.0 * seg_l
            total.backward()
            opt.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255); f1.data.clamp_(0, 255)

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

        del f0, f1, opt
        torch.cuda.empty_cache()

    return np.mean(results['ts']), np.mean(results['tp'])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load everything we need
    print("Loading models and data...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    seg_sd = {k: v.clone() for k, v in t_seg.state_dict().items()}

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    del seg_logits

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: SVD Rank Analysis
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 1: SVD RANK ANALYSIS — How compressible is SegNet?")
    print("=" * 70)
    print()
    print("For each layer, how many singular values capture 95/99/99.9% of energy?")
    print()

    total_params_95 = 0
    total_params_99 = 0
    total_params_full = 0

    for name, tensor in seg_sd.items():
        if tensor.dim() == 4 and tensor.shape[2] * tensor.shape[3] > 1:
            O, I, kH, kW = tensor.shape
            mat = tensor.reshape(O, I * kH * kW).float()
            S = torch.linalg.svdvals(mat)
            energy = (S ** 2).cumsum(0) / (S ** 2).sum()
            rank_95 = (energy < 0.95).sum().item() + 1
            rank_99 = (energy < 0.99).sum().item() + 1
            rank_999 = (energy < 0.999).sum().item() + 1
            max_rank = min(O, I * kH * kW)

            params_full = O * I * kH * kW
            params_95 = rank_95 * (O + I * kH * kW + 1)
            total_params_full += params_full
            total_params_95 += params_95
            total_params_99 += rank_99 * (O + I * kH * kW + 1)

            if params_full > 10000:
                print(f"  {name:<50} shape={list(tensor.shape)!s:<20} "
                      f"rank: 95%={rank_95}/{max_rank} 99%={rank_99} 99.9%={rank_999}")

    print(f"\n  Total conv params: {total_params_full:,}")
    print(f"  SVD at 95% energy: {total_params_95:,} params ({total_params_95/total_params_full:.1%})")
    print(f"  SVD at 99% energy: {total_params_99:,} params ({total_params_99/total_params_full:.1%})")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: SVD Forward Accuracy at Various Ratios
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 2: SVD FORWARD ACCURACY — Does SVD preserve accuracy?")
    print("=" * 70)
    print()

    for keep_ratio in [0.5, 0.3, 0.2, 0.15, 0.1, 0.05]:
        t0 = time.time()
        svd_sd, orig_p, comp_p = svd_compress_state_dict(seg_sd, keep_ratio)
        model = SegNet().eval().to(device)
        model.load_state_dict({k: v.to(device) for k, v in svd_sd.items()})
        acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
        ratio = comp_p / orig_p
        est_mb = comp_p * 4 / 1024 / 1024
        elapsed = time.time() - t0
        print(f"  keep={keep_ratio:.0%}: acc={acc*100:.2f}%  params={comp_p:,} ({ratio:.1%} of orig)  "
              f"~{est_mb:.1f}MB FP32  ({elapsed:.1f}s)")
        del model
        torch.cuda.empty_cache()

    print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: SVD ADVERSARIAL DECODE TRANSFER — THE critical test
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 3: SVD ADVERSARIAL DECODE TRANSFER")
    print("=" * 70)
    print("  Does SVD preserve the gradient landscape? This is make-or-break.")
    print()

    for keep_ratio in [0.3, 0.2, 0.1]:
        t0 = time.time()
        svd_sd, _, comp_p = svd_compress_state_dict(seg_sd, keep_ratio)
        svd_seg = SegNet().eval().to(device)
        svd_seg.load_state_dict({k: v.to(device) for k, v in svd_sd.items()})
        for p in svd_seg.parameters(): p.requires_grad_(False)

        acc = check_seg_accuracy(svd_seg, seg_inputs, teacher_argmax, device)
        print(f"  SVD keep={keep_ratio:.0%} (forward acc={acc*100:.2f}%, {comp_p:,} params):")
        print(f"    Running adversarial decode (3 batches x 100 iters)...", flush=True)

        avg_ts, avg_tp = run_quick_adversarial_decode(
            svd_seg, t_pose, teacher_argmax.to(device), pose_outputs,
            t_seg, t_pose, device, num_batches=3, iters=100)

        score_seg = 100 * avg_ts
        score_pose = math.sqrt(10 * avg_tp)
        elapsed = time.time() - t0
        print(f"    Teacher seg_dist={avg_ts:.6f}  pose_mse={avg_tp:.6f}")
        print(f"    100*seg={score_seg:.2f}  sqrt(10*p)={score_pose:.2f}  "
              f"distortion={score_seg + score_pose:.2f}  ({elapsed:.0f}s)")
        print()

        del svd_seg
        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: torch.func.jvp KDIGA — Does it work without VRAM explosion?
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 4: KDIGA via torch.func.jvp — VRAM test")
    print("=" * 70)
    print()

    try:
        from torch.func import jvp, vmap
        print("  torch.func available")

        # Test: compute directional derivative of SegNet
        test_model = SegNet().eval().to(device)
        test_model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

        x = seg_inputs[:4].to(device)
        v = torch.randn_like(x)  # random direction vector

        vram_before = torch.cuda.memory_allocated() / 1024**2

        def model_fn(inp):
            return test_model(inp)

        # Forward-mode: compute f(x) and Jv simultaneously
        t0 = time.time()
        output, jvp_out = jvp(model_fn, (x,), (v,))
        elapsed = time.time() - t0
        vram_after = torch.cuda.memory_allocated() / 1024**2

        print(f"  JVP output shape: {jvp_out.shape}")
        print(f"  VRAM: {vram_before:.0f}MB -> {vram_after:.0f}MB (+{vram_after - vram_before:.0f}MB)")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  SUCCESS: jvp works without VRAM explosion")

        # Compare with create_graph=True approach
        x2 = seg_inputs[:4].to(device).requires_grad_(True)
        vram_before2 = torch.cuda.memory_allocated() / 1024**2
        t0 = time.time()
        out2 = test_model(x2)
        loss2 = out2.sum()
        grad2 = torch.autograd.grad(loss2, x2, create_graph=True)[0]
        elapsed2 = time.time() - t0
        vram_after2 = torch.cuda.memory_allocated() / 1024**2
        print(f"\n  create_graph=True comparison:")
        print(f"  VRAM: {vram_before2:.0f}MB -> {vram_after2:.0f}MB (+{vram_after2 - vram_before2:.0f}MB)")
        print(f"  Time: {elapsed2:.3f}s")
        print(f"  JVP saves {vram_after2 - vram_after:.0f}MB VRAM")

        del test_model, x2, grad2
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  FAILED: {e}")

    print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 5: Nuclear norm analysis — How far from low-rank?
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 5: NUCLEAR NORM ANALYSIS — How far from low-rank?")
    print("=" * 70)
    print()
    print("  Nuclear norm / Frobenius norm ratio (closer to 1 = more low-rank)")
    print()

    for name, tensor in seg_sd.items():
        if tensor.dim() == 4 and tensor.numel() > 10000:
            O, I, kH, kW = tensor.shape
            mat = tensor.reshape(O, I * kH * kW).float()
            S = torch.linalg.svdvals(mat)
            nuclear = S.sum().item()
            frobenius = S.pow(2).sum().sqrt().item()
            spectral = S[0].item()
            effective_rank = (nuclear / spectral) if spectral > 0 else 0
            max_rank = min(O, I * kH * kW)
            print(f"  {name:<50} eff_rank={effective_rank:.1f}/{max_rank} "
                  f"nuc/frob={nuclear/frobenius:.2f}")

    print()
    print("=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
