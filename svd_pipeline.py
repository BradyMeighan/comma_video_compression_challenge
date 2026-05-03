#!/usr/bin/env python
"""
SVD compression pipeline for SegNet.

Phase 1: Nuclear norm prehab — steer weights toward low-rank geometry
Phase 2: SVD decomposition — truncate to target rank
Phase 3: Fine-tune with KDIGA — recover accuracy + preserve gradients
Phase 4: Quantize + compress — INT4/INT5 + entropy coding
Phase 5: Adversarial decode transfer test

Usage:
  python svd_pipeline.py --phase all
  python svd_pipeline.py --phase prehab --epochs 30
  python svd_pipeline.py --phase svd --keep-ratio 0.3
  python svd_pipeline.py --phase finetune --epochs 60
  python svd_pipeline.py --phase quantize
  python svd_pipeline.py --phase validate
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
TRAJ_DIR = DISTILL_DIR / 'trajectory'
OUT_DIR = Path('compressed_models')


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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def nuclear_norm_penalty(model):
    """Sum of nuclear norms of all conv/linear weight matrices."""
    penalty = 0.0
    for name, p in model.named_parameters():
        if p.dim() == 4 and p.numel() > 100:
            O = p.shape[0]
            mat = p.reshape(O, -1)
            # Nuclear norm = sum of singular values
            penalty += torch.linalg.svdvals(mat).sum()
        elif p.dim() == 2 and p.numel() > 100:
            penalty += torch.linalg.svdvals(p).sum()
    return penalty


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 1: NUCLEAR NORM PREHAB
# ═══════════════════════════════════════════════════════════════════════

def phase_prehab(device, epochs=30, lr=5e-5, nuc_weight_start=1e-4, nuc_weight_end=5e-2):
    """Fine-tune teacher with nuclear norm regularization to steer toward low-rank."""
    print("=" * 70)
    print("PHASE 1: NUCLEAR NORM PREHAB")
    print("=" * 70)
    print()
    print("Steering teacher weights toward low-rank geometry before SVD.")
    print("This prevents 'surgical shock' when we truncate singular values.")
    print(f"Nuclear norm weight ramps from {nuc_weight_start} to {nuc_weight_end}")
    print()

    model = SegNet().to(device)
    model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    N = seg_inputs.shape[0]

    # Also load trajectory data
    traj_frames = traj_logits = None
    if TRAJ_DIR.exists():
        traj_frames = torch.load(TRAJ_DIR / 'traj_frames.pt', weights_only=True)
        traj_logits = torch.load(TRAJ_DIR / 'traj_logits.pt', weights_only=True)
        print(f"  Loaded {traj_frames.shape[0]} trajectory frames")

    acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    print(f"  Baseline accuracy: {acc*100:.3f}%")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
    T_kd = 6.0
    batch_size = 8
    t0 = time.time()

    print(f"  {'Ep':>3} {'Loss':>8} {'NucW':>9} {'Acc':>8} {'EffRank':>8} {'Time':>5}")
    print("  " + "-" * 50)

    for epoch in range(epochs):
        model.train()
        progress = epoch / max(epochs - 1, 1)
        nuc_w = nuc_weight_start + (nuc_weight_end - nuc_weight_start) * progress
        ep_loss = 0.0
        n_b = 0

        # Original frames
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = seg_inputs[idx].to(device)
            tl = seg_logits[idx].to(device)
            ta = teacher_argmax[idx].to(device)

            out = model(x)
            kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                          F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
            ce = F.cross_entropy(out, ta)
            mse = F.mse_loss(out, tl)
            task_loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            # Nuclear norm penalty (computed every 3rd batch for speed)
            if n_b % 3 == 0:
                nuc = nuclear_norm_penalty(model)
                loss = task_loss + nuc_w * nuc
            else:
                loss = task_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        # Trajectory frames (subsample)
        if traj_frames is not None:
            traj_perm = torch.randperm(len(traj_frames))[:N]
            for i in range(0, len(traj_perm), batch_size):
                idx = traj_perm[i:i+batch_size]
                x = traj_frames[idx].float().to(device)
                tl = traj_logits[idx].float().to(device)
                ta = tl.argmax(1)

                out = model(x)
                kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                              F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
                ce = F.cross_entropy(out, ta)
                loss = 0.5 * kd + 0.5 * ce

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()
                n_b += 1

        sched.step()

        # Measure effective rank of a representative layer
        model.eval()
        acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
        with torch.no_grad():
            sample_w = None
            for name, p in model.named_parameters():
                if 'decoder.blocks.0.conv1' in name and p.dim() == 4:
                    sample_w = p
                    break
            if sample_w is not None:
                S = torch.linalg.svdvals(sample_w.reshape(sample_w.shape[0], -1))
                eff_rank = (S.sum() / S[0]).item()
                max_rank = min(sample_w.shape[0], np.prod(sample_w.shape[1:]))
            else:
                eff_rank = max_rank = 0

        elapsed = time.time() - t0
        print(f"  {epoch:3d} {ep_loss/n_b:8.1f} {nuc_w:.2e} {acc*100:7.3f}% "
              f"{eff_rank:5.1f}/{max_rank} {elapsed:4.0f}s", flush=True)

    # Save prehabbed model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / 'segnet_prehabbed.pt')
    print(f"\n  Saved to {OUT_DIR / 'segnet_prehabbed.pt'}")

    del seg_inputs, seg_logits
    torch.cuda.empty_cache()
    return model


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 2: SVD DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════

def phase_svd(device, keep_ratio=0.3):
    """Apply truncated SVD to the prehabbed model."""
    print("=" * 70)
    print(f"PHASE 2: SVD DECOMPOSITION (keep_ratio={keep_ratio:.0%})")
    print("=" * 70)
    print()

    # Load prehabbed model
    sd_path = OUT_DIR / 'segnet_prehabbed.pt'
    if not sd_path.exists():
        print("  No prehabbed model found, using original teacher")
        sd = {k: v.clone() for k, v in
              SegNet().to('cpu').state_dict().items()}
        model = SegNet().to(device)
        model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
        sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        del model
    else:
        sd = torch.load(sd_path, weights_only=True)
        print(f"  Loaded prehabbed model from {sd_path}")

    # Apply SVD
    new_sd = {}
    total_orig = total_comp = 0
    print()
    for name, tensor in sd.items():
        if tensor.dim() == 4 and tensor.shape[2] * tensor.shape[3] > 1 and tensor.numel() > 100:
            O, I, kH, kW = tensor.shape
            mat = tensor.reshape(O, I * kH * kW).float()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            max_rank = min(O, I * kH * kW)
            k = max(1, int(max_rank * keep_ratio))
            # Store factored form: reconstruct for now, factor for compression later
            reconstructed = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
            new_sd[name] = reconstructed.reshape(O, I, kH, kW)
            orig_p = tensor.numel()
            comp_p = k * (O + I * kH * kW + 1)
            total_orig += orig_p
            total_comp += comp_p
            if orig_p > 10000:
                energy = (S[:k] ** 2).sum() / (S ** 2).sum()
                print(f"  {name:<45} rank {k}/{max_rank} energy={energy*100:.1f}% "
                      f"({orig_p:,} -> {comp_p:,})")
        elif tensor.dim() == 2 and tensor.numel() > 100:
            mat = tensor.float()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            k = max(1, int(min(mat.shape) * keep_ratio))
            reconstructed = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
            new_sd[name] = reconstructed
            total_orig += tensor.numel()
            total_comp += k * (mat.shape[0] + mat.shape[1] + 1)
        else:
            new_sd[name] = tensor.clone()

    print(f"\n  Conv/linear params: {total_orig:,} -> {total_comp:,} "
          f"({total_comp/total_orig:.1%})")

    # Load into model and check accuracy
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)

    model = SegNet().eval().to(device)
    model.load_state_dict({k: v.to(device) for k, v in new_sd.items()})
    acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    print(f"  Post-SVD accuracy: {acc*100:.3f}% (will recover with fine-tuning)")

    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
               OUT_DIR / 'segnet_svd.pt')
    print(f"  Saved to {OUT_DIR / 'segnet_svd.pt'}")

    del seg_inputs, seg_logits
    torch.cuda.empty_cache()
    return model


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 3: FINE-TUNE WITH KDIGA
# ═══════════════════════════════════════════════════════════════════════

def phase_finetune(device, epochs=60, lr=3e-4, kdiga_weight=0.05):
    """Fine-tune SVD model with output matching + KDIGA gradient alignment."""
    print("=" * 70)
    print(f"PHASE 3: FINE-TUNE WITH KDIGA (epochs={epochs})")
    print("=" * 70)
    print()

    # Load SVD model
    svd_path = OUT_DIR / 'segnet_svd.pt'
    if not svd_path.exists():
        print("ERROR: Run --phase svd first")
        return None
    model = SegNet().to(device)
    model.load_state_dict(torch.load(svd_path, weights_only=True, map_location=device))

    # Frozen reference teacher for KDIGA
    ref_teacher = SegNet().eval().to(device)
    ref_teacher.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    for p in ref_teacher.parameters():
        p.requires_grad_(False)

    # Load data
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    N = seg_inputs.shape[0]

    traj_frames = traj_logits = traj_argmax = None
    N_traj = 0
    if TRAJ_DIR.exists():
        traj_frames = torch.load(TRAJ_DIR / 'traj_frames.pt', weights_only=True)
        traj_logits = torch.load(TRAJ_DIR / 'traj_logits.pt', weights_only=True)
        traj_argmax = traj_logits.float().argmax(1)
        N_traj = traj_frames.shape[0]
        print(f"  Trajectory data: {N_traj} frames")

    acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    print(f"  Pre-finetune accuracy: {acc*100:.3f}%")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr * 0.01)
    T_kd = 6.0
    batch_size = 8
    best_acc = 0.0
    best_state = None
    t0 = time.time()

    print(f"  {'Ep':>3} {'Loss':>8} {'KDIGA':>8} {'Acc':>8} {'Best':>8} {'Time':>5}")
    print("  " + "-" * 50)

    for epoch in range(epochs):
        model.train()
        ep_loss = ep_kdiga = 0.0
        n_b = n_kg = 0

        # Original frames
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            x = seg_inputs[idx].to(device)
            tl = seg_logits[idx].to(device)
            ta = teacher_argmax[idx].to(device)

            out = model(x)
            kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                          F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
            ce = F.cross_entropy(out, ta)
            mse = F.mse_loss(out, tl)
            loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            # KDIGA via create_graph on 2 samples every 5th batch
            if n_b % 5 == 0 and epoch >= 3:
                x_kg = x[:2].detach().requires_grad_(True)
                # Student gradient (with graph for backprop)
                s_out = model(x_kg)
                s_loss = F.cross_entropy(s_out, ta[:2])
                s_grad = torch.autograd.grad(s_loss, x_kg, create_graph=True)[0]
                # Teacher gradient (no graph needed)
                x_t = x[:2].detach().requires_grad_(True)
                t_out = ref_teacher(x_t)
                t_loss = F.cross_entropy(t_out, ta[:2])
                t_grad = torch.autograd.grad(t_loss, x_t)[0]
                # Align gradients
                kg_loss = F.mse_loss(s_grad, t_grad.detach())
                loss = loss + kdiga_weight * kg_loss
                ep_kdiga += kg_loss.item()
                n_kg += 1

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        # Trajectory frames
        if traj_frames is not None:
            traj_perm = torch.randperm(N_traj)[:N]
            for i in range(0, len(traj_perm), batch_size):
                idx = traj_perm[i:i+batch_size]
                x = traj_frames[idx].float().to(device)
                tl = traj_logits[idx].float().to(device)
                ta = traj_argmax[idx].to(device)

                out = model(x)
                kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                              F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
                ce = F.cross_entropy(out, ta)
                mse = F.mse_loss(out, tl)
                loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

                # KDIGA on trajectory too (every 5th batch)
                if n_b % 5 == 0 and epoch >= 3:
                    x_kg = x[:2].detach().requires_grad_(True)
                    s_out = model(x_kg)
                    s_loss = F.cross_entropy(s_out, ta[:2])
                    s_grad = torch.autograd.grad(s_loss, x_kg, create_graph=True)[0]
                    x_t = x[:2].detach().requires_grad_(True)
                    t_out = ref_teacher(x_t)
                    t_loss = F.cross_entropy(t_out, ta[:2])
                    t_grad = torch.autograd.grad(t_loss, x_t)[0]
                    kg_loss = F.mse_loss(s_grad, t_grad.detach())
                    loss = loss + kdiga_weight * kg_loss
                    ep_kdiga += kg_loss.item()
                    n_kg += 1

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
            torch.save(best_state, OUT_DIR / 'segnet_svd_finetuned.pt')

        elapsed = time.time() - t0
        avg_kg = ep_kdiga / max(n_kg, 1)
        print(f"  {epoch:3d} {ep_loss/n_b:8.1f} {avg_kg:8.4f} {acc*100:7.3f}% "
              f"{best_acc*100:7.3f}% {elapsed:4.0f}s", flush=True)

    print(f"\n  Best accuracy: {best_acc*100:.3f}%")
    print(f"  Saved to {OUT_DIR / 'segnet_svd_finetuned.pt'}")

    del ref_teacher, seg_inputs, seg_logits
    if traj_frames is not None:
        del traj_frames, traj_logits
    torch.cuda.empty_cache()
    return model


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 4: QUANTIZE + COMPRESS
# ═══════════════════════════════════════════════════════════════════════

def phase_quantize(device, keep_ratio=0.3):
    """Quantize SVD FACTORS (not full weights) for true compression."""
    print("=" * 70)
    print("PHASE 4: QUANTIZE + COMPRESS (SVD factor storage)")
    print("=" * 70)
    print()

    ft_path = OUT_DIR / 'segnet_svd_finetuned.pt'
    if not ft_path.exists():
        ft_path = OUT_DIR / 'segnet_svd.pt'
    sd = torch.load(ft_path, weights_only=True)

    # Load data for accuracy checking
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)

    print("  Method 1: Full weight storage (old, wasteful)")
    print("  Method 2: SVD factor storage (U, S, V — much smaller)")
    print()

    for bits in [4, 5, 6, 8]:
        max_val = 2**(bits-1) - 1
        min_val = -2**(bits-1)

        # ── Method 1: Store full weights (old approach) ──
        packed_full = {}
        for name, tensor in sd.items():
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
                packed_full[name] = {'q': q.to(torch.int8).numpy(),
                                     's': scales.half().numpy(), 'shape': list(t.shape)}
            else:
                packed_full[name] = {'f16': t.half().numpy()}
        comp_full = bz2.compress(pickle.dumps(packed_full), 9)

        # ── Method 2: Store SVD factors ──
        packed_svd = {}
        total_factor_params = 0
        for name, tensor in sd.items():
            t = tensor.cpu().float()
            is_conv = (t.dim() == 4 and t.shape[2] * t.shape[3] > 1 and t.numel() > 100)
            is_linear = (t.dim() == 2 and t.numel() > 100)

            if is_conv or is_linear:
                if is_conv:
                    O, I, kH, kW = t.shape
                    mat = t.reshape(O, I * kH * kW)
                else:
                    mat = t
                    O = mat.shape[0]

                U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
                k = max(1, int(min(mat.shape) * keep_ratio))

                U_k = (U[:, :k] * S[:k].unsqueeze(0))  # fold S into U
                Vh_k = Vh[:k, :]

                # Quantize U_k per-row
                u_amax = U_k.abs().amax(dim=1).clamp(min=1e-12)
                u_scales = u_amax / max_val
                u_q = (U_k / u_scales.unsqueeze(1)).round().clamp(min_val, max_val)

                # Quantize Vh_k per-row
                v_amax = Vh_k.abs().amax(dim=1).clamp(min=1e-12)
                v_scales = v_amax / max_val
                v_q = (Vh_k / v_scales.unsqueeze(1)).round().clamp(min_val, max_val)

                packed_svd[name] = {
                    'u': u_q.to(torch.int8).numpy(),
                    'us': u_scales.half().numpy(),
                    'v': v_q.to(torch.int8).numpy(),
                    'vs': v_scales.half().numpy(),
                    'k': k,
                    'shape': list(t.shape),
                }
                total_factor_params += U_k.numel() + Vh_k.numel()
            elif 'running' not in name and 'num_batches' not in name:
                packed_svd[name] = {'f16': t.half().numpy()}
            else:
                packed_svd[name] = {'f32': t.numpy()}

        comp_svd = bz2.compress(pickle.dumps(packed_svd), 9)

        # Save SVD factor version
        path = OUT_DIR / f'segnet_svd_factors_int{bits}.bin'
        with open(path, 'wb') as f:
            f.write(comp_svd)

        # Reconstruct and check accuracy
        recon_sd = {}
        for name, entry in packed_svd.items():
            if 'u' in entry:
                u_q = torch.from_numpy(entry['u']).float()
                u_s = torch.from_numpy(entry['us']).float()
                v_q = torch.from_numpy(entry['v']).float()
                v_s = torch.from_numpy(entry['vs']).float()
                U_dq = u_q * u_s.unsqueeze(1)
                V_dq = v_q * v_s.unsqueeze(1)
                mat = U_dq @ V_dq
                recon_sd[name] = mat.view(entry['shape']).to(device)
            elif 'f16' in entry:
                recon_sd[name] = torch.from_numpy(entry['f16']).float().to(device)
            else:
                recon_sd[name] = torch.from_numpy(entry['f32']).to(device)

        model = SegNet().eval().to(device)
        model.load_state_dict(recon_sd)
        acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)

        print(f"  INT{bits}: full={len(comp_full)/1024:.0f}KB  factors={len(comp_svd)/1024:.0f}KB  "
              f"acc={acc*100:.2f}%  -> {path}")
        del model
        torch.cuda.empty_cache()

    print(f"\n  SVD factor params: {total_factor_params:,}")
    del seg_inputs, seg_logits


# ═══════════════════════════════════════════════════════════════════════
#  PHASE 5: ADVERSARIAL DECODE TRANSFER TEST
# ═══════════════════════════════════════════════════════════════════════

def phase_validate(device, iters=150, num_batches=5):
    """Run adversarial decode with the compressed SVD SegNet + full PoseNet for eval."""
    print("=" * 70)
    print("PHASE 5: ADVERSARIAL DECODE TRANSFER TEST")
    print("=" * 70)
    print()

    # Load best available compressed model
    for path_name in ['segnet_svd_finetuned.pt', 'segnet_svd.pt']:
        p = OUT_DIR / path_name
        if p.exists():
            print(f"  Loading {p}")
            sd = torch.load(p, weights_only=True, map_location=device)
            break
    else:
        print("  ERROR: No SVD model found")
        return

    svd_seg = SegNet().eval().to(device)
    svd_seg.load_state_dict(sd if isinstance(sd, dict) else sd)
    for p in svd_seg.parameters():
        p.requires_grad_(False)

    # Full teachers for evaluation
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p_param in t_seg.parameters(): p_param.requires_grad_(False)
    for p_param in t_pose.parameters(): p_param.requires_grad_(False)

    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    del seg_logits

    IDEAL_COLORS_d = torch.tensor([
        [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
        [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
        [26.8595, 41.0758, 46.1465],
    ], dtype=torch.float32).to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    starts = [0, 100, 200, 300, 450][:num_batches]
    results = {'ts': [], 'tp': []}

    print(f"  Adversarial decode: {num_batches} batches x {bs} pairs x {iters} iters")
    print()

    for bi, st in enumerate(starts):
        end = min(st + bs, 600)
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_outputs[st:end]
        B = tgt_s.shape[0]

        init = IDEAL_COLORS_d[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters - 1, 1))) for i in range(iters)]

        print(f"  Batch {bi+1}/{num_batches} (pairs {st}-{end-1}): ", end='', flush=True)
        for it in range(iters):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            progress = it / max(iters - 1, 1)

            # SVD SegNet for seg gradients
            seg_l = margin_loss(svd_seg(f1), tgt_s, 0.1 if progress < 0.5 else 5.0)

            # Full PoseNet for pose gradients (used for eval, will test proxy later)
            if progress >= 0.3:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:, :6], tgt_p)
                total = 120.0 * seg_l + 0.2 * pose_l
            else:
                total = 120.0 * seg_l

            total.backward()
            opt.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255); f1.data.clamp_(0, 255)
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

        del f0, f1, opt
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

    for bits in [4, 5, 6]:
        bp = OUT_DIR / f'segnet_svd_int{bits}.bin'
        if bp.exists():
            seg_kb = os.path.getsize(bp) / 1024
            archive_kb = seg_kb + 300  # targets
            rate = (archive_kb * 1024) / 37_545_489
            total = dist + 25 * rate
            print(f"  INT{bits}: seg={seg_kb:.0f}KB archive={archive_kb:.0f}KB "
                  f"25*rate={25*rate:.3f} TOTAL={total:.3f}")

    print(f"\n  Leader: 1.95 | Baseline: 4.39")


# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all',
                        choices=['prehab', 'svd', 'finetune', 'quantize', 'validate', 'all'])
    parser.add_argument('--prehab-epochs', type=int, default=30)
    parser.add_argument('--keep-ratio', type=float, default=0.3)
    parser.add_argument('--finetune-epochs', type=int, default=60)
    parser.add_argument('--kdiga-weight', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.phase in ('prehab', 'all'):
        phase_prehab(device, epochs=args.prehab_epochs)
        print()

    if args.phase in ('svd', 'all'):
        phase_svd(device, keep_ratio=args.keep_ratio)
        print()

    if args.phase in ('finetune', 'all'):
        phase_finetune(device, epochs=args.finetune_epochs, lr=args.lr,
                       kdiga_weight=args.kdiga_weight)
        print()

    if args.phase in ('quantize', 'all'):
        phase_quantize(device)
        print()

    if args.phase in ('validate', 'all'):
        phase_validate(device)


if __name__ == '__main__':
    main()
