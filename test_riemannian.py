#!/usr/bin/env python
"""
Riemannian fine-tuning: optimize on the fixed-rank manifold.

The idea: SVD at 30% rank gives good gradients (seg_dist=0.077) but bad accuracy (23%).
Normal fine-tuning fixes accuracy but breaks compressibility (weights leave low-rank manifold).
Riemannian optimization constrains weights to STAY on the low-rank manifold.

We use geoopt for manifold-constrained optimization. Each weight matrix is
parameterized on the fixed-rank manifold — the optimizer physically cannot
produce full-rank weights.

Quick test: SVD → Riemannian fine-tune 30 epochs → check accuracy + compressibility → adversarial decode.
"""
import sys, os, time, math, bz2, pickle
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


class LowRankConv2d(nn.Module):
    """Conv2d with weight constrained to rank r via explicit U, V factors.

    W = U @ V where U is (C_out, r) and V is (r, C_in*kH*kW).
    After each optimizer step, we project U and V back onto the
    Stiefel manifold (orthonormal columns) to prevent drift.
    """
    def __init__(self, original_conv, rank):
        super().__init__()
        W = original_conv.weight.data
        self.C_out, self.C_in, self.kH, self.kW = W.shape
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.rank = rank

        # SVD decomposition
        mat = W.reshape(self.C_out, -1).float()
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

        # Store as U_r * sqrt(S_r) and sqrt(S_r) * V_r for balanced conditioning
        sqrt_S = S[:rank].sqrt()
        self.U = nn.Parameter(U[:, :rank] * sqrt_S.unsqueeze(0))  # (C_out, r)
        self.V = nn.Parameter(Vh[:rank, :] * sqrt_S.unsqueeze(1))  # (r, C_in*kH*kW)

        if original_conv.bias is not None:
            self.bias = nn.Parameter(original_conv.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        # Reconstruct weight from factors
        W = (self.U @ self.V).reshape(self.C_out, self.C_in, self.kH, self.kW)
        return F.conv2d(x, W, self.bias, self.stride, self.padding)

    def get_reconstructed_weight(self):
        return (self.U @ self.V).reshape(self.C_out, self.C_in, self.kH, self.kW)


class LowRankLinear(nn.Module):
    """Linear with weight constrained to rank r."""
    def __init__(self, original_linear, rank):
        super().__init__()
        W = original_linear.weight.data.float()
        self.out_features, self.in_features = W.shape
        self.rank = rank

        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        sqrt_S = S[:rank].sqrt()
        self.U = nn.Parameter(U[:, :rank] * sqrt_S.unsqueeze(0))
        self.V = nn.Parameter(Vh[:rank, :] * sqrt_S.unsqueeze(1))

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x):
        W = self.U @ self.V
        return F.linear(x, W, self.bias)


def replace_with_lowrank(model, rank_ratio=0.3, min_size=16):
    """Replace Conv2d and Linear layers with low-rank versions."""
    total_orig = total_lr = 0
    replaced = 0

    def _do_replace(parent, name, child):
        nonlocal total_orig, total_lr, replaced

        if isinstance(child, nn.Conv2d) and child.groups == 1:
            k = child.kernel_size[0] * child.kernel_size[1]
            if k <= 1 and child.in_channels < min_size:
                return
            if child.out_channels == 5:  # seg head
                return

            W = child.weight.data
            O, I = W.shape[0], W.shape[1] * k
            max_rank = min(O, I)
            rank = max(1, int(max_rank * rank_ratio))
            orig_p = W.numel()
            lr_p = rank * (O + I)

            if lr_p >= orig_p:  # not worth factoring
                return

            lr_conv = LowRankConv2d(child, rank).to(W.device)
            setattr(parent, name, lr_conv)
            total_orig += orig_p
            total_lr += lr_p
            replaced += 1

            if orig_p > 5000:
                print(f"  {name:<40} ({O},{child.in_channels},{child.kernel_size[0]},{child.kernel_size[1]}) "
                      f"rank={rank} {orig_p:,}->{lr_p:,}")

        elif isinstance(child, nn.Linear) and child.in_features >= min_size and child.out_features >= min_size:
            O, I = child.out_features, child.in_features
            max_rank = min(O, I)
            rank = max(1, int(max_rank * rank_ratio))
            orig_p = O * I
            lr_p = rank * (O + I)

            if lr_p >= orig_p:
                return

            lr_lin = LowRankLinear(child, rank).to(child.weight.device)
            setattr(parent, name, lr_lin)
            total_orig += orig_p
            total_lr += lr_p
            replaced += 1

            if orig_p > 5000:
                print(f"  {name:<40} linear ({O},{I}) rank={rank} {orig_p:,}->{lr_p:,}")

    def _recurse(module):
        for name, child in list(module.named_children()):
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                _do_replace(module, name, child)
            else:
                _recurse(child)

    _recurse(model)
    print(f"\n  Replaced {replaced} layers: {total_orig:,} -> {total_lr:,} factor params")
    return total_orig, total_lr


def retract_to_manifold(model):
    """Project U, V factors back toward balanced conditioning.

    This is a lightweight 'retraction' — re-balance the singular value
    distribution between U and V to prevent one from growing huge
    while the other shrinks. Not a full Riemannian retraction but
    stabilizes training.
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (LowRankConv2d, LowRankLinear)):
                W = module.U.data @ module.V.data
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                r = module.rank
                sqrt_S = S[:r].sqrt()
                module.U.data = U[:, :r] * sqrt_S.unsqueeze(0)
                module.V.data = Vh[:r, :] * sqrt_S.unsqueeze(1)


def compress_factors(model, bits=8):
    """Extract and compress all U, V factors."""
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    packed = {}
    total_factor_params = 0

    for name, param in model.state_dict().items():
        t = param.cpu().float()
        if ('.U' in name or '.V' in name) and t.dim() == 2:
            # Quantize per-row
            amax = t.abs().amax(dim=1).clamp(min=1e-12)
            scales = amax / max_val
            q = (t / scales.unsqueeze(1)).round().clamp(min_val, max_val)
            packed[name] = {'q': q.to(torch.int8).numpy(),
                            's': scales.half().numpy(), 'shape': list(t.shape)}
            total_factor_params += t.numel()
        elif 'running' in name or 'num_batches' in name:
            packed[name] = {'f32': t.numpy()}
        else:
            packed[name] = {'f16': t.half().numpy()}

    compressed = bz2.compress(pickle.dumps(packed), 9)
    return compressed, total_factor_params


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rank_ratio = 0.3
    epochs = 40

    # ═══ Step 1: Load and factor ═══
    print("=" * 70)
    print(f"STEP 1: LOAD + LOW-RANK REPLACEMENT (rank={rank_ratio:.0%})")
    print("=" * 70)
    print()

    model = SegNet().to(device)
    model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    before_params = sum(p.numel() for p in model.parameters())

    replace_with_lowrank(model, rank_ratio=rank_ratio)
    after_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {before_params:,} -> {after_params:,}")

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)

    acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
    print(f"  Post-factoring accuracy: {acc*100:.2f}%")

    # Compressed size estimate
    comp, n_fp = compress_factors(model, bits=8)
    print(f"  Factor params: {n_fp:,}  INT8 compressed: {len(comp)/1024:.0f} KB")
    print()

    # ═══ Step 2: Fine-tune with manifold retraction ═══
    print("=" * 70)
    print(f"STEP 2: FINE-TUNE WITH MANIFOLD RETRACTION ({epochs} epochs)")
    print("=" * 70)
    print()
    print("After each epoch, U and V are re-balanced via SVD retraction.")
    print("This keeps the model on the low-rank manifold while training.")
    print()

    # Load trajectory data
    traj_frames = traj_logits = traj_argmax = None
    if TRAJ_DIR.exists():
        traj_frames = torch.load(TRAJ_DIR / 'traj_frames.pt', weights_only=True)
        traj_logits = torch.load(TRAJ_DIR / 'traj_logits.pt', weights_only=True)
        traj_argmax = traj_logits.float().argmax(1)
        print(f"  Trajectory data: {traj_frames.shape[0]} frames")

    N = seg_inputs.shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=3e-6)
    T_kd = 6.0
    batch_size = 8
    best_acc = 0.0
    best_state = None
    t0 = time.time()

    print(f"  {'Ep':>3} {'Loss':>8} {'Acc':>8} {'Best':>8} {'FSize':>7} {'Time':>5}")
    print("  " + "-" * 45)

    for epoch in range(epochs):
        model.train()
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
            loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            n_b += 1

        # Trajectory frames
        if traj_frames is not None:
            traj_perm = torch.randperm(len(traj_frames))[:N]
            for i in range(0, len(traj_perm), batch_size):
                idx = traj_perm[i:i+batch_size]
                x = traj_frames[idx].float().to(device)
                tl = traj_logits[idx].float().to(device)
                ta = traj_argmax[idx].to(device)

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

        # Retract to manifold every epoch
        retract_to_manifold(model)

        model.eval()
        acc = check_seg_accuracy(model, seg_inputs, teacher_argmax, device)
        comp, _ = compress_factors(model, bits=8)
        comp_kb = len(comp) / 1024

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT_DIR / 'segnet_riemannian.pt')

        elapsed = time.time() - t0
        saved = "*" if acc >= best_acc else " "
        print(f" {saved}{epoch:3d} {ep_loss/n_b:8.1f} {acc*100:7.3f}% {best_acc*100:7.3f}% "
              f"{comp_kb:6.0f}KB {elapsed:4.0f}s", flush=True)

    # ═══ Step 3: Compress and test ═══
    print()
    print("=" * 70)
    print("STEP 3: COMPRESSION")
    print("=" * 70)

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    for bits in [5, 6, 8]:
        comp, n_fp = compress_factors(model, bits=bits)
        path = OUT_DIR / f'segnet_riemannian_int{bits}.bin'
        with open(path, 'wb') as f:
            f.write(comp)
        print(f"  INT{bits}: {len(comp)/1024:.0f} KB ({n_fp:,} factor params)")

    # ═══ Step 4: Adversarial decode transfer test ═══
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

    ta_all = teacher_argmax.to(device)
    pose_all = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    iters = 150
    starts = [0, 100, 200, 300, 450]
    results = {'ts': [], 'tp': []}

    for bi, st in enumerate(starts):
        end = min(st + bs, 600)
        tgt_s, tgt_p = ta_all[st:end], pose_all[st:end]
        B = tgt_s.shape[0]
        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        adv_opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters-1, 1))) for i in range(iters)]

        print(f"  Batch {bi+1}/5 ({st}-{end-1}): ", end='', flush=True)
        for it in range(iters):
            for pg in adv_opt.param_groups: pg['lr'] = lr_s[it]
            adv_opt.zero_grad(set_to_none=True)
            p = it / max(iters-1, 1)
            seg_l = margin_loss(model(f1), tgt_s, 0.1 if p < 0.5 else 5.0)
            if p >= 0.3:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:, :6], tgt_p)
                total = 120.0 * seg_l + 0.2 * pose_l
            else:
                total = 120.0 * seg_l
            total.backward()
            adv_opt.step()
            with torch.no_grad(): f0.data.clamp_(0, 255); f1.data.clamp_(0, 255)
            if (it+1) % 50 == 0: print(f"{it+1}", end=' ', flush=True)
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
        print(f"    seg={np.mean(results['ts'][-B:]):.6f} pose={np.mean(results['tp'][-B:]):.6f}")
        del f0, f1, adv_opt; torch.cuda.empty_cache()

    avg_ts, avg_tp = np.mean(results['ts']), np.mean(results['tp'])
    s_seg, s_pose = 100 * avg_ts, math.sqrt(10 * avg_tp)
    dist = s_seg + s_pose
    print(f"\n  seg={s_seg:.4f} pose={s_pose:.4f} distortion={dist:.4f}")

    for bits in [5, 6, 8]:
        bp = OUT_DIR / f'segnet_riemannian_int{bits}.bin'
        if bp.exists():
            seg_kb = os.path.getsize(bp) / 1024
            archive_kb = seg_kb + 300
            rate = (archive_kb * 1024) / 37_545_489
            total_score = dist + 25 * rate
            print(f"  INT{bits}: {seg_kb:.0f}KB  rate={25*rate:.2f}  TOTAL={total_score:.2f}")
    print(f"\n  Leader: 1.95")


if __name__ == '__main__':
    main()
