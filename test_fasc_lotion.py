#!/usr/bin/env python
"""
Quick tests for FASC and LOTION — the two untested approaches
that might break the 5MB adversarial decode floor.

Test 1: FASC (Fisher-Aligned SVD) — SVD guided by gradient sensitivity instead of variance
Test 2: LOTION-style INT3 QAT — stochastic rounding during training to smooth INT3 landscape
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
OUT_DIR = Path('compressed_models')
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


def check_acc(model, seg_inputs, teacher_argmax, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(seg_inputs), 32):
            x = seg_inputs[i:i+32].to(device)
            ta = teacher_argmax[i:i+32].to(device)
            correct += (model(x).argmax(1) == ta).sum().item()
            total += ta.numel()
    return correct / total


def quick_adv_test(model, t_seg, t_pose, ta, pt, device):
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    results = {'ts': [], 'tp': []}
    for st in [0, 200, 400]:
        end = min(st+4, 600); tgt_s = ta[st:end]; tgt_p = pt[st:end]; B = tgt_s.shape[0]
        init = colors[tgt_s].permute(0,3,1,2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1),keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0,f1], lr=1.2, weight_decay=0)
        lr_s = [0.06+0.57*(1+math.cos(math.pi*i/99)) for i in range(100)]
        for it in range(100):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            p = it/99
            seg_l = margin_loss(model(f1), tgt_s, 0.1 if p<0.5 else 5.0)
            if p >= 0.3:
                both = torch.stack([f0,f1],dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:,:6], tgt_p)
                total = 120*seg_l + 0.2*pose_l
            else: total = 120*seg_l
            total.backward(); opt.step()
            with torch.no_grad(): f0.data.clamp_(0,255); f1.data.clamp_(0,255)
        with torch.no_grad():
            f1u = F.interpolate(f1.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
            f0u = F.interpolate(f0.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
            ts_in = F.interpolate(f1u,(mH,mW),mode='bilinear')
            results['ts'].extend((t_seg(ts_in).argmax(1)!=tgt_s).float().mean((1,2)).cpu().tolist())
            tp_pair = F.interpolate(torch.stack([f0u,f1u],1).reshape(-1,3,Hc,Wc),(mH,mW),mode='bilinear').reshape(B,2,3,mH,mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:,:6]
            results['tp'].extend((tpo-tgt_p).pow(2).mean(1).cpu().tolist())
        del f0,f1,opt; torch.cuda.empty_cache()
    seg_d, pose_d = np.mean(results['ts']), np.mean(results['tp'])
    return seg_d, pose_d, 100*seg_d + math.sqrt(10*pose_d)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: FASC — Fisher-Aligned SVD
# Standard SVD minimizes ||W - W_approx||_F (activation variance)
# FASC minimizes ||W - W_approx||_F weighted by gradient sensitivity
# This preserves the singular values that matter for GRADIENTS, not just outputs
# ═══════════════════════════════════════════════════════════════════════

def test_fasc(device):
    print("=" * 70)
    print("TEST 1: FASC (Fisher-Aligned SVD)")
    print("=" * 70)
    print("SVD guided by gradient sensitivity instead of variance.")
    print("Preserves singular values that matter for Jacobian, not just output.")
    print()

    # Load teacher + data
    model = SegNet().eval().to(device)
    model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)

    # Step 1: Compute per-layer Fisher Information (gradient sensitivity)
    print("  Computing Fisher Information per layer...")
    fisher_diags = {}  # {layer_name: diagonal Fisher per output channel}

    model.train()  # need gradients
    for i in range(0, min(100, len(seg_inputs)), 4):  # sample 100 frames
        x = seg_inputs[i:i+4].to(device).requires_grad_(False)
        ta = teacher_argmax[i:i+4].to(device)
        out = model(x)
        loss = F.cross_entropy(out, ta)
        model.zero_grad()
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None and p.dim() >= 2:
                # Fisher diagonal ≈ E[grad²]
                grad_sq = p.grad.data.pow(2)
                if name not in fisher_diags:
                    fisher_diags[name] = torch.zeros_like(grad_sq)
                fisher_diags[name] += grad_sq

    # Normalize
    for name in fisher_diags:
        fisher_diags[name] /= 25  # 100 frames / 4 batch

    print(f"  Computed Fisher for {len(fisher_diags)} layers")

    # Step 2: Fisher-weighted SVD
    # Instead of SVD on W, do SVD on (sqrt(F) * W) then reconstruct
    # This keeps singular values that have high gradient sensitivity
    model.eval()
    sd = {k: v.clone() for k, v in model.state_dict().items()}

    for keep_ratio in [0.3, 0.2, 0.15]:
        fasc_sd = {}
        total_orig = total_comp = 0

        for name, tensor in sd.items():
            if tensor.dim() == 4 and tensor.numel() > 100 and name in fisher_diags:
                O, I, kH, kW = tensor.shape
                W = tensor.reshape(O, -1).float()
                F_diag = fisher_diags[name].reshape(O, -1).float()

                # Fisher-weighted matrix: scale rows by sqrt(Fisher)
                # This makes SVD prioritize high-Fisher rows (gradient-sensitive)
                row_importance = F_diag.mean(dim=1).sqrt().clamp(min=1e-8)  # (O,)
                W_weighted = W * row_importance.unsqueeze(1)

                U, S, Vh = torch.linalg.svd(W_weighted, full_matrices=False)
                k = max(1, int(min(O, I*kH*kW) * keep_ratio))

                # Reconstruct: undo the weighting
                W_approx_weighted = (U[:, :k] * S[:k]) @ Vh[:k, :]
                W_approx = W_approx_weighted / row_importance.unsqueeze(1)

                fasc_sd[name] = W_approx.reshape(O, I, kH, kW)
                total_orig += tensor.numel()
                total_comp += k * (O + I*kH*kW)
            elif tensor.dim() == 2 and tensor.numel() > 100 and name in fisher_diags:
                W = tensor.float()
                F_diag = fisher_diags[name].float()
                row_importance = F_diag.mean(dim=1).sqrt().clamp(min=1e-8)
                W_weighted = W * row_importance.unsqueeze(1)
                U, S, Vh = torch.linalg.svd(W_weighted, full_matrices=False)
                k = max(1, int(min(W.shape) * keep_ratio))
                W_approx = ((U[:, :k] * S[:k]) @ Vh[:k, :]) / row_importance.unsqueeze(1)
                fasc_sd[name] = W_approx
                total_orig += tensor.numel()
                total_comp += k * sum(W.shape)
            else:
                fasc_sd[name] = tensor.clone()

        # Load and test
        test_model = SegNet().eval().to(device)
        test_model.load_state_dict({k: v.to(device) for k, v in fasc_sd.items()})
        acc = check_acc(test_model, seg_inputs, teacher_argmax, device)

        print(f"\n  FASC keep={keep_ratio:.0%}: acc={acc*100:.2f}%, {total_comp:,} factor params")
        print(f"  Running adversarial decode test...")
        for p in test_model.parameters(): p.requires_grad_(False)

        t_seg = SegNet().eval().to(device)
        t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
        t_pose = PoseNet().eval().to(device)
        t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
        for p in t_seg.parameters(): p.requires_grad_(False)
        for p in t_pose.parameters(): p.requires_grad_(False)

        pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)

        seg_d, pose_d, dist = quick_adv_test(
            test_model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
        print(f"  => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")
        print(f"     (Standard SVD at {keep_ratio:.0%} was: {[0.077, 0.152, None][int((0.3-keep_ratio)/0.1)]}"
              f" — is FASC better?)")

        del test_model, t_seg, t_pose; torch.cuda.empty_cache()

    del model; torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: LOTION-style INT3 QAT
# Instead of STE (which fails at INT3), use stochastic rounding
# during training to smooth the quantized loss landscape
# ═══════════════════════════════════════════════════════════════════════

def stochastic_quantize(x, bits):
    """LOTION-style stochastic rounding quantization."""
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    # Per-channel scale
    if x.dim() >= 2:
        flat = x.view(x.shape[0], -1)
        amax = flat.abs().amax(dim=1).clamp(min=1e-12)
        scale = amax / max_val
        scaled = flat / scale.unsqueeze(1)
    else:
        amax = x.abs().max().clamp(min=1e-12)
        scale = amax / max_val
        scaled = x / scale

    # Stochastic rounding: round up with probability (x - floor(x))
    floor_val = scaled.floor().clamp(min_val, max_val)
    ceil_val = (scaled + 1).floor().clamp(min_val, max_val)
    prob_ceil = scaled - floor_val  # fractional part
    rand = torch.rand_like(prob_ceil)
    quantized = torch.where(rand < prob_ceil, ceil_val, floor_val)

    # Dequantize
    if x.dim() >= 2:
        return (quantized * scale.unsqueeze(1)).view_as(x)
    return quantized * scale


def fake_quantize_ste(x, bits):
    """Standard STE fake quantization for comparison."""
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    if x.dim() >= 2:
        flat = x.view(x.shape[0], -1)
        amax = flat.abs().amax(dim=1).clamp(min=1e-12)
        scale = amax / max_val
        scaled = flat / scale.unsqueeze(1)
        q = scaled.round().clamp(min_val, max_val)
        return (q * scale.unsqueeze(1)).view_as(x)
    amax = x.abs().max().clamp(min=1e-12)
    scale = amax / max_val
    q = (x / scale).round().clamp(min_val, max_val)
    return q * scale


class QuantizedSegNet(nn.Module):
    """Wraps SegNet with fake quantization on all conv/linear weights."""
    def __init__(self, base_model, bits=3, use_stochastic=True):
        super().__init__()
        self.base = base_model
        self.bits = bits
        self.use_stochastic = use_stochastic

    def forward(self, x):
        # Apply fake quantization to all weights
        for module in self.base.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, '_orig_weight'):
                    module.weight.data = module._orig_weight.data.clone()
                else:
                    module._orig_weight = module.weight.data.clone()

                if self.use_stochastic and self.training:
                    module.weight.data = stochastic_quantize(module._orig_weight, self.bits)
                else:
                    module.weight.data = fake_quantize_ste(module._orig_weight, self.bits)

        return self.base(x)

    def restore_weights(self):
        for module in self.base.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, '_orig_weight'):
                module.weight.data = module._orig_weight.data.clone()


def test_lotion_int3(device, epochs=20):
    print("\n" + "=" * 70)
    print(f"TEST 2: LOTION-style INT3 QAT ({epochs} epochs)")
    print("=" * 70)
    print("Stochastic rounding during training to smooth INT3 landscape.")
    print("Compare: STE (standard) vs stochastic rounding (LOTION-style).")
    print()

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    N = seg_inputs.shape[0]

    # Load the SVD fine-tuned model (our best working model)
    base_sd = torch.load(OUT_DIR / 'segnet_svd_finetuned.pt', weights_only=True)

    for method_name, use_stochastic in [("LOTION (stochastic)", True), ("STE (standard)", False)]:
        print(f"\n  --- {method_name} ---")

        base = SegNet().to(device)
        base.load_state_dict({k: v.to(device) for k, v in base_sd.items()})
        model = QuantizedSegNet(base, bits=3, use_stochastic=use_stochastic)

        # Check pre-QAT accuracy
        model.eval()
        acc = check_acc(model, seg_inputs, teacher_argmax, device)
        print(f"  Pre-QAT INT3 accuracy: {acc*100:.2f}%")

        # QAT training
        T_kd = 6.0
        opt = torch.optim.AdamW(
            [p for p in base.parameters() if p.requires_grad],
            lr=5e-5, weight_decay=0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)
        t0 = time.time()

        for ep in range(epochs):
            model.train()
            perm = torch.randperm(N)
            ep_loss = n_b = 0
            for i in range(0, N, 8):
                idx = perm[i:i+8]
                x = seg_inputs[idx].to(device)
                tl = seg_logits[idx].to(device)
                ta = teacher_argmax[idx].to(device)

                out = model(x)
                kd = F.kl_div(F.log_softmax(out/T_kd,1), F.softmax(tl/T_kd,1),
                              reduction='batchmean') * T_kd**2
                ce = F.cross_entropy(out, ta)
                loss = 0.5*kd + 0.5*ce

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item(); n_b += 1

            sched.step()
            model.eval()
            acc = check_acc(model, seg_inputs, teacher_argmax, device)
            print(f"  ep {ep:2d}: loss={ep_loss/n_b:.1f} acc={acc*100:.2f}% ({time.time()-t0:.0f}s)",
                  flush=True)

        # Adversarial decode test
        print(f"\n  Running adversarial decode test ({method_name})...")
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)

        t_seg = SegNet().eval().to(device)
        t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
        t_pose = PoseNet().eval().to(device)
        t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
        for p in t_seg.parameters(): p.requires_grad_(False)
        for p in t_pose.parameters(): p.requires_grad_(False)
        pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)

        seg_d, pose_d, dist = quick_adv_test(
            model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
        print(f"  => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")

        # Compressed size at INT3
        model.restore_weights()
        sd = base.state_dict()
        packed = {}
        for name, tensor in sd.items():
            t = tensor.cpu().float()
            if t.dim() >= 2 and t.numel() > 10 and 'running' not in name:
                flat = t.view(t.shape[0], -1)
                amax = flat.abs().amax(dim=1).clamp(min=1e-12)
                scales = amax / 3  # INT3: max_val=3
                q = (flat / scales.unsqueeze(1)).round().clamp(-4, 3)
                packed[name] = {'q': q.to(torch.int8).numpy(),
                                's': scales.half().numpy(), 'shape': list(t.shape)}
            else:
                packed[name] = {'f16': t.half().numpy()}
        compressed = bz2.compress(pickle.dumps(packed), 9)
        print(f"  INT3 compressed size: {len(compressed)/1024:.0f} KB")

        del model, base, t_seg, t_pose; torch.cuda.empty_cache()


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_fasc(device)
    torch.cuda.empty_cache()

    test_lotion_int3(device, epochs=20)

    print("\n" + "=" * 70)
    print("Leader: 1.95 | Our 5MB floor: 4.40")
    print("=" * 70)


if __name__ == '__main__':
    main()
