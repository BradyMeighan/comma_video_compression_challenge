#!/usr/bin/env python
"""
Novel approach: SVD model provides gradients, stored correction fixes accuracy.

The insight:
- SVD at 30% rank (no fine-tuning): gradients are GOOD (seg_dist=0.077)
- But accuracy is BAD (23%) because logits are miscalibrated
- Fine-tuning fixes accuracy but breaks compressibility

Solution: DON'T fine-tune. Instead, store a per-pixel logit correction:
    logits = svd_model(frame) + correction[frame_id]

The correction is ADDITIVE and DOESN'T DEPEND ON THE INPUT.
So: d(logits)/d(pixels) = d(svd_model)/d(pixels)  ← unchanged!
The gradient landscape is EXACTLY the SVD model's landscape.
But the output (for accuracy) is corrected to match the teacher.

The correction is (600, 5, 384, 512) — huge uncompressed.
But most pixels need zero correction (SVD already gets them right).
The sparse correction should compress well.

Quick test:
1. Compute correction = teacher_logits - svd_model_logits for all 600 frames
2. Compress the correction
3. Run adversarial decode with svd_model + correction
4. Evaluate with real teacher
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


class CorrectedSVDModel(nn.Module):
    """SVD model + stored logit correction. Gradients come from SVD only."""
    def __init__(self, svd_model, corrections):
        super().__init__()
        self.svd_model = svd_model
        # corrections: dict mapping frame_id -> (5, H, W) tensor
        # During adversarial decode we don't know the frame_id,
        # so we use a SPATIAL correction (average across frames)
        # or better: correction based on the TARGET segmap
        self.register_buffer('spatial_correction', corrections)
        self.current_correction = None

    def set_correction(self, correction):
        """Set correction for current batch."""
        self.current_correction = correction

    def forward(self, x):
        logits = self.svd_model(x)
        if self.current_correction is not None:
            logits = logits + self.current_correction
        return logits


def svd_compress(state_dict, keep_ratio):
    """Apply SVD truncation, return new state dict + factor sizes."""
    new_sd = {}
    total_factor_params = 0
    for name, tensor in state_dict.items():
        if tensor.dim() == 4 and tensor.numel() > 100:
            O, I, kH, kW = tensor.shape
            mat = tensor.reshape(O, -1).float()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            k = max(1, int(min(O, I*kH*kW) * keep_ratio))
            reconstructed = (U[:,:k] * S[:k]) @ Vh[:k,:]
            new_sd[name] = reconstructed.reshape(O, I, kH, kW)
            total_factor_params += k * (O + I*kH*kW)
        elif tensor.dim() == 2 and tensor.numel() > 100:
            U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
            k = max(1, int(min(tensor.shape) * keep_ratio))
            new_sd[name] = (U[:,:k] * S[:k]) @ Vh[:k,:]
            total_factor_params += k * sum(tensor.shape)
        else:
            new_sd[name] = tensor.clone()
    return new_sd, total_factor_params


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")

    # Load teacher
    teacher = SegNet().eval().to(device)
    teacher.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    for p in teacher.parameters(): p.requires_grad_(False)

    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_pose.parameters(): p.requires_grad_(False)

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)

    for keep_ratio in [0.3, 0.2, 0.15]:
        print(f"\n{'='*70}")
        print(f"SVD keep={keep_ratio:.0%} + LOGIT CORRECTION")
        print(f"{'='*70}\n")

        # Step 1: Build SVD model
        print("  Building SVD model...")
        svd_sd, factor_params = svd_compress(
            {k: v.clone() for k, v in teacher.state_dict().items()},
            keep_ratio)
        svd_model = SegNet().eval().to(device)
        svd_model.load_state_dict({k: v.to(device) for k, v in svd_sd.items()})

        # Step 2: Compute corrections for all 600 frames
        print("  Computing logit corrections (teacher - svd_model)...")
        corrections = []
        with torch.no_grad():
            for i in range(0, 600, 32):
                x = seg_inputs[i:i+32].to(device)
                teacher_out = teacher(x)
                svd_out = svd_model(x)
                diff = teacher_out - svd_out  # (B, 5, 384, 512)
                corrections.append(diff.cpu())
        all_corrections = torch.cat(corrections, 0)  # (600, 5, 384, 512)

        # Check: SVD + correction should give perfect accuracy
        corrected_acc = 0
        total_px = 0
        with torch.no_grad():
            for i in range(0, 600, 32):
                x = seg_inputs[i:i+32].to(device)
                out = svd_model(x) + all_corrections[i:i+32].to(device)
                pred = out.argmax(1)
                tgt = teacher_argmax[i:i+32].to(device)
                corrected_acc += (pred == tgt).sum().item()
                total_px += tgt.numel()
        print(f"  SVD + full correction accuracy: {corrected_acc/total_px*100:.4f}%")

        # Step 3: Compress the corrections
        # Strategy A: store raw corrections quantized
        # Strategy B: store only the argmax-critical corrections (where SVD gets wrong class)
        # Strategy C: store a low-rank approximation of the corrections

        # Strategy A: INT8 quantized corrections
        corr_flat = all_corrections.float()
        max_val = 127
        amax = corr_flat.abs().amax()
        scale = amax / max_val
        q_corr = (corr_flat / scale).round().clamp(-128, 127).to(torch.int8)
        raw_a = pickle.dumps({'q': q_corr.numpy(), 's': scale.item()})
        comp_a = bz2.compress(raw_a, 9)

        # Strategy B: sparse — only store where argmax differs
        svd_argmax = []
        with torch.no_grad():
            for i in range(0, 600, 32):
                x = seg_inputs[i:i+32].to(device)
                svd_argmax.append(svd_model(x).argmax(1).cpu())
        svd_argmax = torch.cat(svd_argmax, 0)  # (600, 384, 512)
        wrong_mask = (svd_argmax != teacher_argmax.cpu())
        n_wrong = wrong_mask.sum().item()
        n_total = wrong_mask.numel()
        print(f"  SVD wrong pixels: {n_wrong:,} / {n_total:,} ({n_wrong/n_total*100:.1f}%)")

        # Store only the target class at wrong pixels (1 byte each)
        wrong_targets = teacher_argmax.cpu()[wrong_mask].to(torch.uint8)
        wrong_indices = wrong_mask.nonzero(as_tuple=False).to(torch.int32)
        raw_b = pickle.dumps({
            'indices': wrong_indices.numpy(),
            'targets': wrong_targets.numpy(),
            'shape': list(all_corrections.shape),
        })
        comp_b = bz2.compress(raw_b, 9)

        # Strategy C: per-frame mean correction (one 5-dim vector per frame)
        mean_corr = all_corrections.mean(dim=(2, 3))  # (600, 5)
        raw_c = pickle.dumps({'mean_corr': mean_corr.half().numpy()})
        comp_c = bz2.compress(raw_c, 9)

        # SVD factors compressed
        svd_factor_bytes = factor_params * 2  # FP16
        svd_comp = bz2.compress(pickle.dumps({'size': svd_factor_bytes}), 9)
        # Actual SVD factor compression estimate (from earlier tests)
        svd_kb = factor_params * 2 / 1024  # rough FP16

        print(f"\n  SVD factors: {svd_kb:.0f} KB (FP16)")
        print(f"  Correction strategy A (full INT8): {len(comp_a)/1024:.0f} KB")
        print(f"  Correction strategy B (sparse targets): {len(comp_b)/1024:.0f} KB")
        print(f"  Correction strategy C (per-frame mean): {len(comp_c)/1024:.0f} KB")

        # Step 4: Adversarial decode test
        # Use strategy C (smallest) — per-frame mean correction
        # During adversarial decode, we know the frame_id from the target index
        print(f"\n  Running adversarial decode with mean correction...")

        colors = IDEAL_COLORS.to(device)
        mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
        Wc, Hc = camera_size
        results = {'ts': [], 'tp': []}

        for p in svd_model.parameters(): p.requires_grad_(False)

        for st in [0, 100, 200, 300, 450]:
            end = min(st+4, 600)
            tgt_s = teacher_argmax[st:end].to(device)
            tgt_p = pose_targets[st:end]
            B = tgt_s.shape[0]
            batch_corr = mean_corr[st:end].to(device).unsqueeze(-1).unsqueeze(-1)  # (B, 5, 1, 1)

            init = colors.to(device)[tgt_s].permute(0,3,1,2).clone()
            f1 = init.requires_grad_(True)
            f0 = init.detach().mean(dim=(-2,-1),keepdim=True).expand_as(init).clone().requires_grad_(True)
            opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
            lr_s = [0.06+0.57*(1+math.cos(math.pi*i/99)) for i in range(100)]

            for it in range(100):
                for pg in opt.param_groups: pg['lr'] = lr_s[it]
                opt.zero_grad(set_to_none=True)
                p = it/99

                # SVD model + additive correction (correction doesn't affect gradients)
                logits = svd_model(f1) + batch_corr
                seg_l = margin_loss(logits, tgt_s, 0.1 if p<0.5 else 5.0)

                if p >= 0.3:
                    both = torch.stack([f0,f1],dim=1)
                    pn_in = posenet_preprocess_diff(both)
                    pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:,:6], tgt_p)
                    total = 120*seg_l + 0.2*pose_l
                else:
                    total = 120*seg_l
                total.backward()
                opt.step()
                with torch.no_grad(): f0.data.clamp_(0,255); f1.data.clamp_(0,255)

            with torch.no_grad():
                f1u = F.interpolate(f1.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
                f0u = F.interpolate(f0.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
                ts_in = F.interpolate(f1u,(mH,mW),mode='bilinear')
                ts = (teacher(ts_in).argmax(1) != tgt_s).float().mean((1,2))
                results['ts'].extend(ts.cpu().tolist())
                tp_pair = F.interpolate(torch.stack([f0u,f1u],1).reshape(-1,3,Hc,Wc),(mH,mW),mode='bilinear').reshape(B,2,3,mH,mW)
                tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:,:6]
                results['tp'].extend((tpo-tgt_p).pow(2).mean(1).cpu().tolist())
            del f0, f1, opt; torch.cuda.empty_cache()

        seg_d = np.mean(results['ts'])
        pose_d = np.mean(results['tp'])
        s_seg = 100 * seg_d
        s_pose = math.sqrt(10 * pose_d)
        dist = s_seg + s_pose

        total_kb = svd_kb + len(comp_c)/1024 + 300  # SVD + correction + targets
        rate = (total_kb * 1024) / 37_545_489
        score = dist + 25 * rate

        print(f"\n  seg_dist={seg_d:.6f} pose_mse={pose_d:.6f}")
        print(f"  100*seg={s_seg:.4f}  sqrt(10*p)={s_pose:.4f}  distortion={dist:.4f}")
        print(f"  Archive: SVD={svd_kb:.0f}KB + corr={len(comp_c)/1024:.0f}KB + tgt=300KB = {total_kb:.0f}KB")
        print(f"  25*rate={25*rate:.3f}  TOTAL SCORE={score:.3f}")
        print(f"  Leader: 1.95")

        del svd_model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
