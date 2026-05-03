#!/usr/bin/env python
"""
Test extreme quantization of teacher SegNet + PoseNet.
Same architecture, same weights, just fewer bits per weight.

Tests INT8, INT4, INT3, INT2 quantization:
1. Forward accuracy on 600 frames (does argmax still match?)
2. Adversarial decode transfer (do gradients still work?)
3. Compressed size (what goes in the archive?)
"""
import sys, time, math, bz2, pickle
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import PoseLUT, posenet_preprocess_diff, margin_loss

DISTILL_DIR = Path('distill_data')
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


def quantize_state_dict(state_dict, bits):
    """Quantize conv/linear WEIGHTS to N bits. Keep BN params + biases as FP16.

    Uses per-CHANNEL quantization for conv weights (much better than per-tensor).
    BatchNorm running_mean/running_var/weight/bias are NEVER quantized — they're
    tiny but critical for normalization.
    """
    max_val = 2**(bits-1) - 1
    min_val = -2**(bits-1)
    packed = {}
    for name, tensor in state_dict.items():
        t = tensor.cpu().float()
        is_weight = ('weight' in name and t.dim() >= 2
                     and 'bn' not in name.lower()
                     and 'norm' not in name.lower()
                     and 'running' not in name)
        if is_weight and t.numel() > 10:
            # Per-channel quantization (one scale per output channel)
            out_ch = t.shape[0]
            flat = t.view(out_ch, -1)
            amax = flat.abs().amax(dim=1).clamp(min=1e-12)  # (out_ch,)
            scales = amax / max_val  # (out_ch,)
            q = (flat / scales.unsqueeze(1)).round().clamp(min_val, max_val)
            packed[name] = {
                'q': q.to(torch.int8).numpy(),
                's': scales.half().numpy(),
                'shape': list(t.shape),
            }
        else:
            # Keep as FP16 (biases, BN params, small tensors)
            packed[name] = {'f16': t.half().numpy()}

    raw = pickle.dumps(packed)
    compressed = bz2.compress(raw, 9)
    return packed, len(compressed)


def dequantize_state_dict(packed, device):
    """Reconstruct FP32 state dict from quantized."""
    sd = {}
    for name, entry in packed.items():
        if 'q' in entry:
            q = torch.from_numpy(entry['q']).float()
            scales = torch.from_numpy(entry['s']).float()
            shape = entry['shape']
            out_ch = shape[0]
            flat = q * scales.unsqueeze(1)
            sd[name] = flat.view(shape).to(device)
        else:
            sd[name] = torch.from_numpy(entry['f16']).float().to(device)
    return sd


def check_seg_accuracy(model, seg_inputs, teacher_argmax, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(seg_inputs), 32):
            x = seg_inputs[i:i+32].to(device)
            ta = teacher_argmax[i:i+32].to(device)
            correct += (model(x).argmax(1) == ta).sum().item()
            total += ta.numel()
    return correct / total


def run_adversarial_decode_test(seg_model, pose_model, teacher_argmax, pose_targets,
                                 t_seg, t_pose, device, num_batches=5, iters=150):
    """Run adversarial decode with quantized models, evaluate with fp32 teacher."""
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    starts = [0, 100, 200, 300, 450][:num_batches]
    results = {'ts': [], 'tp': []}

    for bi, st in enumerate(starts):
        end = min(st + bs, len(teacher_argmax))
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_targets[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_sched = [0.06 + 0.57*(1+math.cos(math.pi*i/max(iters-1,1))) for i in range(iters)]

        for it in range(iters):
            for pg in opt.param_groups: pg['lr'] = lr_sched[it]
            opt.zero_grad(set_to_none=True)
            progress = it / max(iters-1, 1)

            seg_l = margin_loss(seg_model(f1), tgt_s, 0.1 if progress < 0.5 else 5.0)
            if progress >= 0.3:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_out = pose_model(pn_in)
                # Handle both PoseNet (returns dict) and PoseLUT (returns dict)
                if isinstance(pose_out, dict):
                    pose_pred = pose_out['pose'][:, :6] if pose_out['pose'].shape[-1] > 6 else pose_out['pose']
                else:
                    pose_pred = pose_out[:, :6]
                pose_l = F.smooth_l1_loss(pose_pred, tgt_p)
                total = 120.0 * seg_l + 0.2 * pose_l
            else:
                total = 120.0 * seg_l

            total.backward()
            opt.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)

        # Eval with real teacher
        with torch.no_grad():
            f1_up = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0,255).round().byte().float()
            f0_up = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0,255).round().byte().float()
            ts_in = F.interpolate(f1_up, (mH, mW), mode='bilinear')
            ts = (t_seg(ts_in).argmax(1) != tgt_s).float().mean((1,2))
            results['ts'].extend(ts.cpu().tolist())

            tp_pair = F.interpolate(
                torch.stack([f0_up, f1_up], 1).reshape(-1, 3, Hc, Wc),
                (mH, mW), mode='bilinear').reshape(B, 2, 3, mH, mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:, :6]
            results['tp'].extend((tpo - tgt_p).pow(2).mean(1).cpu().tolist())

        print(f"    batch {bi+1}: seg={np.mean(results['ts'][-B:]):.6f} "
              f"pose={np.mean(results['tp'][-B:]):.6f}", flush=True)

        del f0, f1, opt
        torch.cuda.empty_cache()

    return np.mean(results['ts']), np.mean(results['tp'])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load real teacher (for evaluation)
    print("Loading FP32 teachers (for evaluation)...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    # Get original state dicts
    seg_sd = {k: v.clone() for k, v in t_seg.state_dict().items()}
    pose_sd = {k: v.clone() for k, v in t_pose.state_dict().items()}
    seg_params = sum(v.numel() for v in seg_sd.values())
    pose_params = sum(v.numel() for v in pose_sd.values())
    print(f"  SegNet: {seg_params:,} params")
    print(f"  PoseNet: {pose_params:,} params\n")

    # Load data
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    del seg_logits

    # Test each bit depth
    print("=" * 80)
    print(f"{'Bits':>4} {'SegSize':>8} {'PoseSize':>9} {'Total':>8} {'SegAcc':>8} "
          f"{'AdvSeg':>8} {'AdvPose':>9} {'Score':>7}")
    print("=" * 80)

    for bits in [8, 6, 5, 4, 3, 2]:
        # Quantize SegNet
        seg_packed, seg_compressed = quantize_state_dict(seg_sd, bits)
        seg_dq = dequantize_state_dict(seg_packed, device)

        # Quantize PoseNet
        pose_packed, pose_compressed = quantize_state_dict(pose_sd, bits)
        pose_dq = dequantize_state_dict(pose_packed, device)

        total_compressed = seg_compressed + pose_compressed

        # Load quantized weights into fresh models
        q_seg = SegNet().eval().to(device)
        q_seg.load_state_dict(seg_dq)

        q_pose = PoseNet().eval().to(device)
        q_pose.load_state_dict(pose_dq)

        # Forward accuracy
        seg_acc = check_seg_accuracy(q_seg, seg_inputs, teacher_argmax, device)

        # Adversarial decode transfer (quick test: 3 batches, 100 iters)
        print(f"\n  INT{bits} adversarial decode test:", flush=True)
        for p in q_seg.parameters(): p.requires_grad_(False)
        for p in q_pose.parameters(): p.requires_grad_(False)

        avg_ts, avg_tp = run_adversarial_decode_test(
            q_seg, q_pose, teacher_argmax.to(device), pose_outputs,
            t_seg, t_pose, device, num_batches=3, iters=100)

        # Score
        distortion = 100 * avg_ts + math.sqrt(10 * avg_tp)
        target_size = 300 * 1024  # targets
        archive = total_compressed + target_size
        rate = archive / 37_545_489
        score = distortion + 25 * rate

        print(f"  INT{bits:d} {seg_compressed/1024:7.0f}KB {pose_compressed/1024:8.0f}KB "
              f"{total_compressed/1024:7.0f}KB {seg_acc*100:7.2f}% "
              f"{avg_ts:.6f} {avg_tp:9.6f} {score:7.3f}")

        del q_seg, q_pose
        torch.cuda.empty_cache()

    print("\n  Leader: 1.95 | Baseline: 4.39")


if __name__ == '__main__':
    main()
