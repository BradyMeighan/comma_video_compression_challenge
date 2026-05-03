#!/usr/bin/env python
"""
SVDQuant sweep: vary outlier % and residual bit depth.
Find the smallest size that still preserves gradient quality.
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


def svdquant_compress(sd, outlier_pct, residual_bits):
    """SVDQuant: extract outlier SVs to FP16, quantize residual."""
    max_val = 2**(residual_bits-1) - 1
    min_val = -2**(residual_bits-1)
    packed = {}

    for name, tensor in sd.items():
        t = tensor.cpu().float()
        if t.dim() >= 2 and t.numel() > 100 and 'running' not in name and 'bn' not in name.lower() and 'norm' not in name.lower():
            mat = t.reshape(t.shape[0], -1) if t.dim() == 4 else t
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            n_outlier = max(1, int(min(mat.shape) * outlier_pct))

            U_out = (U[:, :n_outlier] * S[:n_outlier]).half()
            Vh_out = Vh[:n_outlier, :].half()
            outlier_recon = U_out.float() @ Vh_out.float()
            residual = mat - outlier_recon

            # Per-channel quantization of residual
            out_ch = residual.shape[0]
            amax = residual.abs().amax(dim=1).clamp(min=1e-12)
            scales = amax / max_val
            q_res = (residual / scales.unsqueeze(1)).round().clamp(min_val, max_val)

            packed[name] = {
                'u': U_out.numpy(), 'v': Vh_out.numpy(),
                'q': q_res.to(torch.int8).numpy(),
                's': scales.half().numpy(),
                'shape': list(t.shape),
            }
        elif 'running' in name or 'num_batches' in name:
            packed[name] = {'f32': t.numpy()}
        else:
            packed[name] = {'f16': t.half().numpy()}

    compressed = bz2.compress(pickle.dumps(packed), 9)
    return packed, compressed


def reconstruct_svdquant(packed, device):
    sd = {}
    for name, entry in packed.items():
        if 'q' in entry:
            u = torch.from_numpy(entry['u']).float()
            v = torch.from_numpy(entry['v']).float()
            q = torch.from_numpy(entry['q']).float()
            s = torch.from_numpy(entry['s']).float()
            mat = (u @ v) + (q * s.unsqueeze(1))
            sd[name] = mat.view(entry['shape']).to(device)
        elif 'f32' in entry:
            sd[name] = torch.from_numpy(entry['f32']).to(device)
        else:
            sd[name] = torch.from_numpy(entry['f16']).float().to(device)
    return sd


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")

    # Load teachers
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    # Load fine-tuned SVD model
    sd = torch.load(OUT_DIR / 'segnet_svd_finetuned.pt', weights_only=True)

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    del seg_logits

    print("=" * 80)
    print("SVDQuant SWEEP: outlier % x residual bits")
    print("=" * 80)
    print()
    print(f"{'Outlier%':>8} {'ResBits':>7} {'Size':>7} {'Acc':>8} {'SegDist':>8} {'PoseMSE':>9} {'Distort':>8}")
    print("-" * 70)

    for outlier_pct in [0.03, 0.05, 0.10, 0.15, 0.20]:
        for res_bits in [4, 3, 2]:
            packed, compressed = svdquant_compress(sd, outlier_pct, res_bits)
            size_kb = len(compressed) / 1024

            # Reconstruct and check accuracy
            recon = reconstruct_svdquant(packed, device)
            model = SegNet().eval().to(device)
            model.load_state_dict(recon)
            acc = check_acc(model, seg_inputs, teacher_argmax, device)

            # Only run adversarial test if accuracy is reasonable
            if acc > 0.90:
                for p in model.parameters(): p.requires_grad_(False)
                seg_d, pose_d, dist = quick_adv_test(
                    model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
                print(f"  {outlier_pct:6.0%}   INT{res_bits}  {size_kb:6.0f}KB {acc*100:7.2f}% "
                      f"{seg_d:.6f} {pose_d:9.6f} {dist:8.4f}")
            else:
                print(f"  {outlier_pct:6.0%}   INT{res_bits}  {size_kb:6.0f}KB {acc*100:7.2f}% "
                      f"{'SKIP (acc too low)':>35}")

            del model; torch.cuda.empty_cache()

    print(f"\n  Proven: non-factored INT5 = 5500KB, distortion=1.31")
    print(f"  Leader: 1.95")


if __name__ == '__main__':
    main()
