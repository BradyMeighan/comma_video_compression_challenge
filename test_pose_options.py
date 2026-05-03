#!/usr/bin/env python
"""
Test PoseNet replacements with the working SVD SegNet.

Test A: SVD SegNet + PoseLUT (216KB) — does PoseLUT work with structured frames?
Test B: SVD SegNet + geometric proxy (0KB) — photometric alignment between frame pairs
Test C: SVD SegNet + no pose at all — what's the pose penalty if we just skip it?

All tests use the fine-tuned non-factored SVD SegNet (1.31 distortion proven).
"""
import sys, time, math
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


def photometric_pose_loss(f0, f1):
    """Zero-parameter differentiable photometric loss between frame pairs.

    Measures multi-scale structural difference between frames.
    Gradient w.r.t. pixels encourages temporal coherence.
    """
    loss = 0.0
    a, b = f0, f1
    for scale in range(3):
        # L1 photometric difference
        diff = (a - b).abs().mean()
        # SSIM-like structural term (simplified)
        mu_a = F.avg_pool2d(a, 7, stride=1, padding=3)
        mu_b = F.avg_pool2d(b, 7, stride=1, padding=3)
        sigma_ab = F.avg_pool2d(a * b, 7, stride=1, padding=3) - mu_a * mu_b
        sigma_a2 = F.avg_pool2d(a * a, 7, stride=1, padding=3) - mu_a * mu_a
        sigma_b2 = F.avg_pool2d(b * b, 7, stride=1, padding=3) - mu_b * mu_b
        C1, C2 = 0.01 ** 2 * 255 ** 2, 0.03 ** 2 * 255 ** 2
        ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / \
               ((mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a2 + sigma_b2 + C2))
        loss += diff + 0.15 * (1 - ssim).mean()
        # Downsample for next scale
        a = F.avg_pool2d(a, 2)
        b = F.avg_pool2d(b, 2)
    return loss / 3


def run_test(name, svd_seg, pose_fn, teacher_argmax, pose_targets,
             t_seg, t_pose, device, iters=150):
    """Run adversarial decode with given pose function, eval with real teacher."""
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    starts = [0, 100, 200, 300, 450]
    results = {'ts': [], 'tp': []}

    print(f"\n  === TEST: {name} ===")
    for bi, st in enumerate(starts):
        end = min(st + bs, 600)
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_targets[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters - 1, 1)))
                for i in range(iters)]

        print(f"  Batch {bi+1}/5 ({st}-{end-1}): ", end='', flush=True)
        for it in range(iters):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            progress = it / max(iters - 1, 1)

            seg_l = margin_loss(svd_seg(f1), tgt_s, 0.1 if progress < 0.5 else 5.0)

            if progress >= 0.3:
                pose_l = pose_fn(f0, f1, tgt_p)
                total = 120.0 * seg_l + 0.2 * pose_l
            else:
                total = 120.0 * seg_l

            total.backward()
            opt.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)
            if (it + 1) % 50 == 0: print(f"{it+1}", end=' ', flush=True)
        print("done", flush=True)

        # Eval with real teacher
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
        del f0, f1, opt
        torch.cuda.empty_cache()

    avg_ts = np.mean(results['ts'])
    avg_tp = np.mean(results['tp'])
    s_seg = 100 * avg_ts
    s_pose = math.sqrt(10 * avg_tp)
    dist = s_seg + s_pose
    print(f"\n  RESULT [{name}]: seg={s_seg:.4f} pose={s_pose:.4f} distortion={dist:.4f}")
    return dist


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")

    # Load SVD SegNet (the working one)
    print("Loading SVD SegNet (non-factored fine-tuned, 1.31 distortion proven)...")
    svd_seg = SegNet().eval().to(device)
    svd_sd = torch.load('compressed_models/segnet_svd_finetuned.pt',
                         weights_only=True, map_location=device)
    svd_seg.load_state_dict(svd_sd)
    for p in svd_seg.parameters(): p.requires_grad_(False)

    # Load real teachers for eval
    print("Loading teachers (eval only)...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    # Load PoseLUT
    print("Loading PoseLUT...")
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True)
    s_pose = PoseLUT(n_frames=600, pose_dim=6, embed_dim=32, base_ch=16).to(device)
    s_pose.load_state_dict(torch.load('tiny_models/pose_lut.pt',
                                       weights_only=True, map_location=device))
    s_pose.eval()
    for p in s_pose.parameters(): p.requires_grad_(False)

    # Load targets
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)
    pose_targets = pose_outputs.to(device)
    del seg_logits

    # ─── Test A: PoseLUT ───
    def pose_fn_poselut(f0, f1, tgt_p):
        both = torch.stack([f0, f1], dim=1)
        pn_in = posenet_preprocess_diff(both)
        return F.smooth_l1_loss(s_pose(pn_in)['pose'], tgt_p)

    dist_a = run_test("PoseLUT (216KB)", svd_seg, pose_fn_poselut,
                       teacher_argmax, pose_targets, t_seg, t_pose, device)

    torch.cuda.empty_cache()

    # ─── Test B: Geometric proxy (photometric alignment) ───
    def pose_fn_geometric(f0, f1, tgt_p):
        return photometric_pose_loss(f0, f1)

    dist_b = run_test("Geometric proxy (0KB)", svd_seg, pose_fn_geometric,
                       teacher_argmax, pose_targets, t_seg, t_pose, device)

    torch.cuda.empty_cache()

    # ─── Test C: No pose optimization ───
    def pose_fn_none(f0, f1, tgt_p):
        return f0.new_zeros(())

    dist_c = run_test("No pose (0KB)", svd_seg, pose_fn_none,
                       teacher_argmax, pose_targets, t_seg, t_pose, device)

    # ─── Summary ───
    print("\n" + "=" * 70)
    print("SUMMARY — Pose replacement options with SVD SegNet")
    print("=" * 70)
    print(f"  Full PoseNet (56MB):     distortion = 1.31 (proven)")
    print(f"  PoseLUT (216KB):         distortion = {dist_a:.4f}")
    print(f"  Geometric proxy (0KB):   distortion = {dist_b:.4f}")
    print(f"  No pose at all (0KB):    distortion = {dist_c:.4f}")
    print()

    for name, dist, model_kb in [("PoseLUT", dist_a, 216),
                                   ("Geometric", dist_b, 0),
                                   ("No pose", dist_c, 0)]:
        # SegNet 5.5MB INT5 + pose + targets
        archive_kb = 5500 + model_kb + 300
        rate = (archive_kb * 1024) / 37_545_489
        total = dist + 25 * rate
        print(f"  {name}: archive={archive_kb}KB rate={25*rate:.2f} total={total:.2f}")
    print(f"\n  Leader: 1.95")


if __name__ == '__main__':
    main()
