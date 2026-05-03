#!/usr/bin/env python
"""
Test: color-distance seg proxy + PoseLUT for pose.
No teacher SegNet. No teacher PoseNet.
Only PoseLUT (216KB) as a model in the archive.
"""
import sys, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import PoseLUT, rgb_to_yuv6_diff, posenet_preprocess_diff

IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251],
    [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493],
    [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


def seg_proxy(frame, ideal_colors):
    diff = frame.unsqueeze(1) - ideal_colors.view(1, 5, 3, 1, 1)
    return -(diff ** 2).sum(dim=2)


def margin_loss(logits, target, margin=3.0):
    target_logits = logits.gather(1, target.unsqueeze(1))
    competitor = logits.clone()
    competitor.scatter_(1, target.unsqueeze(1), float('-inf'))
    max_other = competitor.max(dim=1, keepdim=True).values
    return F.relu(max_other - target_logits + margin).mean()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load teachers for EVALUATION ONLY
    print("Loading teachers (eval only)...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    # Load PoseLUT (THIS goes in the archive — 216KB)
    print("Loading PoseLUT (216KB — the only model in archive)...")
    pose_outputs = torch.load('distill_data/pose_outputs.pt', weights_only=True)
    N_frames = pose_outputs.shape[0]
    s_pose = PoseLUT(n_frames=N_frames, pose_dim=6, embed_dim=32, base_ch=16).to(device)
    s_pose.load_state_dict(torch.load('tiny_models/pose_lut.pt', weights_only=True,
                                       map_location=device))
    s_pose.eval()
    for p in s_pose.parameters(): p.requires_grad_(False)
    print(f"  PoseLUT: {sum(p.numel() for p in s_pose.parameters()):,} params")

    # Load targets
    seg_logits = torch.load('distill_data/seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)
    pose_targets = pose_outputs.to(device)
    del seg_logits
    print(f"  {teacher_argmax.shape[0]} targets loaded")
    print()

    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    num_iters = 200
    test_starts = [0, 50, 100, 200, 300, 400, 500]

    print(f"Adversarial decode: color proxy (seg) + PoseLUT (pose)")
    print(f"  {len(test_starts)} batches x {bs} pairs x {num_iters} iters")
    print()

    results = {'ts': [], 'tp': []}

    for bi, st in enumerate(test_starts):
        end = min(st + bs, len(teacher_argmax))
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_targets[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1), keepdim=True).expand_as(init).clone().requires_grad_(True)

        optimizer = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_sched = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(num_iters-1, 1)))
                    for i in range(num_iters)]

        t0 = time.time()
        print(f"  Batch {bi+1}/{len(test_starts)} (pairs {st}-{end-1}): ", end='', flush=True)

        for it in range(num_iters):
            for pg in optimizer.param_groups: pg['lr'] = lr_sched[it]
            optimizer.zero_grad(set_to_none=True)

            progress = it / max(num_iters - 1, 1)

            # SEG: color-distance proxy (0 params)
            proxy_logits = seg_proxy(f1, colors)
            seg_l = margin_loss(proxy_logits, tgt_s, 5.0)

            # POSE: PoseLUT (216KB)
            run_pose = (progress >= 0.3)  # start pose after 30% of iters
            if run_pose:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_pred = s_pose(pn_in)['pose']
                pose_l = F.smooth_l1_loss(pose_pred, tgt_p)
            else:
                pose_l = seg_l.new_zeros(())

            # Same weights as the real inflate.py
            a_eff = 120.0 if progress < 0.9 else 24.0
            b_eff = 0.06 if progress < 0.3 else (0.3 if progress < 0.9 else 1.6)
            total = a_eff * seg_l + b_eff * pose_l

            total.backward()
            optimizer.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)

            if (it+1) % 50 == 0:
                print(f"{it+1}", end=' ', flush=True)

        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)", flush=True)

        # Evaluate with REAL teacher
        with torch.no_grad():
            f1_up = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0,255).round().byte().float()
            f0_up = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0,255).round().byte().float()

            ts_in = F.interpolate(f1_up, (mH, mW), mode='bilinear')
            seg_dist = (t_seg(ts_in).argmax(1) != tgt_s).float().mean((1,2))
            results['ts'].extend(seg_dist.cpu().tolist())

            tp_pair = F.interpolate(
                torch.stack([f0_up, f1_up], 1).reshape(-1, 3, Hc, Wc),
                (mH, mW), mode='bilinear'
            ).reshape(B, 2, 3, mH, mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:, :6]
            pose_mse = (tpo - tgt_p).pow(2).mean(1)
            results['tp'].extend(pose_mse.cpu().tolist())

        print(f"    seg_dist={np.mean(results['ts'][-B:]):.6f}  "
              f"pose_mse={np.mean(results['tp'][-B:]):.6f}")

        del f0, f1, optimizer
        torch.cuda.empty_cache()

    avg_seg = np.mean(results['ts'])
    avg_pose = np.mean(results['tp'])
    score_seg = 100 * avg_seg
    score_pose = math.sqrt(10 * avg_pose) if avg_pose > 0 else 0
    distortion = score_seg + score_pose

    print()
    print("=" * 70)
    print("RESULTS: Color Proxy (seg) + PoseLUT (pose)")
    print("=" * 70)
    print(f"  seg_dist:  {avg_seg:.6f}  ({(1-avg_seg)*100:.2f}% pixel agreement)")
    print(f"  pose_mse:  {avg_pose:.6f}")
    print(f"  100*seg    = {score_seg:.4f}")
    print(f"  sqrt(10*p) = {score_pose:.4f}")
    print(f"  distortion = {distortion:.4f}")
    print()

    # Only PoseLUT (216KB) + targets (~300KB) in archive
    archive_kb = 516
    rate = (archive_kb * 1024) / 37_545_489
    total = distortion + 25 * rate
    print(f"  Archive ~{archive_kb}KB: 25*rate={25*rate:.3f}")
    print(f"  TOTAL SCORE = {total:.4f}")
    print(f"  Leader: 1.95 | Baseline: 4.39")


if __name__ == '__main__':
    main()
