#!/usr/bin/env python
"""
Test zero-weight adversarial decode: NO SegNet, NO PoseNet during inflate.

Seg proxy: color distance to ideal class colors (0 parameters)
Pose proxy: none — just optimize seg, ignore pose entirely

Then evaluate with the REAL teacher to see what distortion we'd actually get.
"""
import sys, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size

IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251],
    [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493],
    [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


def seg_proxy(frame, ideal_colors):
    """Zero-parameter seg proxy: negative squared color distance per class.

    Returns (B, 5, H, W) logits. Gradient pushes pixels toward their
    nearest class's ideal color.
    """
    # frame: (B, 3, H, W), ideal_colors: (5, 3)
    diff = frame.unsqueeze(1) - ideal_colors.view(1, 5, 3, 1, 1)
    return -(diff ** 2).sum(dim=2)  # (B, 5, H, W)


def margin_loss(logits, target, margin=3.0):
    target_logits = logits.gather(1, target.unsqueeze(1))
    competitor = logits.clone()
    competitor.scatter_(1, target.unsqueeze(1), float('-inf'))
    max_other = competitor.max(dim=1, keepdim=True).values
    return F.relu(max_other - target_logits + margin).mean()


def rgb_to_yuv6_diff(rgb_chw):
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2],
                        Y[..., 0::2, 1::2], Y[..., 1::2, 1::2],
                        U_sub, V_sub], dim=-3)


def posenet_preprocess_diff(x):
    B, T = x.shape[0], x.shape[1]
    flat = x.reshape(B * T, *x.shape[2:])
    yuv = rgb_to_yuv6_diff(flat)
    return yuv.reshape(B, T * yuv.shape[1], *yuv.shape[2:])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Load teacher for EVALUATION ONLY (not used during optimization)
    print("Loading teacher models (for evaluation only, NOT used in proxy)...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters():
        p.requires_grad_(False)
    for p in t_pose.parameters():
        p.requires_grad_(False)
    print("  Teachers loaded (evaluation only)")
    print()

    # Load targets
    print("Loading targets...")
    seg_logits = torch.load('distill_data/seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)
    pose_outputs = torch.load('distill_data/pose_outputs.pt', weights_only=True).to(device)
    del seg_logits
    print(f"  {teacher_argmax.shape[0]} seg targets, {pose_outputs.shape[0]} pose targets")
    print()

    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size

    # Test batches
    bs = 4
    num_iters = 200
    test_starts = [0, 50, 100, 200, 300, 400, 500]  # 7 batches across the video

    print(f"Running adversarial decode with ZERO-WEIGHT PROXY")
    print(f"  No SegNet model. No PoseNet model. Pure color-distance function.")
    print(f"  {len(test_starts)} batches x {bs} pairs x {num_iters} iterations")
    print()

    results = {'ts': [], 'tp': []}

    for bi, st in enumerate(test_starts):
        end = min(st + bs, len(teacher_argmax))
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_outputs[st:end]
        B = tgt_s.shape[0]

        # Initialize from ideal colors
        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone()
        f0 = f0.requires_grad_(True)

        optimizer = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_sched = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(num_iters - 1, 1)))
                    for i in range(num_iters)]

        t0 = time.time()
        print(f"  Batch {bi+1}/{len(test_starts)} (pairs {st}-{end-1}): ", end='', flush=True)

        for it in range(num_iters):
            for pg in optimizer.param_groups:
                pg['lr'] = lr_sched[it]
            optimizer.zero_grad(set_to_none=True)

            # ZERO-WEIGHT SEG PROXY: just color distance, no neural network
            proxy_logits = seg_proxy(f1, colors)
            seg_l = margin_loss(proxy_logits, tgt_s, 5.0)

            # POSE: use stored target vectors + simple L2 loss on YUV features
            # This gives SOME pose gradient without a neural network
            both = torch.stack([f0, f1], dim=1)
            pn_in = posenet_preprocess_diff(both)
            # No pose model — just push YUV toward a "neutral" distribution
            # that won't cause huge pose errors
            pose_l = seg_l.new_zeros(())  # still no pose optimization

            total = 120.0 * seg_l

            total.backward()
            optimizer.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)

            if (it + 1) % 50 == 0:
                print(f"{it+1}", end=' ', flush=True)

        elapsed = time.time() - t0
        print(f"({elapsed:.1f}s)", flush=True)

        # Evaluate with REAL TEACHER
        with torch.no_grad():
            # Upscale to camera resolution (like real eval pipeline)
            f1_up = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round().byte().float()
            f0_up = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round().byte().float()

            # Teacher SegNet evaluation
            ts_in = F.interpolate(f1_up, (mH, mW), mode='bilinear')
            teacher_pred = t_seg(ts_in).argmax(1)
            seg_dist = (teacher_pred != tgt_s).float().mean((1, 2))
            results['ts'].extend(seg_dist.cpu().tolist())

            # Teacher PoseNet evaluation
            tp_pair = F.interpolate(
                torch.stack([f0_up, f1_up], 1).reshape(-1, 3, Hc, Wc),
                (mH, mW), mode='bilinear'
            ).reshape(B, 2, 3, mH, mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:, :6]
            pose_dist = (tpo - tgt_p).pow(2).mean(1)
            results['tp'].extend(pose_dist.cpu().tolist())

        print(f"    Teacher eval: seg_dist={np.mean(results['ts'][-B:]):.6f}  "
              f"pose_mse={np.mean(results['tp'][-B:]):.6f}")

        del f0, f1, optimizer
        torch.cuda.empty_cache()

    # Final summary
    avg_seg = np.mean(results['ts'])
    avg_pose = np.mean(results['tp'])

    print()
    print("=" * 70)
    print("ZERO-WEIGHT PROXY RESULTS")
    print("=" * 70)
    print()
    print(f"  Teacher seg_dist (avg):  {avg_seg:.6f}  ({(1-avg_seg)*100:.3f}% pixel agreement)")
    print(f"  Teacher pose_mse (avg):  {avg_pose:.6f}")
    print()
    print(f"  SCORE BREAKDOWN:")
    score_seg = 100 * avg_seg
    score_pose = math.sqrt(10 * avg_pose) if avg_pose > 0 else 0
    distortion = score_seg + score_pose
    print(f"    100 * seg_dist      = {score_seg:.4f}")
    print(f"    sqrt(10 * pose_mse) = {score_pose:.4f}")
    print(f"    Distortion          = {distortion:.4f}")
    print()

    # Archive estimate: targets (~300KB) + PoseLUT (~216KB) or even less
    # With no models at all: just targets
    for archive_kb in [300, 500, 800]:
        rate = (archive_kb * 1024) / 37_545_489
        total = distortion + 25 * rate
        print(f"    Archive {archive_kb}KB: 25*rate={25*rate:.3f}, TOTAL={total:.4f}")

    print()
    print(f"  Current leader: 1.95")
    print(f"  Baseline:       4.39")


if __name__ == '__main__':
    main()
