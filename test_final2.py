"""Test combinations of avg_last_k and noise_sigma. Also fine-tune avg K."""
import numpy as np, time, sys
import torch
import torch.nn.functional as F
import einops
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from submissions.phase2.inflate import (
    load_targets, rgb_to_yuv6_diff, posenet_preprocess_diff, margin_loss
)

device = torch.device('cuda')
segnet = SegNet().eval().to(device)
segnet.load_state_dict(load_file(str(segnet_sd_path), device='cuda'))
posenet = PoseNet().eval().to(device)
posenet.load_state_dict(load_file(str(posenet_sd_path), device='cuda'))
for p in segnet.parameters():
    p.requires_grad_(False)
for p in posenet.parameters():
    p.requires_grad_(False)

seg_maps_np, pose_vectors_np, ideal_colors_np = load_targets(Path('submissions/phase2/archive/0'))
ideal_colors = torch.from_numpy(ideal_colors_np.copy()).float().to(device)
W_cam, H_cam = camera_size

TEST_PAIRS = 64
seg_gt = torch.from_numpy(seg_maps_np[:TEST_PAIRS]).long().to(device)
pose_gt = torch.from_numpy(pose_vectors_np[:TEST_PAIRS].copy()).float().to(device)


def custom_optimize(
    target_seg, target_pose,
    num_iters=150, lr=1.2, seg_margin=0.3,
    alpha=80.0, beta=0.15, weight_decay=0.025,
    avg_last_k=0, noise_sigma=0.0,
):
    B = target_seg.shape[0]
    scaler = torch.amp.GradScaler('cuda')

    init_odd = ideal_colors[target_seg].permute(0, 3, 1, 2).clone()
    if noise_sigma > 0:
        init_odd = (init_odd + torch.randn_like(init_odd) * noise_sigma).clamp_(0, 255)

    frame_1 = init_odd.requires_grad_(True)
    f0_init = init_odd.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init_odd).clone()
    frame_0 = f0_init.requires_grad_(True)

    optimizer = torch.optim.AdamW([frame_0, frame_1], lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=lr * 0.05
    )

    if avg_last_k > 0:
        f0_accum = torch.zeros_like(init_odd)
        f1_accum = torch.zeros_like(init_odd)
        avg_count = 0

    for it in range(num_iters):
        progress = it / max(num_iters - 1, 1)
        if progress < 0.5:
            a_eff = alpha
            b_eff = beta * 0.3
        elif progress < 0.85:
            a_eff = alpha
            b_eff = beta * 1.5
        else:
            a_eff = alpha * 0.2
            b_eff = beta * 8.0

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            seg_logits = segnet(frame_1)
            seg_l = margin_loss(seg_logits, target_seg, seg_margin)
            both = torch.stack([frame_0, frame_1], dim=1)
            pn_in = posenet_preprocess_diff(both)
            pn_out = posenet(pn_in)['pose'][:, :6]
            pose_l = F.smooth_l1_loss(pn_out, target_pose)
            total = a_eff * seg_l + b_eff * pose_l

        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            frame_0.data.clamp_(0, 255)
            frame_1.data.clamp_(0, 255)
            if avg_last_k > 0 and it >= num_iters - avg_last_k:
                f0_accum += frame_0.data
                f1_accum += frame_1.data
                avg_count += 1

    if avg_last_k > 0:
        final_f0 = f0_accum / avg_count
        final_f1 = f1_accum / avg_count
    else:
        final_f0 = frame_0
        final_f1 = frame_1

    with torch.no_grad():
        f0_up = F.interpolate(final_f0.detach(), size=(H_cam, W_cam),
                              mode='bicubic', align_corners=False).clamp(0, 255).round()
        f1_up = F.interpolate(final_f1.detach(), size=(H_cam, W_cam),
                              mode='bicubic', align_corners=False).clamp(0, 255).round()
    return f0_up.to(torch.uint8), f1_up.to(torch.uint8)


def evaluate(label, **kwargs):
    all_seg, all_pose = [], []
    BS = 16
    t0 = time.time()
    for start in range(0, TEST_PAIRS, BS):
        end = min(start + BS, TEST_PAIRS)
        f0, f1 = custom_optimize(seg_gt[start:end], pose_gt[start:end], **kwargs)
        with torch.inference_mode():
            x = torch.stack([f0.float(), f1.float()], dim=1)
            seg_in = segnet.preprocess_input(x)
            seg_dist = (segnet(seg_in).argmax(1) != seg_gt[start:end]).float().mean().item()
            pn_in = posenet.preprocess_input(x)
            pn_out = posenet(pn_in)['pose'][:, :6]
            pose_dist = ((pn_out - pose_gt[start:end])**2).mean().item()
        all_seg.append(seg_dist)
        all_pose.append(pose_dist)
    dt = time.time() - t0
    s = np.mean(all_seg)
    p = np.mean(all_pose)
    score = 100*s + (10*p)**0.5
    print(f'{label:<55} seg={s:.6f} pose={p:.6f} score={score:.4f} ({dt:.0f}s)')
    sys.stdout.flush()
    return score


print('=== FINE-TUNE avg_last_k ===')
for k in [7, 8, 9, 10, 11, 12, 13, 14]:
    evaluate(f'avg_last_{k}', avg_last_k=k)

print('\n=== COMBOS: avg + noise ===')
evaluate('avg10 + noise5', avg_last_k=10, noise_sigma=5.0)
evaluate('avg10 + noise10', avg_last_k=10, noise_sigma=10.0)
evaluate('avg10 + noise15', avg_last_k=10, noise_sigma=15.0)
evaluate('avg12 + noise10', avg_last_k=12, noise_sigma=10.0)
evaluate('avg8 + noise10', avg_last_k=8, noise_sigma=10.0)

print('\n=== STABILITY CHECK (5 runs) ===')
for i in range(5):
    evaluate(f'avg10 run{i+1}', avg_last_k=10)

print('\nDONE')
