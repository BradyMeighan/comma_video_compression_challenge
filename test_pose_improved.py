#!/usr/bin/env python
"""Test improved geometric pose proxies with SVD SegNet."""
import sys, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn.functional as F, numpy as np
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss

DISTILL_DIR = 'distill_data'
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


def photometric_pose_loss(f0, f1, weight=1.0):
    """Multi-scale photometric + SSIM."""
    loss = 0.0
    a, b = f0, f1
    for scale in range(3):
        diff = (a - b).abs().mean()
        mu_a = F.avg_pool2d(a, 7, stride=1, padding=3)
        mu_b = F.avg_pool2d(b, 7, stride=1, padding=3)
        sigma_ab = F.avg_pool2d(a * b, 7, stride=1, padding=3) - mu_a * mu_b
        sigma_a2 = F.avg_pool2d(a * a, 7, stride=1, padding=3) - mu_a * mu_a
        sigma_b2 = F.avg_pool2d(b * b, 7, stride=1, padding=3) - mu_b * mu_b
        C1, C2 = 0.01**2 * 255**2, 0.03**2 * 255**2
        ssim = ((2*mu_a*mu_b+C1)*(2*sigma_ab+C2)) / ((mu_a**2+mu_b**2+C1)*(sigma_a2+sigma_b2+C2))
        loss += diff + 0.15 * (1 - ssim).mean()
        a = F.avg_pool2d(a, 2)
        b = F.avg_pool2d(b, 2)
    return loss / 3 * weight


def warp_by_pose(f0, target_pose, H=384, W=512):
    """Create a differentiable warp of f0 using the target pose vector.

    target_pose: (B, 6) — [tx, ty, tz, rx, ry, rz]
    For a dashcam, the dominant motion is forward translation (tz) which
    creates an expansion flow pattern from the vanishing point.
    We approximate this as a simple affine transform.
    """
    B = f0.shape[0]
    device = f0.device

    # Extract pose components (small values, typically -0.5 to 0.5)
    tx, ty, tz = target_pose[:, 0], target_pose[:, 1], target_pose[:, 2]
    rx, ry, rz = target_pose[:, 3], target_pose[:, 4], target_pose[:, 5]

    # Scale factors for the warp (empirically tuned)
    # These convert pose units to pixel-level shifts
    scale_t = 0.01  # translation to pixel shift
    scale_r = 0.02  # rotation to pixel rotation

    # Build 2x3 affine matrix for each sample
    # Approximate: rotation + translation
    cos_rz = torch.cos(rz * scale_r)
    sin_rz = torch.sin(rz * scale_r)

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos_rz + tz * scale_t  # scale + rotation
    theta[:, 0, 1] = -sin_rz
    theta[:, 0, 2] = tx * scale_t  # horizontal shift
    theta[:, 1, 0] = sin_rz
    theta[:, 1, 1] = cos_rz + tz * scale_t
    theta[:, 1, 2] = ty * scale_t  # vertical shift

    grid = F.affine_grid(theta, f0.shape, align_corners=False)
    warped = F.grid_sample(f0, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return warped


def run_test(name, svd_seg, pose_fn, teacher_argmax, pose_targets,
             t_seg, t_pose, device, iters=150):
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    starts = [0, 100, 200, 300, 450]
    results = {'ts': [], 'tp': []}

    print(f"\n  === {name} ===")
    for bi, st in enumerate(starts):
        end = min(st + bs, 600)
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_targets[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2, -1), keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters-1, 1))) for i in range(iters)]

        print(f"  Batch {bi+1}/5: ", end='', flush=True)
        for it in range(iters):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            p = it / max(iters-1, 1)
            seg_l = margin_loss(svd_seg(f1), tgt_s, 0.1 if p < 0.5 else 5.0)
            if p >= 0.3:
                pose_l = pose_fn(f0, f1, tgt_p)
                total = 120.0 * seg_l + pose_l  # pose_fn handles its own weighting
            else:
                total = 120.0 * seg_l
            total.backward()
            opt.step()
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
        del f0, f1, opt; torch.cuda.empty_cache()

    avg_ts, avg_tp = np.mean(results['ts']), np.mean(results['tp'])
    s_seg, s_pose = 100 * avg_ts, math.sqrt(10 * avg_tp)
    dist = s_seg + s_pose
    print(f"  RESULT: seg={s_seg:.4f} pose={s_pose:.4f} distortion={dist:.4f}")
    return dist


def main():
    device = torch.device('cuda')

    print("Loading models...")
    svd_seg = SegNet().eval().to(device)
    svd_seg.load_state_dict(torch.load('compressed_models/segnet_svd_finetuned.pt',
                                        weights_only=True, map_location=device))
    for p in svd_seg.parameters(): p.requires_grad_(False)

    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    seg_logits = torch.load(f'{DISTILL_DIR}/seg_logits.pt', weights_only=True)
    ta = seg_logits.argmax(1).to(device)
    pt = torch.load(f'{DISTILL_DIR}/pose_outputs.pt', weights_only=True).to(device)
    del seg_logits

    # Test 1: Photometric with higher weight (2.0 instead of 0.2)
    run_test("Photometric weight=2.0", svd_seg,
             lambda f0, f1, tp: photometric_pose_loss(f0, f1, weight=2.0),
             ta, pt, t_seg, t_pose, device)
    torch.cuda.empty_cache()

    # Test 2: Photometric with weight=5.0
    run_test("Photometric weight=5.0", svd_seg,
             lambda f0, f1, tp: photometric_pose_loss(f0, f1, weight=5.0),
             ta, pt, t_seg, t_pose, device)
    torch.cuda.empty_cache()

    # Test 3: Warp-based (use target pose to warp f0, compare with f1)
    def warp_pose_fn(f0, f1, tgt_p):
        warped_f0 = warp_by_pose(f0, tgt_p)
        return photometric_pose_loss(warped_f0, f1, weight=2.0)

    run_test("Warp + photometric w=2.0", svd_seg, warp_pose_fn,
             ta, pt, t_seg, t_pose, device)
    torch.cuda.empty_cache()

    # Test 4: Combined warp + raw photometric
    def combined_pose_fn(f0, f1, tgt_p):
        warped = warp_by_pose(f0, tgt_p)
        return photometric_pose_loss(warped, f1, weight=1.5) + photometric_pose_loss(f0, f1, weight=1.0)

    run_test("Warp + both photometric", svd_seg, combined_pose_fn,
             ta, pt, t_seg, t_pose, device)

    print(f"\n  Previous best (geometric w=0.2): distortion=8.66")
    print(f"  Full PoseNet (56MB): distortion=1.31")
    print(f"  Leader: 1.95")


if __name__ == '__main__':
    main()
