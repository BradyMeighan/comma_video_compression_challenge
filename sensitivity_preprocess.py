#!/usr/bin/env python
"""
Use model sensitivity maps to guide preprocessing before encoding.

Strategy: Compute PoseNet gradient sensitivity for each frame, then apply
spatially-varying denoising: aggressive blur in low-sensitivity regions
(sky, margins, uniform surfaces), preserve detail in high-sensitivity
regions (lane markings, horizon, vehicle edges).

Unlike our previous ROI attempts that used static masks (sky only, corridor),
this uses MODEL-DERIVED importance maps that precisely match what the models
care about.
"""
import subprocess, sys, os, time, math, zipfile, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file
import einops

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)


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
    y00, y10 = Y[..., 0::2, 0::2], Y[..., 1::2, 0::2]
    y01, y11 = Y[..., 0::2, 1::2], Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)
    return segnet, posenet


def load_gt_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def compute_combined_sensitivity(gt_frames, segnet, posenet):
    """Compute per-pixel combined sensitivity (seg + pose) for each frame."""
    n_pairs = len(gt_frames) // 2
    model_w, model_h = segnet_model_input_size

    all_sens = []
    batch_size = 8

    for bs in range(0, n_pairs, batch_size):
        be = min(bs + batch_size, n_pairs)
        B = be - bs

        pairs = []
        for i in range(bs, be):
            pairs.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
        batch_data = torch.stack(pairs).to(DEVICE)

        # SegNet sensitivity
        seg_input = batch_data.clone().requires_grad_(True)
        x_seg = einops.rearrange(seg_input, 'b t h w c -> b t c h w')
        seg_in = F.interpolate(x_seg[:, -1], size=(model_h, model_w),
                               mode='bilinear', align_corners=False)
        seg_out = segnet(seg_in)
        top2 = seg_out.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).mean()
        margin.backward()
        seg_grad = seg_input.grad.abs().mean(dim=(1, 4)).detach()

        # PoseNet sensitivity
        pose_input = batch_data.clone().requires_grad_(True)
        x_pose = einops.rearrange(pose_input, 'b t h w c -> b t c h w')
        b_sz, s_len = x_pose.shape[:2]
        flat = einops.rearrange(x_pose, 'b t c h w -> (b t) c h w')
        flat = F.interpolate(flat, size=(model_h, model_w), mode='bilinear', align_corners=False)
        yuv = rgb_to_yuv6_diff(flat)
        pn_in = einops.rearrange(yuv, '(b t) c h w -> b (t c) h w', b=b_sz, t=s_len)
        pn_out = posenet(pn_in)['pose'][:, :6]
        pn_out.pow(2).sum().backward()
        pose_grad = pose_input.grad.abs().mean(dim=(1, 4)).detach()

        # Normalize each to [0, 1] individually, then combine
        seg_norm = seg_grad / (seg_grad.max() + 1e-10)
        pose_norm = pose_grad / (pose_grad.max() + 1e-10)
        combined = 0.5 * seg_norm + 0.5 * pose_norm

        # We want per-frame sensitivity (not per-pair).
        # Expand pair-level to frame-level by duplicating.
        for i in range(B):
            pair_sens = combined[i].cpu().numpy()
            all_sens.append(pair_sens)
            all_sens.append(pair_sens)

        if (bs // batch_size) % 15 == 0:
            print(f"  Batch {bs//batch_size + 1}/{(n_pairs+batch_size-1)//batch_size}",
                  flush=True)

    return all_sens  # list of 1200 numpy arrays (H, W)


def sensitivity_guided_blur(frames_np, sensitivity_maps,
                             blur_sigma=5.0, blur_threshold=0.3):
    """Apply spatially-varying blur guided by sensitivity maps.

    Low-sensitivity pixels get blurred, high-sensitivity pixels are preserved.
    """
    result = []
    ksize = int(blur_sigma * 6) | 1

    for i, (frame, sens) in enumerate(zip(frames_np, sensitivity_maps)):
        # Smooth the sensitivity map to avoid sharp mask transitions
        sens_smooth = cv2.GaussianBlur(sens, (31, 31), 8.0)

        # Create alpha mask: 1 = keep original, 0 = use blurred
        alpha = np.clip(sens_smooth / (blur_threshold + 1e-8), 0, 1)
        alpha = alpha[..., np.newaxis]  # H, W, 1

        blurred = cv2.GaussianBlur(frame, (ksize, ksize), blur_sigma)
        result.append((frame * alpha + blurred * (1 - alpha)).astype(np.uint8))

    return result


def encode_piped(frames_np, out_mkv, crf=33, scale=0.45, film_grain=22,
                 preset=0, keyint=180):
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', f'film-grain={film_grain}:keyint={keyint}:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames_np:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size if proc.returncode == 0 else None


def decode_and_inflate(mkv_path, strength=0.45):
    import av
    from PIL import Image
    container = av.open(str(mkv_path))
    raw_bytes = bytearray()
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL_9, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())
    container.close()
    return raw_bytes


def fast_eval(raw_bytes, archive_size, segnet, posenet):
    gt_cache = ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt'
    gt = torch.load(gt_cache, weights_only=True)
    N = gt['seg'].shape[0]
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)

    seg_dists, pose_dists = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, 'b t h w c -> b t c h w')
            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            gt_seg = gt['seg'][i:end].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())
            pn_out = posenet(posenet.preprocess_input(x))['pose'][:, :6]
            gt_pose = gt['pose'][i:end].to(DEVICE)
            pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())

    s = np.mean(seg_dists)
    p = np.mean(pose_dists)
    r = archive_size / 37_545_489
    return 100*s + math.sqrt(10*p) + 25*r, s, p, r


def run_test(name, frames_np, segnet, posenet, crf=33, preset=0, unsharp=0.45):
    tmp_dir = ROOT / '_sens_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'

    t0 = time.time()
    encode_piped(frames_np, out_mkv, crf=crf, preset=preset)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size
    raw = decode_and_inflate(out_mkv, strength=unsharp)
    score, s, p, r = fast_eval(raw, archive_size, segnet, posenet)

    print(f"[{name}] score={score:.4f} seg={100*s:.4f} pose={math.sqrt(10*p):.4f} "
          f"rate={25*r:.4f} size={archive_size/1024:.1f}KB ({time.time()-t0:.0f}s)",
          flush=True)

    out_mkv.unlink(missing_ok=True)
    archive_zip.unlink(missing_ok=True)
    return score


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)

    # Step 1: Compute sensitivity maps
    print("\n=== Computing sensitivity maps ===", flush=True)
    t0 = time.time()
    sensitivity_maps = compute_combined_sensitivity(gt_frames, segnet, posenet)
    print(f"  Computed in {time.time()-t0:.0f}s", flush=True)

    # Save sensitivity statistics
    avg_sens = np.mean([s.mean() for s in sensitivity_maps])
    max_sens = np.max([s.max() for s in sensitivity_maps])
    print(f"  Avg sensitivity: {avg_sens:.6f}, Max: {max_sens:.6f}", flush=True)

    frames_np = [f.numpy() for f in gt_frames]

    # Step 2: Test baseline (no preprocessing)
    print("\n=== Baseline (no preprocess) ===", flush=True)
    score_none = run_test("no_preproc", frames_np, segnet, posenet)

    # Step 3: Test sky blur baseline
    sky_mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * 0.15)
    for y in range(sky_end):
        sky_mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * 0.03)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        sky_mask[:, x] = np.minimum(sky_mask[:, x], t)
        sky_mask[:, W_CAM - 1 - x] = np.minimum(sky_mask[:, W_CAM - 1 - x], t)
    sky_mask = cv2.GaussianBlur(sky_mask, (31, 31), 10)
    sky_alpha = sky_mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    sky_frames = [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - sky_alpha)
                   ).astype(np.uint8) for f in frames_np]
    print("\n=== Sky blur baseline ===", flush=True)
    score_sky = run_test("sky_blur", sky_frames, segnet, posenet)

    # Step 4: Test sensitivity-guided blur
    configs = [
        # (name, blur_sigma, threshold)
        ("sens_s3_t0.2", 3.0, 0.2),
        ("sens_s3_t0.3", 3.0, 0.3),
        ("sens_s3_t0.5", 3.0, 0.5),
        ("sens_s5_t0.2", 5.0, 0.2),
        ("sens_s5_t0.3", 5.0, 0.3),
        ("sens_s5_t0.5", 5.0, 0.5),
        ("sens_s7_t0.3", 7.0, 0.3),
        ("sens_s7_t0.5", 7.0, 0.5),
    ]

    print("\n=== Sensitivity-guided blur tests ===", flush=True)
    best_score = min(score_none, score_sky)
    best_name = "sky_blur" if score_sky < score_none else "no_preproc"

    for name, sigma, threshold in configs:
        blurred = sensitivity_guided_blur(frames_np, sensitivity_maps,
                                           blur_sigma=sigma, blur_threshold=threshold)
        score = run_test(name, blurred, segnet, posenet)
        if score < best_score:
            best_score = score
            best_name = name

    # Step 5: Test combining sensitivity blur with sky blur
    print("\n=== Combined: sensitivity + sky blur ===", flush=True)
    for sigma, threshold in [(3.0, 0.3), (5.0, 0.3), (3.0, 0.5)]:
        name = f"sens+sky_s{sigma}_t{threshold}"
        # First apply sensitivity blur, then sky blur on top
        sens_frames = sensitivity_guided_blur(frames_np, sensitivity_maps,
                                               blur_sigma=sigma, blur_threshold=threshold)
        combined = [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - sky_alpha)
                     ).astype(np.uint8) for f in sens_frames]
        score = run_test(name, combined, segnet, posenet)
        if score < best_score:
            best_score = score
            best_name = name

    # Step 6: Test per-frame binary importance mask from sensitivity
    print("\n=== Binary importance mask from sensitivity ===", flush=True)
    for percentile in [25, 50, 75]:
        name = f"binary_p{percentile}_s5"
        processed = []
        for frame, sens in zip(frames_np, sensitivity_maps):
            thresh = np.percentile(sens, percentile)
            mask = (sens > thresh).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (15, 15), 3.0)[..., np.newaxis]
            blurred = cv2.GaussianBlur(frame, (ksize, ksize), 5.0)
            processed.append((frame * mask + blurred * (1 - mask)).astype(np.uint8))
        score = run_test(name, processed, segnet, posenet)
        if score < best_score:
            best_score = score
            best_name = name

    print(f"\n{'='*60}")
    print(f"BEST: {best_name} = {best_score:.4f}")
    print(f"Sky blur baseline: {score_sky:.4f}")
    print(f"No preprocess: {score_none:.4f}")
    print(f"{'='*60}")

    shutil.rmtree(ROOT / '_sens_tmp', ignore_errors=True)
