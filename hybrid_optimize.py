#!/usr/bin/env python
"""
Hybrid optimization: model-in-the-loop encode-time optimization.

Strategy: Use SegNet/PoseNet to identify which frames have highest distortion
after compression, then apply targeted preprocessing to those frames and
re-encode. Also tests decode-side improvements.

Key ideas:
1. Edge-aware spatially-varying unsharp at decode
2. Per-frame quality analysis to guide preprocessing
3. Bilateral filter at decode (edge-preserving denoising)
4. Different Lanczos lobe sizes
5. Combined sky blur + adaptive unsharp
"""
import subprocess, sys, os, time, struct, zipfile, shutil
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

# 9-tap binomial kernel
_r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

# 11-tap binomial kernel
_r11 = torch.tensor([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
KERNEL_11 = (torch.outer(_r11, _r11) / (_r11.sum()**2)).to(DEVICE).expand(3, 1, 11, 11)

# 7-tap binomial kernel
_r7 = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
KERNEL_7 = (torch.outer(_r7, _r7) / (_r7.sum()**2)).to(DEVICE).expand(3, 1, 7, 7)


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
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame))
    container.close()
    return frames


def make_sky_mask(top_frac=0.15, side_frac=0.03):
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * top_frac)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * side_frac)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t)
    return cv2.GaussianBlur(mask, (31, 31), 10)


def encode_piped(frames_np, out_mkv, crf=33, scale=0.45, film_grain=22,
                 preset=4, keyint=180):
    """Encode preprocessed frames via pipe to ffmpeg."""
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
        '-r', '20',
        str(out_mkv),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames_np:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        print(f"  ENCODE FAILED: {proc.stderr.read().decode()[:300]}")
        return None
    return out_mkv.stat().st_size


def decode_frames(mkv_path):
    """Decode MKV to list of uint8 numpy arrays (H, W, 3)."""
    import av
    container = av.open(str(mkv_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame).numpy())
    container.close()
    return frames


def apply_inflate(decoded_frames, strength=0.45, kernel=KERNEL_9, pad=4):
    """Lanczos upscale + binomial unsharp."""
    from PIL import Image
    result = []
    for frame_np in decoded_frames:
        h, w, _ = frame_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(frame_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            frame_np = np.array(pil)

        x = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                        kernel, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        result.append(t.numpy())
    return result


def apply_edge_aware_unsharp(decoded_frames, base_strength=0.35, edge_strength=0.70,
                              kernel=KERNEL_9, pad=4):
    """Spatially-varying unsharp: stronger at edges, weaker in flat regions."""
    from PIL import Image
    result = []
    for frame_np in decoded_frames:
        h, w, _ = frame_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(frame_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            frame_np = np.array(pil)

        gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = cv2.GaussianBlur(edges.astype(np.float32) / 255.0,
                                      (7, 7), 1.5)
        strength_map = base_strength + (edge_strength - base_strength) * edge_mask
        strength_map = torch.from_numpy(strength_map).to(DEVICE).unsqueeze(0).unsqueeze(0)

        x = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                        kernel, padding=0, groups=3)
        x = x + strength_map * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        result.append(t.numpy())
    return result


def apply_bilateral_postfilter(inflated_frames, d=5, sigma_color=30, sigma_space=30):
    """Apply bilateral filter after inflate to remove compression artifacts."""
    result = []
    for frame_np in inflated_frames:
        filtered = cv2.bilateralFilter(frame_np, d, sigma_color, sigma_space)
        result.append(filtered)
    return result


def evaluate_frames(inflated_frames, gt_frames, segnet, posenet, archive_size):
    """Evaluate inflated frames against ground truth using SegNet + PoseNet."""
    model_w, model_h = segnet_model_input_size  # (512, 384)
    n_pairs = len(gt_frames) // 2
    seg_losses = []
    pose_losses = []

    batch_size = 16
    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        B = batch_end - batch_start

        gt_batch = []
        comp_batch = []
        for i in range(batch_start, batch_end):
            f0_idx = i * 2
            f1_idx = i * 2 + 1

            gt_pair = torch.stack([gt_frames[f0_idx].float(),
                                   gt_frames[f1_idx].float()])
            comp_pair = torch.stack([
                torch.from_numpy(inflated_frames[f0_idx]).float(),
                torch.from_numpy(inflated_frames[f1_idx]).float()
            ])
            gt_batch.append(gt_pair)
            comp_batch.append(comp_pair)

        gt_batch = torch.stack(gt_batch).to(DEVICE)
        comp_batch = torch.stack(comp_batch).to(DEVICE)

        gt_x = einops.rearrange(gt_batch, 'b t h w c -> b t c h w')
        comp_x = einops.rearrange(comp_batch, 'b t h w c -> b t c h w')

        with torch.inference_mode():
            gt_seg_in = segnet.preprocess_input(gt_x)
            gt_seg_out = segnet(gt_seg_in)
            gt_seg_map = gt_seg_out.argmax(dim=1)

            comp_seg_in = segnet.preprocess_input(comp_x)
            comp_seg_out = segnet(comp_seg_in)
            comp_seg_map = comp_seg_out.argmax(dim=1)

            seg_mismatch = (gt_seg_map != comp_seg_map).float().mean(dim=(1, 2))
            seg_losses.extend(seg_mismatch.cpu().tolist())

            gt_pn_in = posenet.preprocess_input(gt_x)
            gt_pose = posenet(gt_pn_in)['pose'][:, :6]

            comp_pn_in = posenet.preprocess_input(comp_x)
            comp_pose = posenet(comp_pn_in)['pose'][:, :6]

            pose_mse = ((gt_pose - comp_pose) ** 2).mean(dim=1)
            pose_losses.extend(pose_mse.cpu().tolist())

    avg_seg = np.mean(seg_losses)
    avg_pose = np.mean(pose_losses)
    rate = archive_size / 37_545_489

    score = 100 * avg_seg + np.sqrt(10 * avg_pose) + 25 * rate
    return {
        'seg': avg_seg,
        'pose': avg_pose,
        'rate': rate,
        'score': score,
        'seg_term': 100 * avg_seg,
        'pose_term': np.sqrt(10 * avg_pose),
        'rate_term': 25 * rate,
    }


def run_experiment(name, preprocess_fn, inflate_fn, crf=33, scale=0.45,
                   film_grain=22, preset=4, keyint=180):
    """Run a complete compress→inflate→evaluate experiment."""
    tmp_dir = ROOT / '_hybrid_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'

    t0 = time.time()

    gt_raw = load_gt_frames()
    frames_np = [f.numpy() for f in gt_raw]

    if preprocess_fn is not None:
        frames_np = preprocess_fn(frames_np)

    mkv_size = encode_piped(frames_np, out_mkv, crf=crf, scale=scale,
                            film_grain=film_grain, preset=preset, keyint=keyint)
    if mkv_size is None:
        return None

    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size

    decoded = decode_frames(out_mkv)
    inflated = inflate_fn(decoded)

    result = evaluate_frames(inflated, gt_raw, segnet, posenet, archive_size)
    elapsed = time.time() - t0

    print(f"[{name}] score={result['score']:.4f} "
          f"seg={result['seg_term']:.4f} pose={result['pose_term']:.4f} "
          f"rate={result['rate_term']:.4f} size={archive_size/1024:.1f}KB "
          f"({elapsed:.1f}s)", flush=True)

    # Cleanup
    out_mkv.unlink(missing_ok=True)
    archive_zip.unlink(missing_ok=True)

    return result


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    print("Loading models...", flush=True)
    segnet, posenet = load_models()
    print("Models loaded.", flush=True)

    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]

    def sky_blur(frames, sigma=5.0):
        ksize = int(sigma * 6) | 1
        result = []
        for f in frames:
            blurred = cv2.GaussianBlur(f, (ksize, ksize), sigma)
            result.append((f * sky_alpha + blurred * (1 - sky_alpha)).astype(np.uint8))
        return result

    def uniform_unsharp(strength):
        def fn(decoded):
            return apply_inflate(decoded, strength=strength)
        return fn

    def edge_aware_fn(base, edge):
        def fn(decoded):
            return apply_edge_aware_unsharp(decoded, base_strength=base,
                                           edge_strength=edge)
        return fn

    def bilateral_post(d, sc, ss):
        def fn(decoded):
            inflated = apply_inflate(decoded, strength=0.45)
            return apply_bilateral_postfilter(inflated, d=d, sigma_color=sc, sigma_space=ss)
        return fn

    def unsharp_then_bilateral(ush_str, d, sc, ss):
        def fn(decoded):
            inflated = apply_inflate(decoded, strength=ush_str)
            return apply_bilateral_postfilter(inflated, d=d, sigma_color=sc, sigma_space=ss)
        return fn

    def kernel11_unsharp(strength):
        def fn(decoded):
            return apply_inflate(decoded, strength=strength, kernel=KERNEL_11, pad=5)
        return fn

    def kernel7_unsharp(strength):
        def fn(decoded):
            return apply_inflate(decoded, strength=strength, kernel=KERNEL_7, pad=3)
        return fn

    results = {}

    experiments = [
        # Baseline: sky blur + standard unsharp
        ("baseline_sky", lambda f: sky_blur(f), uniform_unsharp(0.45), 33),

        # Edge-aware unsharp variations
        ("edge_0.30_0.65", lambda f: sky_blur(f), edge_aware_fn(0.30, 0.65), 33),
        ("edge_0.35_0.70", lambda f: sky_blur(f), edge_aware_fn(0.35, 0.70), 33),
        ("edge_0.30_0.80", lambda f: sky_blur(f), edge_aware_fn(0.30, 0.80), 33),
        ("edge_0.20_0.60", lambda f: sky_blur(f), edge_aware_fn(0.20, 0.60), 33),
        ("edge_0.40_0.60", lambda f: sky_blur(f), edge_aware_fn(0.40, 0.60), 33),

        # Bilateral post-filter after unsharp
        ("bilateral_5_30_30", lambda f: sky_blur(f), bilateral_post(5, 30, 30), 33),
        ("bilateral_5_20_20", lambda f: sky_blur(f), bilateral_post(5, 20, 20), 33),
        ("bilateral_3_15_15", lambda f: sky_blur(f), bilateral_post(3, 15, 15), 33),

        # Unsharp + mild bilateral
        ("ush45_bil3_10", lambda f: sky_blur(f), unsharp_then_bilateral(0.45, 3, 10, 10), 33),
        ("ush50_bil3_10", lambda f: sky_blur(f), unsharp_then_bilateral(0.50, 3, 10, 10), 33),

        # Different kernel sizes
        ("kernel11_0.45", lambda f: sky_blur(f), kernel11_unsharp(0.45), 33),
        ("kernel11_0.50", lambda f: sky_blur(f), kernel11_unsharp(0.50), 33),
        ("kernel7_0.45", lambda f: sky_blur(f), kernel7_unsharp(0.45), 33),
        ("kernel7_0.50", lambda f: sky_blur(f), kernel7_unsharp(0.50), 33),

        # Higher unsharp
        ("ush_0.55", lambda f: sky_blur(f), uniform_unsharp(0.55), 33),
        ("ush_0.60", lambda f: sky_blur(f), uniform_unsharp(0.60), 33),

        # Sky blur parameter tuning
        ("sky_s3_ush45", lambda f: sky_blur(f, sigma=3.0), uniform_unsharp(0.45), 33),
        ("sky_s7_ush45", lambda f: sky_blur(f, sigma=7.0), uniform_unsharp(0.45), 33),

        # CRF fine-tuning with sky blur
        ("crf32_sky", lambda f: sky_blur(f), uniform_unsharp(0.45), 32),
        ("crf34_sky", lambda f: sky_blur(f), uniform_unsharp(0.45), 34),

        # No preprocessing baseline (piped RGB only)
        ("no_preproc", None, uniform_unsharp(0.45), 33),
    ]

    for name, preproc, inflate, crf in experiments:
        try:
            result = run_experiment(name, preproc, inflate, crf=crf, preset=4)
            if result:
                results[name] = result
        except Exception as e:
            print(f"[{name}] ERROR: {e}", flush=True)

    print("\n" + "=" * 80)
    print("SUMMARY (sorted by score):")
    print("=" * 80)
    for name, r in sorted(results.items(), key=lambda x: x[1]['score']):
        print(f"  {name:25s} score={r['score']:.4f} "
              f"seg={r['seg_term']:.4f} pose={r['pose_term']:.4f} "
              f"rate={r['rate_term']:.4f}")

    # Cleanup
    tmp_dir = ROOT / '_hybrid_tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
