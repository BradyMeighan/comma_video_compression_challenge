#!/usr/bin/env python
"""
Iterative encode optimization: identify worst-performing frame pairs
and apply targeted preprocessing, then re-encode.

Strategy:
1. Encode with current best settings
2. Decode + inflate
3. Evaluate per-pair distortion
4. For the worst pairs, try different preprocessing (more/less blur,
   different unsharp, spatial adjustments)
5. Re-encode with per-frame optimized preprocessing
6. Repeat

Also tests: storing compact per-frame correction data in archive.
A small quantized delta map (~few KB) that adjusts pixel values
at decode time could help.
"""
import subprocess, sys, os, time, zipfile, shutil, struct, zlib
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
    frames = []
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
        frames.append(x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round()
                      .cpu().to(torch.uint8).numpy())
    container.close()
    return frames


def evaluate_per_pair(inflated, gt_frames, segnet, posenet):
    """Return per-pair seg and pose scores."""
    n_pairs = len(gt_frames) // 2
    seg_scores = []
    pose_scores = []
    
    for bs in range(0, n_pairs, 16):
        be = min(bs + 16, n_pairs)
        gt_b = []
        comp_b = []
        for i in range(bs, be):
            gt_b.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
            comp_b.append(torch.stack([
                torch.from_numpy(inflated[i*2]).float(),
                torch.from_numpy(inflated[i*2+1]).float()
            ]))
        
        gt_b = torch.stack(gt_b).to(DEVICE)
        comp_b = torch.stack(comp_b).to(DEVICE)
        gt_x = einops.rearrange(gt_b, 'b t h w c -> b t c h w')
        comp_x = einops.rearrange(comp_b, 'b t h w c -> b t c h w')
        
        with torch.inference_mode():
            gs = segnet(segnet.preprocess_input(gt_x)).argmax(dim=1)
            cs = segnet(segnet.preprocess_input(comp_x)).argmax(dim=1)
            seg_scores.extend((gs != cs).float().mean(dim=(1,2)).cpu().tolist())
            
            gp = posenet(posenet.preprocess_input(gt_x))['pose'][:, :6]
            cp = posenet(posenet.preprocess_input(comp_x))['pose'][:, :6]
            pose_scores.extend(((gp - cp)**2).mean(dim=1).cpu().tolist())
    
    return np.array(seg_scores), np.array(pose_scores)


def total_score(seg_arr, pose_arr, archive_size):
    s = seg_arr.mean()
    p = pose_arr.mean()
    r = archive_size / 37_545_489
    return 100*s + np.sqrt(10*p) + 25*r


def sky_blur_frame(frame_np, sky_alpha, sigma=5.0):
    ksize = int(sigma * 6) | 1
    blurred = cv2.GaussianBlur(frame_np, (ksize, ksize), sigma)
    return (frame_np * sky_alpha + blurred * (1 - sky_alpha)).astype(np.uint8)


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)
    
    tmp_dir = ROOT / '_iter_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'
    
    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    
    # Step 1: Baseline encode
    print("\n=== Step 1: Baseline encode (sky blur) ===", flush=True)
    frames_base = [sky_blur_frame(f.numpy(), sky_alpha) for f in gt_frames]
    
    t0 = time.time()
    encode_piped(frames_base, out_mkv, preset=0)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size
    inflated = decode_and_inflate(out_mkv)
    seg_base, pose_base = evaluate_per_pair(inflated, gt_frames, segnet, posenet)
    score_base = total_score(seg_base, pose_base, archive_size)
    print(f"  Baseline: score={score_base:.4f} "
          f"seg={100*seg_base.mean():.4f} pose={np.sqrt(10*pose_base.mean()):.4f} "
          f"rate={25*archive_size/37545489:.4f} ({time.time()-t0:.0f}s)", flush=True)
    
    # Step 2: Analyze per-pair scores
    print("\n=== Step 2: Per-pair analysis ===", flush=True)
    worst_seg_pairs = np.argsort(seg_base)[-20:]
    worst_pose_pairs = np.argsort(pose_base)[-20:]
    
    print(f"  Worst seg pairs (top 20): "
          f"mean={seg_base[worst_seg_pairs].mean():.6f} "
          f"vs overall {seg_base.mean():.6f}", flush=True)
    print(f"  Worst pose pairs (top 20): "
          f"mean={pose_base[worst_pose_pairs].mean():.6f} "
          f"vs overall {pose_base.mean():.6f}", flush=True)
    
    worst_combined = np.argsort(100*seg_base + np.sqrt(10*pose_base))[-20:]
    print(f"  Worst combined pairs: indices={worst_combined.tolist()}", flush=True)
    
    for i in worst_combined[-5:]:
        print(f"    Pair {i}: seg={seg_base[i]:.6f} pose={pose_base[i]:.6f} "
              f"combined={100*seg_base[i] + np.sqrt(10*pose_base[i]):.4f} "
              f"frames={i*2},{i*2+1}", flush=True)
    
    # Step 3: Try per-frame adaptive preprocessing
    # For worst frames, try different blur/no-blur strategies
    print("\n=== Step 3: Adaptive preprocessing ===", flush=True)
    
    # Idea: for the worst pairs, DON'T apply sky blur (preserve all detail)
    # For good pairs, apply MORE blur (save bits for important frames)
    n_pairs = len(gt_frames) // 2
    pair_importance = 100 * seg_base + np.sqrt(10 * pose_base)
    
    # Adaptive sky blur: high importance = less blur, low importance = more blur
    # Normalize importance to [0, 1]
    imp_norm = (pair_importance - pair_importance.min()) / (pair_importance.max() - pair_importance.min() + 1e-8)
    
    configs_to_test = [
        # (name, sigma_for_worst, sigma_for_best, threshold_percentile)
        ("adaptive_s0_s7_p75", 0.0, 7.0, 75),
        ("adaptive_s2_s7_p75", 2.0, 7.0, 75),
        ("adaptive_s3_s8_p50", 3.0, 8.0, 50),
        ("adaptive_s0_s5_p90", 0.0, 5.0, 90),
        ("no_sky_blur", 0.0, 0.0, 0),
        ("more_sky_blur_s8", 8.0, 8.0, 0),
        ("wider_sky_top20", None, None, None),  # Special: wider sky mask
        ("gentle_bilateral", None, None, None),  # Special: bilateral instead of Gaussian
    ]
    
    results = {}
    
    for name, sigma_worst, sigma_best, threshold_pct in configs_to_test:
        print(f"\n  Testing: {name}...", flush=True)
        t0 = time.time()
        
        if name == "wider_sky_top20":
            wider_mask = make_sky_mask(top_frac=0.20, side_frac=0.04)
            wider_alpha = wider_mask[..., np.newaxis]
            frames = [sky_blur_frame(f.numpy(), wider_alpha, sigma=5.0) for f in gt_frames]
        elif name == "gentle_bilateral":
            frames = []
            for f in gt_frames:
                fn = f.numpy()
                sm = sky_mask
                top_region = (sm < 0.5).astype(np.uint8)
                if top_region.any():
                    bilateral = cv2.bilateralFilter(fn, 5, 20, 20)
                    alpha3 = sm[..., np.newaxis]
                    frames.append((fn * alpha3 + bilateral * (1 - alpha3)).astype(np.uint8))
                else:
                    frames.append(fn)
        else:
            frames = []
            threshold = np.percentile(pair_importance, threshold_pct) if threshold_pct > 0 else float('inf')
            for pair_i in range(n_pairs):
                is_important = pair_importance[pair_i] >= threshold
                sigma = sigma_worst if is_important else sigma_best
                
                for fi in [pair_i*2, pair_i*2+1]:
                    fn = gt_frames[fi].numpy()
                    if sigma > 0:
                        frames.append(sky_blur_frame(fn, sky_alpha, sigma=sigma))
                    else:
                        frames.append(fn)
        
        encode_piped(frames, out_mkv, preset=0)
        with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(out_mkv, '0.mkv')
        a_size = archive_zip.stat().st_size
        infl = decode_and_inflate(out_mkv)
        seg_s, pose_s = evaluate_per_pair(infl, gt_frames, segnet, posenet)
        score = total_score(seg_s, pose_s, a_size)
        delta = score - score_base
        
        print(f"  [{name}] score={score:.4f} (delta={delta:+.4f}) "
              f"seg={100*seg_s.mean():.4f} pose={np.sqrt(10*pose_s.mean()):.4f} "
              f"rate={25*a_size/37545489:.4f} size={a_size/1024:.1f}KB "
              f"({time.time()-t0:.0f}s)", flush=True)
        results[name] = score
    
    print("\n" + "="*70)
    print("SUMMARY:")
    print("="*70)
    print(f"  baseline:    {score_base:.4f}")
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        delta = score - score_base
        print(f"  {name:30s} {score:.4f} (delta={delta:+.4f})")
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
