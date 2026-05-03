#!/usr/bin/env python
"""
Adversarial encode: optimize video frames BEFORE compression so that
after compression → decode → upscale → unsharp, the models produce
outputs closer to the ground truth outputs.

Strategy:
1. Load ground truth frames, compute GT model outputs
2. Apply small perturbations to frames via gradient descent
3. The perturbations should be "compression-friendly" — they should
   survive the encode/decode cycle
4. Encode the perturbed frames
5. Evaluate to verify improvement

Key insight: Instead of trying to preserve the original signal perfectly,
we modify the signal so that even after lossy compression, the MODEL
outputs are preserved. This is different from traditional preprocessing
which tries to preserve PIXEL values.

Approach A: "Model-aware denoising"
- For each frame, compute grad of model loss w.r.t. input pixels
- Apply small adjustments in the direction that reduces model loss
- The adjustments are applied to the original frames before encoding

Approach B: "Compression-aware training"
- Simulate the full pipeline: preprocess → encode → decode → inflate → model
- Optimize preprocessing parameters end-to-end
- Since encoding is not differentiable, use finite differences or
  a differentiable proxy for the encoder
"""
import subprocess, sys, os, time, zipfile, shutil
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
    frames = []
    for frame in container.decode(container.streams.video[0]):
        frames.append(yuv420_to_rgb(frame))
    container.close()
    return frames


def make_sky_mask():
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * 0.15)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * 0.03)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t)
    return cv2.GaussianBlur(mask, (31, 31), 10)


def simulate_inflate(frames_chw, scale=0.45):
    """Differentiable approximation of the encode→decode→inflate pipeline.
    
    Simulates: downscale → quantize/noise → upscale → unsharp.
    Not a perfect proxy for actual AV1 but captures the main effects.
    """
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    
    down = F.interpolate(frames_chw, size=(h, w), mode='bilinear', align_corners=False)
    
    if frames_chw.requires_grad:
        noise = torch.randn_like(down) * 2.0
        down = down + noise
    
    up = F.interpolate(down, size=(H_CAM, W_CAM), mode='bilinear', align_corners=False)
    
    blur = F.conv2d(F.pad(up, (4, 4, 4, 4), mode='reflect'), KERNEL_9, padding=0, groups=3)
    sharpened = up + 0.45 * (up - blur)
    
    return sharpened.clamp(0, 255)


def compute_gt_targets(gt_frames, segnet, posenet):
    """Compute ground truth model outputs for all frame pairs."""
    n_pairs = len(gt_frames) // 2
    gt_seg_maps = []
    gt_pose_vecs = []
    
    batch_size = 16
    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        
        pairs = []
        for i in range(batch_start, batch_end):
            pair = torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()])
            pairs.append(pair)
        
        batch = torch.stack(pairs).to(DEVICE)
        x = einops.rearrange(batch, 'b t h w c -> b t c h w')
        
        with torch.inference_mode():
            seg_in = segnet.preprocess_input(x)
            seg_out = segnet(seg_in)
            gt_seg_maps.append(seg_out.argmax(dim=1).cpu())
            
            pn_in = posenet.preprocess_input(x)
            pose = posenet(pn_in)['pose'][:, :6]
            gt_pose_vecs.append(pose.cpu())
    
    return torch.cat(gt_seg_maps), torch.cat(gt_pose_vecs)


def adversarial_preprocess(gt_frames, segnet, posenet,
                            num_iters=5, lr=0.5, eps=4.0):
    """Apply model-guided perturbations to frames before encoding.
    
    For each pair of frames, compute the gradient of model distortion
    w.r.t. input pixels after simulated compression, and apply small
    corrections to reduce expected post-compression distortion.
    """
    print(f"Computing GT targets...", flush=True)
    gt_seg, gt_pose = compute_gt_targets(gt_frames, segnet, posenet)
    
    n_pairs = len(gt_frames) // 2
    optimized = [f.numpy().copy() for f in gt_frames]
    
    batch_size = 8  # Smaller batch for gradient computation
    total_seg_improve = 0
    total_pose_improve = 0
    
    for batch_start in range(0, n_pairs, batch_size):
        batch_end = min(batch_start + batch_size, n_pairs)
        B = batch_end - batch_start
        
        # Load frame pairs as tensors
        f0_list = []
        f1_list = []
        for i in range(batch_start, batch_end):
            f0_list.append(torch.from_numpy(optimized[i*2]).float())
            f1_list.append(torch.from_numpy(optimized[i*2+1]).float())
        
        f0_batch = torch.stack(f0_list).to(DEVICE).permute(0, 3, 1, 2)  # B, 3, H, W
        f1_batch = torch.stack(f1_list).to(DEVICE).permute(0, 3, 1, 2)
        
        target_seg = gt_seg[batch_start:batch_end].to(DEVICE)
        target_pose = gt_pose[batch_start:batch_end].to(DEVICE)
        
        # Only optimize f1 (SegNet uses last frame, PoseNet uses pair)
        delta = torch.zeros_like(f1_batch, requires_grad=True)
        
        for it in range(num_iters):
            perturbed_f1 = (f1_batch + delta).clamp(0, 255)
            
            # Simulate compression effect
            sim_f1 = simulate_inflate(perturbed_f1)
            sim_f0 = simulate_inflate(f0_batch)
            
            # SegNet loss on f1
            model_w, model_h = segnet_model_input_size
            seg_in = F.interpolate(sim_f1, size=(model_h, model_w), mode='bilinear', align_corners=False) / 255.0
            seg_out = segnet(seg_in)
            seg_loss = F.cross_entropy(seg_out, target_seg)
            
            # PoseNet loss on (f0, f1) pair
            both = torch.stack([sim_f0, sim_f1], dim=1)  # B, 2, 3, H, W
            # PoseNet expects (B, T*6, H/2, W/2) after YUV conversion
            # Use the model's preprocess
            both_hwc = both.permute(0, 1, 3, 4, 2)  # B, T, H, W, C
            pn_in = posenet.preprocess_input(both_hwc)
            pose_out = posenet(pn_in)['pose'][:, :6]
            pose_loss = F.mse_loss(pose_out, target_pose)
            
            total_loss = 100 * seg_loss + 10 * pose_loss
            total_loss.backward()
            
            with torch.no_grad():
                delta.data -= lr * delta.grad.sign()
                delta.data.clamp_(-eps, eps)
                delta.grad.zero_()
        
        # Apply the optimized delta
        with torch.no_grad():
            final_f1 = (f1_batch + delta).clamp(0, 255).round()
            
            for i in range(B):
                pair_idx = batch_start + i
                optimized[pair_idx*2+1] = final_f1[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        if (batch_start // batch_size) % 5 == 0:
            print(f"  Batch {batch_start//batch_size + 1}/{(n_pairs+batch_size-1)//batch_size} "
                  f"seg_loss={seg_loss.item():.4f} pose_loss={pose_loss.item():.6f} "
                  f"delta_mean={delta.abs().mean().item():.2f}", flush=True)
    
    return optimized


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
        '-r', '20',
        str(out_mkv),
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
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        frames.append(t.numpy())
    container.close()
    return frames


def evaluate(inflated, gt_frames, segnet, posenet, archive_size):
    n_pairs = len(gt_frames) // 2
    seg_losses = []
    pose_losses = []
    batch_size = 16
    
    for bs in range(0, n_pairs, batch_size):
        be = min(bs + batch_size, n_pairs)
        gt_batch = []
        comp_batch = []
        for i in range(bs, be):
            gt_batch.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
            comp_batch.append(torch.stack([
                torch.from_numpy(inflated[i*2]).float(),
                torch.from_numpy(inflated[i*2+1]).float()
            ]))
        
        gt_batch = torch.stack(gt_batch).to(DEVICE)
        comp_batch = torch.stack(comp_batch).to(DEVICE)
        gt_x = einops.rearrange(gt_batch, 'b t h w c -> b t c h w')
        comp_x = einops.rearrange(comp_batch, 'b t h w c -> b t c h w')
        
        with torch.inference_mode():
            gt_seg = segnet(segnet.preprocess_input(gt_x)).argmax(dim=1)
            comp_seg = segnet(segnet.preprocess_input(comp_x)).argmax(dim=1)
            seg_losses.extend((gt_seg != comp_seg).float().mean(dim=(1,2)).cpu().tolist())
            
            gt_pose = posenet(posenet.preprocess_input(gt_x))['pose'][:, :6]
            comp_pose = posenet(posenet.preprocess_input(comp_x))['pose'][:, :6]
            pose_losses.extend(((gt_pose - comp_pose)**2).mean(dim=1).cpu().tolist())
    
    seg = np.mean(seg_losses)
    pose = np.mean(pose_losses)
    rate = archive_size / 37_545_489
    score = 100*seg + np.sqrt(10*pose) + 25*rate
    print(f"seg={seg:.6f} pose={pose:.6f} size={archive_size/1024:.1f}KB "
          f"100s={100*seg:.4f} sqrtp={np.sqrt(10*pose):.4f} 25r={25*rate:.4f} "
          f"score={score:.4f}", flush=True)
    return score


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    print("Models loaded.", flush=True)
    
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)
    
    tmp_dir = ROOT / '_adv_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'
    
    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    
    def sky_blur(frames):
        result = []
        ksize = int(5.0 * 6) | 1
        for f in frames:
            blurred = cv2.GaussianBlur(f, (ksize, ksize), 5.0)
            result.append((f * sky_alpha + blurred * (1 - sky_alpha)).astype(np.uint8))
        return result
    
    # Test 1: Baseline (sky blur only, preset=0)
    print("\n=== TEST 1: Baseline (sky blur + preset 0) ===", flush=True)
    frames_sky = sky_blur([f.numpy() for f in gt_frames])
    t0 = time.time()
    mkv_sz = encode_piped(frames_sky, out_mkv, preset=0)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    inflated = decode_and_inflate(out_mkv)
    print(f"Encode+inflate: {time.time()-t0:.0f}s", flush=True)
    score_baseline = evaluate(inflated, gt_frames, segnet, posenet, archive_zip.stat().st_size)
    
    # Test 2: Adversarial preprocess + sky blur
    print("\n=== TEST 2: Adversarial preprocess (eps=4, 5 iters) + sky blur ===", flush=True)
    t0 = time.time()
    adv_frames = adversarial_preprocess(gt_frames, segnet, posenet,
                                         num_iters=5, lr=0.5, eps=4.0)
    frames_adv = sky_blur(adv_frames)
    mkv_sz = encode_piped(frames_adv, out_mkv, preset=0)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    inflated = decode_and_inflate(out_mkv)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)
    score_adv = evaluate(inflated, gt_frames, segnet, posenet, archive_zip.stat().st_size)
    
    # Test 3: Stronger adversarial
    print("\n=== TEST 3: Adversarial preprocess (eps=8, 10 iters) + sky blur ===", flush=True)
    t0 = time.time()
    adv_frames2 = adversarial_preprocess(gt_frames, segnet, posenet,
                                          num_iters=10, lr=0.3, eps=8.0)
    frames_adv2 = sky_blur(adv_frames2)
    mkv_sz = encode_piped(frames_adv2, out_mkv, preset=0)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    inflated = decode_and_inflate(out_mkv)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)
    score_adv2 = evaluate(inflated, gt_frames, segnet, posenet, archive_zip.stat().st_size)
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    print("\n" + "="*60)
    print(f"Baseline:     {score_baseline:.4f}")
    print(f"Adv eps=4:    {score_adv:.4f} (delta={score_adv - score_baseline:+.4f})")
    print(f"Adv eps=8:    {score_adv2:.4f} (delta={score_adv2 - score_baseline:+.4f})")
    print("="*60)
