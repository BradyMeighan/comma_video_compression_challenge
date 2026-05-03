#!/usr/bin/env python
"""
Gradient correction at decode time: after Lanczos upscale + unsharp,
run SegNet/PoseNet on the decoded frames to compute per-pixel gradients,
then apply small corrections in the gradient direction.

This is a FREE optimization — no extra archive data needed.
The correction is computed at decode time from the models themselves.

Key insight: The models are available at decode time (they're part of the
evaluation environment, not the archive). We can use them to compute
optimal pixel corrections.

IMPORTANT: This requires models at decode time. The constraint says
"Any neural network or large artifact used during inflation (decode) 
MUST be included in archive.zip". So this approach violates the rules
unless we include the models in the archive, which would make the archive
too large.

HOWEVER — the decode environment already has the models available (they're
used for evaluation). If we can use them WITHOUT including them in the
archive, this would work. Let me check the rules more carefully.

Actually, the rules say the models must be in the archive. So this
approach is not viable unless the correction is very small and can be
precomputed at encode time and stored compactly.

REVISED APPROACH: Precompute gradient-based corrections at encode time,
store as a compact quantized delta map in the archive. At decode time,
just apply the stored corrections.
"""
import subprocess, sys, os, time, zipfile, shutil, struct
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


def decode_and_inflate_tensors(mkv_path, strength=0.45):
    """Decode and inflate, returning tensors with gradients enabled."""
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
        frames.append(x.clamp(0, 255).squeeze(0))  # 3, H, W float
    container.close()
    return frames


def compute_per_frame_gradients(inflated_tensors, gt_frames, segnet, posenet):
    """Compute gradient of model loss w.r.t. each inflated frame.
    
    Returns per-frame gradient tensors and per-frame loss values.
    """
    n_pairs = len(gt_frames) // 2
    gradients = [None] * len(inflated_tensors)
    seg_losses = []
    pose_losses = []
    
    batch_size = 8
    for bs in range(0, n_pairs, batch_size):
        be = min(bs + batch_size, n_pairs)
        B = be - bs
        
        # Build pairs with gradients
        gt_batch = []
        for i in range(bs, be):
            gt_batch.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
        gt_batch = torch.stack(gt_batch).to(DEVICE)
        gt_x = einops.rearrange(gt_batch, 'b t h w c -> b t c h w')
        
        comp_f0 = torch.stack([inflated_tensors[i*2].clone() for i in range(bs, be)])
        comp_f1 = torch.stack([inflated_tensors[i*2+1].clone() for i in range(bs, be)])
        comp_f0.requires_grad_(True)
        comp_f1.requires_grad_(True)
        
        # SegNet: uses last frame only
        model_w, model_h = segnet_model_input_size
        seg_in_gt = F.interpolate(gt_x[:, 1], size=(model_h, model_w),
                                   mode='bilinear', align_corners=False) / 255.0
        seg_in_comp = F.interpolate(comp_f1, size=(model_h, model_w),
                                     mode='bilinear', align_corners=False) / 255.0
        
        with torch.inference_mode():
            gt_seg_logits = segnet(seg_in_gt)
            gt_seg_map = gt_seg_logits.argmax(dim=1)
        
        comp_seg_logits = segnet(seg_in_comp)
        seg_loss = F.cross_entropy(comp_seg_logits, gt_seg_map)
        
        # PoseNet: uses pair
        comp_both = torch.stack([comp_f0, comp_f1], dim=1)  # B, 2, 3, H, W
        comp_both_hwc = comp_both.permute(0, 1, 3, 4, 2)
        comp_pn_in = posenet.preprocess_input(comp_both_hwc)
        comp_pose = posenet(comp_pn_in)['pose'][:, :6]
        
        with torch.inference_mode():
            gt_pn_in = posenet.preprocess_input(gt_x.permute(0, 1, 3, 4, 2))
            gt_pose = posenet(gt_pn_in)['pose'][:, :6]
        
        pose_loss = F.mse_loss(comp_pose, gt_pose.detach())
        
        total_loss = 100 * seg_loss + 10 * pose_loss
        total_loss.backward()
        
        seg_losses.append(seg_loss.item())
        pose_losses.append(pose_loss.item())
        
        for i in range(B):
            pair_idx = bs + i
            gradients[pair_idx*2] = comp_f0.grad[i].detach().cpu()
            gradients[pair_idx*2+1] = comp_f1.grad[i].detach().cpu()
    
    return gradients, seg_losses, pose_losses


def apply_gradient_correction(inflated_tensors, gradients, step_size=1.0):
    """Apply gradient-based correction to inflated frames."""
    corrected = []
    for i in range(len(inflated_tensors)):
        frame = inflated_tensors[i].clone()
        if gradients[i] is not None:
            grad = gradients[i].to(DEVICE)
            grad_norm = grad.abs().max()
            if grad_norm > 0:
                frame = frame - step_size * (grad / grad_norm)
        corrected.append(frame.clamp(0, 255).round().to(torch.uint8)
                        .permute(1, 2, 0).cpu().numpy())
    return corrected


def evaluate(inflated_np, gt_frames, segnet, posenet, archive_size):
    n_pairs = len(gt_frames) // 2
    seg_losses = []
    pose_losses = []
    
    for bs in range(0, n_pairs, 16):
        be = min(bs + 16, n_pairs)
        gt_b = []
        comp_b = []
        for i in range(bs, be):
            gt_b.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
            comp_b.append(torch.stack([
                torch.from_numpy(inflated_np[i*2]).float(),
                torch.from_numpy(inflated_np[i*2+1]).float()
            ]))
        
        gt_b = torch.stack(gt_b).to(DEVICE)
        comp_b = torch.stack(comp_b).to(DEVICE)
        gt_x = einops.rearrange(gt_b, 'b t h w c -> b t c h w')
        comp_x = einops.rearrange(comp_b, 'b t h w c -> b t c h w')
        
        with torch.inference_mode():
            gs = segnet(segnet.preprocess_input(gt_x)).argmax(dim=1)
            cs = segnet(segnet.preprocess_input(comp_x)).argmax(dim=1)
            seg_losses.extend((gs != cs).float().mean(dim=(1,2)).cpu().tolist())
            
            gp = posenet(posenet.preprocess_input(gt_x))['pose'][:, :6]
            cp = posenet(posenet.preprocess_input(comp_x))['pose'][:, :6]
            pose_losses.extend(((gp - cp)**2).mean(dim=1).cpu().tolist())
    
    s = np.mean(seg_losses)
    p = np.mean(pose_losses)
    r = archive_size / 37_545_489
    score = 100*s + np.sqrt(10*p) + 25*r
    print(f"  seg={s:.6f} pose={p:.6f} 100s={100*s:.4f} sqrtp={np.sqrt(10*p):.4f} "
          f"25r={25*r:.4f} score={score:.4f}", flush=True)
    return score


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    print("Models loaded.", flush=True)
    
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)
    
    tmp_dir = ROOT / '_grad_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'
    
    # Encode with sky blur + best params
    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    frames_np = []
    for f in gt_frames:
        fn = f.numpy()
        blurred = cv2.GaussianBlur(fn, (ksize, ksize), 5.0)
        frames_np.append((fn * sky_alpha + blurred * (1 - sky_alpha)).astype(np.uint8))
    
    print("Encoding with preset=0...", flush=True)
    t0 = time.time()
    mkv_sz = encode_piped(frames_np, out_mkv, preset=0)
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size
    print(f"  Encoded in {time.time()-t0:.0f}s, archive={archive_size/1024:.1f}KB", flush=True)
    
    # Decode and inflate
    print("Decoding and inflating...", flush=True)
    inflated = decode_and_inflate_tensors(out_mkv)
    print(f"  {len(inflated)} frames decoded.", flush=True)
    
    # Baseline eval (no correction)
    print("\n=== Baseline (no correction) ===", flush=True)
    baseline_np = [f.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                   for f in inflated]
    score_baseline = evaluate(baseline_np, gt_frames, segnet, posenet, archive_size)
    
    # Compute gradients
    print("\nComputing per-frame gradients...", flush=True)
    t0 = time.time()
    gradients, seg_l, pose_l = compute_per_frame_gradients(
        inflated, gt_frames, segnet, posenet)
    print(f"  Gradients computed in {time.time()-t0:.0f}s", flush=True)
    print(f"  Avg seg loss: {np.mean(seg_l):.6f}, avg pose loss: {np.mean(pose_l):.6f}")
    
    # Test different correction step sizes
    for step in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        print(f"\n=== Gradient correction step={step} ===", flush=True)
        corrected = apply_gradient_correction(inflated, gradients, step_size=step)
        evaluate(corrected, gt_frames, segnet, posenet, archive_size)
    
    # Multi-step gradient descent on inflated frames
    print("\n=== Multi-step gradient descent (5 steps, lr=1.0) ===", flush=True)
    current = [f.clone() for f in inflated]
    for step in range(5):
        grads, sl, pl = compute_per_frame_gradients(current, gt_frames, segnet, posenet)
        for i in range(len(current)):
            if grads[i] is not None:
                g = grads[i].to(DEVICE)
                gn = g.abs().max()
                if gn > 0:
                    current[i] = (current[i] - 1.0 * g / gn).clamp(0, 255)
        print(f"  Step {step+1}: seg={np.mean(sl):.6f} pose={np.mean(pl):.6f}", flush=True)
    
    corrected_ms = [f.round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    for f in current]
    evaluate(corrected_ms, gt_frames, segnet, posenet, archive_size)
    
    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
