#!/usr/bin/env python
"""
Temporal ROI preprocessing based on Spatial Complexity and Model Sensitivity.

Key insights:
1. The leader used 4 separate hand-drawn polygons mapped to different temporal frame ranges.
2. Automated per-frame ROI fails because it causes temporal flickering, destroying PoseNet.
3. Solution: Divide the video into K temporal segments. Compute a single, stable importance
   map for each segment using spatial edges and model sensitivity. Apply this stable mask
   to all frames in the segment.

This combines "static/simple regions" with "model sensitivity" to find safe areas to denoise,
while maintaining the temporal consistency PoseNet needs.
"""
import subprocess, sys, math, shutil
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

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)
TMP = ROOT / '_temporal_roi_tmp'


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


def load_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def compute_segment_importance(frames_np, start_idx, end_idx, segnet, posenet, sample_rate=10):
    """Compute a single stable importance map for a segment of frames."""
    model_w, model_h = segnet_model_input_size
    
    seg_sens_accum = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    pose_sens_accum = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    edge_accum = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    count = 0
    
    # We sample pairs to compute gradients
    for i in range(start_idx, end_idx - 1, sample_rate):
        if i % 2 != 0: i -= 1 # Ensure even start for pairs
        
        # 1. Spatial complexity (edges)
        gray = cv2.cvtColor(frames_np[i], cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
        edge_accum += cv2.GaussianBlur(edges, (15, 15), 5.0)
        
        # 2. Model sensitivity
        batch_data = torch.stack([
            torch.from_numpy(frames_np[i]).float(),
            torch.from_numpy(frames_np[i+1]).float()
        ]).unsqueeze(0).to(DEVICE) # 1, 2, H, W, C
        
        # SegNet sensitivity
        seg_input = batch_data.clone().requires_grad_(True)
        x_seg = einops.rearrange(seg_input, 'b t h w c -> b t c h w')
        seg_in = F.interpolate(x_seg[:, -1], size=(model_h, model_w),
                               mode='bilinear', align_corners=False)
        seg_out = segnet(seg_in)
        top2 = seg_out.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).mean()
        margin.backward()
        
        seg_grad = seg_input.grad.abs().mean(dim=(0, 1, 4)).detach().cpu().numpy()
        seg_sens_accum += seg_grad
        
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
        
        pose_grad = pose_input.grad.abs().mean(dim=(0, 1, 4)).detach().cpu().numpy()
        pose_sens_accum += pose_grad
        
        count += 1

    # Normalize components
    edge_map = edge_accum / count
    seg_map = seg_sens_accum / count
    pose_map = pose_sens_accum / count
    
    edge_map /= (edge_map.max() + 1e-8)
    seg_map /= (seg_map.max() + 1e-8)
    pose_map /= (pose_map.max() + 1e-8)
    
    # Combine and smooth heavily to create a "polygon-like" contiguous region
    importance = 0.3 * edge_map + 0.4 * seg_map + 0.3 * pose_map
    importance = cv2.GaussianBlur(importance, (91, 91), 30.0)
    
    return importance


def encode_piped(frames, out_mkv, crf='33', scale=0.45):
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', crf,
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def decode_and_upscale(mkv_path, unsharp_strength=0.44):
    import av
    from PIL import Image
    container = av.open(str(mkv_path))
    decoded = []
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        decoded.append(f_np)
    container.close()

    raw = bytearray()
    for f_np in decoded:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp_strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())
    return bytes(raw)


def eval_raw(raw_bytes, archive_size, segnet, posenet):
    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N*2, H_CAM, W_CAM, 3)
    seg_dists, pose_dists = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw_np[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw_np[2*i+1:2*end:2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, 'b t h w c -> b t c h w')
            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            gt_seg = gt['seg'][i:end].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())
            pn_out = posenet(posenet.preprocess_input(x))['pose'][:, :6]
            gt_pose = gt['pose'][i:end].to(DEVICE)
            pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())
    s, p = np.mean(seg_dists), np.mean(pose_dists)
    r = archive_size / 37_545_489
    score = 100*s + math.sqrt(10*p) + 25*r
    return score, s, p, r


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    TMP.mkdir(exist_ok=True)
    out_mkv = TMP / '0.mkv'

    orig_frames = load_frames()
    n_frames = len(orig_frames)
    print(f"Loaded {n_frames} original frames.", flush=True)

    # Leader used 4 polygons, let's use 4 segments
    K = 4
    seg_len = n_frames // K
    
    print("\nComputing stable importance maps per segment...", flush=True)
    segment_masks = []
    
    for k in range(K):
        start_idx = k * seg_len
        end_idx = n_frames if k == K - 1 else (k + 1) * seg_len
        
        print(f"  Segment {k+1}/{K} (frames {start_idx}-{end_idx})", flush=True)
        importance = compute_segment_importance(orig_frames, start_idx, end_idx, segnet, posenet)
        
        # Save visualization of importance map
        imp_vis = (importance / importance.max() * 255).astype(np.uint8)
        imp_color = cv2.applyColorMap(imp_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(TMP / f'importance_seg{k}.png'), imp_color)
        
        segment_masks.append((importance, start_idx, end_idx))

    def apply_masks(frames, threshold_pct=0.60, blur_sigma=5.0):
        processed = []
        ksize = int(blur_sigma * 6) | 1
        
        for k in range(K):
            importance, start_idx, end_idx = segment_masks[k]
            
            # Threshold to create a binary-like mask
            thresh_val = np.percentile(importance, (1.0 - threshold_pct) * 100)
            
            # Soft threshold for smooth transitions
            mask = np.clip((importance - thresh_val) / (importance.max() - thresh_val + 1e-8), 0, 1)
            # Ensure the center driving corridor is always kept (safety buffer)
            corridor = np.zeros_like(mask)
            cv2.fillPoly(corridor, [np.array([[200, 874], [500, 450], [664, 450], [964, 874]])], 1.0)
            corridor = cv2.GaussianBlur(corridor, (91, 91), 30.0)
            
            final_mask = np.clip(mask + corridor, 0, 1)
            alpha = final_mask[..., np.newaxis]
            
            # Save mask vis
            cv2.imwrite(str(TMP / f'mask_seg{k}_pct{threshold_pct}.png'), (final_mask * 255).astype(np.uint8))

            for i in range(start_idx, end_idx):
                f = frames[i]
                blurred = cv2.GaussianBlur(f, (ksize, ksize), blur_sigma)
                processed.append((f * alpha + blurred * (1 - alpha)).astype(np.uint8))
                
        return processed

    def test(name, threshold_pct, blur_sigma, crf='33'):
        processed_frames = apply_masks(orig_frames, threshold_pct, blur_sigma)
        sz = encode_piped(processed_frames, out_mkv, crf=crf)
        raw = decode_and_upscale(out_mkv)
        score, s, p, r = eval_raw(raw, sz, segnet, posenet)
        print(f"[{name:40s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={sz//1024}KB", flush=True)
        return score

    print("\n=== Testing Temporal ROI Configurations ===", flush=True)
    # Test different coverage percentages
    test("roi_temporal_keep70pct_s5.0", threshold_pct=0.70, blur_sigma=5.0)
    test("roi_temporal_keep60pct_s5.0", threshold_pct=0.60, blur_sigma=5.0)
    test("roi_temporal_keep50pct_s5.0", threshold_pct=0.50, blur_sigma=5.0)
    
    # Test different blur strengths
    test("roi_temporal_keep60pct_s3.0", threshold_pct=0.60, blur_sigma=3.0)
    test("roi_temporal_keep60pct_s8.0", threshold_pct=0.60, blur_sigma=8.0)
    
    # Test with lower CRF since we are blurring more of the frame than just sky
    test("roi_temporal_keep60pct_s5.0_crf32", threshold_pct=0.60, blur_sigma=5.0, crf='32')

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
