#!/usr/bin/env python
import sys, math
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
TMP = ROOT / '_entropy_tmp'

def load_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames

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

def norm_map(m):
    return np.clip(m / (np.percentile(m, 99) + 1e-8), 0, 1)

if __name__ == '__main__':
    TMP.mkdir(exist_ok=True)
    frames = load_frames()
    segment = np.array(frames[0:300]) # 300, H, W, C
    
    gray_segment = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in segment]).astype(np.float32)
    temp_std = np.std(gray_segment, axis=0) # H, W
    
    edges_accum = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    for g in gray_segment:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        edges_accum += np.sqrt(gx**2 + gy**2)
    spatial_complexity = edges_accum / len(gray_segment)
    
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters(): p.requires_grad_(False)
    for p in posenet.parameters(): p.requires_grad_(False)
        
    model_w, model_h = segnet_model_input_size
    seg_sens = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    pose_sens = np.zeros((H_CAM, W_CAM), dtype=np.float32)
    count = 0
    
    with torch.enable_grad():
        for i in range(0, 300, 10):
            if i % 2 != 0: i -= 1
            batch_data = torch.stack([
                torch.from_numpy(segment[i]).float(),
                torch.from_numpy(segment[i+1]).float()
            ]).unsqueeze(0).to(DEVICE)
            
            si = batch_data.clone().requires_grad_(True)
            x_seg = einops.rearrange(si, 'b t h w c -> b t c h w')
            seg_in = F.interpolate(x_seg[:, -1], size=(model_h, model_w), mode='bilinear', align_corners=False)
            seg_out = segnet(seg_in)
            top2 = seg_out.topk(2, dim=1).values
            margin = (top2[:, 0] - top2[:, 1]).mean()
            margin.backward()
            seg_sens += si.grad.abs().mean(dim=(0, 1, 4)).detach().cpu().numpy()
            
            pi = batch_data.clone().requires_grad_(True)
            x_pose = einops.rearrange(pi, 'b t h w c -> b t c h w')
            b_sz, s_len = x_pose.shape[:2]
            flat = einops.rearrange(x_pose, 'b t c h w -> (b t) c h w')
            flat = F.interpolate(flat, size=(model_h, model_w), mode='bilinear', align_corners=False)
            yuv = rgb_to_yuv6_diff(flat)
            pn_in = einops.rearrange(yuv, '(b t) c h w -> b (t c) h w', b=b_sz, t=s_len)
            pn_out = posenet(pn_in)['pose'][:, :6]
            pn_out.pow(2).sum().backward()
            pose_sens += pi.grad.abs().mean(dim=(0, 1, 4)).detach().cpu().numpy()
            
            count += 1

    seg_sens /= count
    pose_sens /= count
    
    n_temp = norm_map(temp_std)
    n_spat = norm_map(spatial_complexity)
    n_seg = norm_map(seg_sens)
    n_pose = norm_map(pose_sens)

    # 1. Base importance (max of all factors)
    importance = np.maximum.reduce([n_temp, n_spat, n_seg, n_pose])
    
    # 2. Force car hood to be important
    hood_start = int(H_CAM * 0.85)
    importance[hood_start:, :] = 1.0
    
    # 3. Smooth heavily to simulate a hand-drawn polygon
    smooth_imp = cv2.GaussianBlur(importance, (91, 91), 30.0)
    
    # 4. Threshold
    threshold_vals = [0.10, 0.15, 0.20, 0.25, 0.30]
    for th in threshold_vals:
        mask = (smooth_imp > th).astype(np.float32)
        # Soften edges slightly
        mask = cv2.GaussianBlur(mask, (31, 31), 10.0)
        cv2.imwrite(str(TMP / f'combined_mask_th{th}.png'), (mask * 255).astype(np.uint8))
        
    cv2.imwrite(str(TMP / 'smooth_importance.png'), cv2.applyColorMap((smooth_imp * 255).astype(np.uint8), cv2.COLORMAP_JET))
    
    print("Done generating combined masks.", flush=True)
