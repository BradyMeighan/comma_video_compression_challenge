#!/usr/bin/env python
"""
Analyze WHERE segmentation and pose distortion occurs in codec-decoded frames.

Generates diagnostic images:
1. Per-pixel seg mismatch maps (where argmax flips)
2. Side-by-side original vs decoded seg maps
3. Per-pair distortion histogram with worst-pair analysis
4. Spatial heatmap of cumulative seg errors

This should reveal whether errors are at class boundaries, in specific regions,
or at specific temporal points — guiding targeted fixes.
"""
import sys, time, math, os
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
OUT_DIR = ROOT / 'viz_analysis'

# Class colors for visualization
CLASS_COLORS = np.array([
    [0, 0, 200],     # class 0: blue (sky)
    [128, 128, 128],  # class 1: grey (vehicle)
    [0, 128, 0],      # class 2: green (road/vegetation)
    [128, 64, 0],     # class 3: brown
    [64, 0, 64],      # class 4: purple
], dtype=np.uint8)


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_frames(path):
    import av
    container = av.open(str(path))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def seg_map_to_color(seg_map):
    """Convert (H, W) class labels to (H, W, 3) color image."""
    h, w = seg_map.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(5):
        color[seg_map == c] = CLASS_COLORS[c]
    return color


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    OUT_DIR.mkdir(exist_ok=True)

    segnet, posenet = load_models()

    # Load original frames
    print("Loading original frames...", flush=True)
    orig_frames = load_frames(VIDEO)
    n_frames = len(orig_frames)
    n_pairs = n_frames // 2
    print(f"  {n_frames} frames, {n_pairs} pairs", flush=True)

    # Load codec-decoded frames
    codec_raw_path = ROOT / 'submissions' / 'av1_repro' / 'inflated' / '0.raw'
    if not codec_raw_path.exists():
        print("No inflated raw file found. Run inflate first.", flush=True)
        sys.exit(1)

    raw_data = np.fromfile(codec_raw_path, dtype=np.uint8).reshape(n_frames, H_CAM, W_CAM, 3)
    print(f"  Loaded codec output: {raw_data.shape}", flush=True)

    model_h, model_w = segnet_model_input_size[1], segnet_model_input_size[0]

    # Compute per-pair seg and pose distortion
    print("Computing per-pair distortion...", flush=True)
    seg_dists = []
    pose_dists = []
    orig_seg_maps = []
    codec_seg_maps = []
    seg_mismatch_maps = []

    with torch.inference_mode():
        for i in range(n_pairs):
            # Original pair
            of0 = orig_frames[i*2].float().unsqueeze(0).to(DEVICE)
            of1 = orig_frames[i*2+1].float().unsqueeze(0).to(DEVICE)
            ox = torch.stack([of0, of1], dim=1)
            ox = einops.rearrange(ox, 'b t h w c -> b t c h w')

            o_seg_in = segnet.preprocess_input(ox)
            o_seg_out = segnet(o_seg_in)
            o_seg_labels = o_seg_out.argmax(1).cpu()  # 1, H, W

            o_pn_in = posenet.preprocess_input(ox)
            o_pose = posenet(o_pn_in)['pose'][:, :6].cpu()

            # Codec pair
            cf0 = torch.from_numpy(raw_data[i*2].copy()).float().unsqueeze(0).to(DEVICE)
            cf1 = torch.from_numpy(raw_data[i*2+1].copy()).float().unsqueeze(0).to(DEVICE)
            cx = torch.stack([cf0, cf1], dim=1)
            cx = einops.rearrange(cx, 'b t h w c -> b t c h w')

            c_seg_in = segnet.preprocess_input(cx)
            c_seg_out = segnet(c_seg_in)
            c_seg_labels = c_seg_out.argmax(1).cpu()

            c_pn_in = posenet.preprocess_input(cx)
            c_pose = posenet(c_pn_in)['pose'][:, :6].cpu()

            # Distortion
            mismatch = (o_seg_labels != c_seg_labels).float()
            seg_d = mismatch.mean().item()
            pose_d = (o_pose - c_pose).pow(2).mean().item()

            seg_dists.append(seg_d)
            pose_dists.append(pose_d)

            if i < 20 or seg_d > 0.01:
                orig_seg_maps.append(o_seg_labels[0].numpy())
                codec_seg_maps.append(c_seg_labels[0].numpy())
                seg_mismatch_maps.append(mismatch[0].numpy())

            if i % 50 == 0:
                print(f"  Pair {i}/{n_pairs}: seg={seg_d:.6f} pose={pose_d:.6f}", flush=True)

    seg_dists = np.array(seg_dists)
    pose_dists = np.array(pose_dists)

    print(f"\nOverall: seg_mean={seg_dists.mean():.6f} pose_mean={pose_dists.mean():.6f}")
    print(f"Score components: 100*seg={100*seg_dists.mean():.4f} "
          f"sqrt(10*pose)={math.sqrt(10*pose_dists.mean()):.4f}")

    # Find worst pairs
    worst_seg_idx = np.argsort(seg_dists)[-20:][::-1]
    worst_pose_idx = np.argsort(pose_dists)[-20:][::-1]

    print(f"\nWorst 20 seg pairs:")
    for idx in worst_seg_idx:
        print(f"  Pair {idx}: seg={seg_dists[idx]:.6f} pose={pose_dists[idx]:.6f}")

    print(f"\nWorst 20 pose pairs:")
    for idx in worst_pose_idx:
        print(f"  Pair {idx}: seg={seg_dists[idx]:.6f} pose={pose_dists[idx]:.6f}")

    # Generate visualization for worst pairs
    print("\nGenerating visualizations...", flush=True)

    # Cumulative spatial error heatmap
    all_mismatch = np.zeros((model_h, model_w), dtype=np.float64)
    with torch.inference_mode():
        for i in range(n_pairs):
            of1 = orig_frames[i*2+1].float().unsqueeze(0).to(DEVICE)
            cf1 = torch.from_numpy(raw_data[i*2+1].copy()).float().unsqueeze(0).to(DEVICE)

            # Only need last frame for seg
            o_in = F.interpolate(of1.permute(0,3,1,2), size=(model_h, model_w),
                                 mode='bilinear', align_corners=False)
            c_in = F.interpolate(cf1.permute(0,3,1,2), size=(model_h, model_w),
                                 mode='bilinear', align_corners=False)

            o_seg = segnet(o_in).argmax(1).cpu().numpy()[0]
            c_seg = segnet(c_in).argmax(1).cpu().numpy()[0]

            all_mismatch += (o_seg != c_seg).astype(np.float64)

    heatmap = (all_mismatch / n_pairs * 255).clip(0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUT_DIR / 'seg_error_heatmap.png'), heatmap_color)
    print(f"  Saved seg_error_heatmap.png", flush=True)

    # Save statistics
    # Where do errors concentrate? Top/bottom/left/right/center?
    top_half = all_mismatch[:model_h//2, :].mean()
    bot_half = all_mismatch[model_h//2:, :].mean()
    left_half = all_mismatch[:, :model_w//2].mean()
    right_half = all_mismatch[:, model_w//2:].mean()
    center = all_mismatch[model_h//4:3*model_h//4, model_w//4:3*model_w//4].mean()
    edge = (all_mismatch.sum() - all_mismatch[model_h//4:3*model_h//4,
            model_w//4:3*model_w//4].sum()) / (model_h*model_w - (model_h//2)*(model_w//2))

    print(f"\nSpatial error distribution:")
    print(f"  Top half:    {top_half:.6f}")
    print(f"  Bottom half: {bot_half:.6f}")
    print(f"  Left half:   {left_half:.6f}")
    print(f"  Right half:  {right_half:.6f}")
    print(f"  Center:      {center:.6f}")
    print(f"  Edges:       {edge:.6f}")

    # Visualize worst 5 seg pairs: original seg vs codec seg vs mismatch
    for rank, idx in enumerate(worst_seg_idx[:5]):
        with torch.inference_mode():
            of1 = orig_frames[idx*2+1].float().unsqueeze(0).to(DEVICE)
            cf1 = torch.from_numpy(raw_data[idx*2+1].copy()).float().unsqueeze(0).to(DEVICE)

            o_in = F.interpolate(of1.permute(0,3,1,2), size=(model_h, model_w),
                                 mode='bilinear', align_corners=False)
            c_in = F.interpolate(cf1.permute(0,3,1,2), size=(model_h, model_w),
                                 mode='bilinear', align_corners=False)

            o_seg = segnet(o_in).argmax(1).cpu().numpy()[0]
            c_seg = segnet(c_in).argmax(1).cpu().numpy()[0]

        mismatch = (o_seg != c_seg)

        o_color = seg_map_to_color(o_seg)
        c_color = seg_map_to_color(c_seg)
        diff_color = np.zeros_like(o_color)
        diff_color[mismatch] = [255, 0, 0]  # red for errors

        # Side-by-side: original_seg | codec_seg | mismatch
        vis = np.concatenate([o_color, c_color, diff_color], axis=1)
        cv2.imwrite(str(OUT_DIR / f'worst_seg_{rank}_{idx}.png'),
                    cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # Also save the actual frames
        orig_frame = cv2.resize(orig_frames[idx*2+1].numpy(), (model_w, model_h))
        codec_frame = cv2.resize(raw_data[idx*2+1], (model_w, model_h))
        frame_vis = np.concatenate([orig_frame, codec_frame], axis=1)
        cv2.imwrite(str(OUT_DIR / f'worst_frame_{rank}_{idx}.png'),
                    cv2.cvtColor(frame_vis, cv2.COLOR_RGB2BGR))

    print(f"  Saved worst pair visualizations", flush=True)

    # Per-class error analysis
    print(f"\nPer-class error analysis:")
    class_errors = np.zeros(5, dtype=np.float64)
    class_counts = np.zeros(5, dtype=np.float64)
    class_flip_to = np.zeros((5, 5), dtype=np.float64)

    with torch.inference_mode():
        for i in range(n_pairs):
            of1 = orig_frames[i*2+1].float().unsqueeze(0).to(DEVICE)
            cf1 = torch.from_numpy(raw_data[i*2+1].copy()).float().unsqueeze(0).to(DEVICE)

            o_in = F.interpolate(of1.permute(0,3,1,2), size=(model_h, model_w),
                                 mode='bilinear', align_corners=False)
            c_in = F.interpolate(cf1.permute(0,3,1,2), size=(model_h, model_w),
                                 mode='bilinear', align_corners=False)

            o_seg = segnet(o_in).argmax(1).cpu().numpy()[0]
            c_seg = segnet(c_in).argmax(1).cpu().numpy()[0]

            for c in range(5):
                mask = o_seg == c
                class_counts[c] += mask.sum()
                mismatched = mask & (o_seg != c_seg)
                class_errors[c] += mismatched.sum()

                for c2 in range(5):
                    class_flip_to[c, c2] += (mask & (c_seg == c2)).sum()

    for c in range(5):
        if class_counts[c] > 0:
            err_rate = class_errors[c] / class_counts[c]
            print(f"  Class {c}: count={class_counts[c]:.0f} "
                  f"({class_counts[c]/(model_h*model_w*n_pairs)*100:.1f}%) "
                  f"error_rate={err_rate*100:.2f}%")

    print(f"\nClass confusion matrix (GT row → Pred col, errors only):")
    print(f"     {''.join(f'  C{c}   ' for c in range(5))}")
    for c in range(5):
        row = []
        for c2 in range(5):
            if c == c2:
                row.append("  ---  ")
            elif class_flip_to[c, c2] > 0:
                pct = class_flip_to[c, c2] / max(class_counts[c], 1) * 100
                row.append(f"{pct:5.2f}%")
            else:
                row.append("  0   ")
        print(f"  C{c}: {'  '.join(row)}")

    print(f"\nDone. Visualizations saved to {OUT_DIR}/", flush=True)
