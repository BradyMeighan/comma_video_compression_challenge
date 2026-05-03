#!/usr/bin/env python
"""
Deep analysis of worst-performing frame pairs.

Goal: understand WHY certain pairs have 10x higher pose distortion.
If we can identify a pattern (scene change, motion, specific content),
we might be able to preprocess those frames differently.

Key questions:
1. Are worst pairs clustered temporally? (scene changes)
2. Do worst pairs have more motion between frames?
3. Are worst pairs in specific visual conditions (night, rain, curves)?
4. How much would fixing JUST the worst 20 pairs improve the total score?
"""
import sys, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file
import einops

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()

    # Load original and decoded frames
    import av
    container = av.open(str(VIDEO))
    orig_frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    n_frames = len(orig_frames)
    n_pairs = n_frames // 2
    print(f"Loaded {n_frames} original frames, {n_pairs} pairs.", flush=True)

    codec_path = ROOT / 'submissions' / 'av1_repro' / 'inflated' / '0.raw'
    codec_raw = np.fromfile(codec_path, dtype=np.uint8).reshape(n_frames, H_CAM, W_CAM, 3)
    print(f"Loaded codec decoded frames.", flush=True)

    # Compute per-pair distortion for both original and codec
    print("\nComputing per-pair distortion...", flush=True)
    seg_dists = []
    pose_dists = []
    orig_pose_vals = []
    codec_pose_vals = []
    inter_frame_motion = []

    with torch.inference_mode():
        for i in range(n_pairs):
            # Original pair
            of0 = torch.from_numpy(orig_frames[i*2].copy()).float().unsqueeze(0).to(DEVICE)
            of1 = torch.from_numpy(orig_frames[i*2+1].copy()).float().unsqueeze(0).to(DEVICE)
            ox = torch.stack([of0, of1], dim=1)
            ox = einops.rearrange(ox, 'b t h w c -> b t c h w')
            o_seg = segnet(segnet.preprocess_input(ox)).argmax(1).cpu()
            o_pose = posenet(posenet.preprocess_input(ox))['pose'][:, :6].cpu()
            orig_pose_vals.append(o_pose[0].numpy())

            # Codec pair
            cf0 = torch.from_numpy(codec_raw[i*2].copy()).float().unsqueeze(0).to(DEVICE)
            cf1 = torch.from_numpy(codec_raw[i*2+1].copy()).float().unsqueeze(0).to(DEVICE)
            cx = torch.stack([cf0, cf1], dim=1)
            cx = einops.rearrange(cx, 'b t h w c -> b t c h w')
            c_seg = segnet(segnet.preprocess_input(cx)).argmax(1).cpu()
            c_pose = posenet(posenet.preprocess_input(cx))['pose'][:, :6].cpu()
            codec_pose_vals.append(c_pose[0].numpy())

            # Distortion
            seg_d = (o_seg != c_seg).float().mean().item()
            pose_d = (o_pose - c_pose).pow(2).mean().item()
            seg_dists.append(seg_d)
            pose_dists.append(pose_d)

            # Inter-frame motion (mean pixel difference between frame 0 and frame 1)
            motion = np.abs(orig_frames[i*2].astype(np.float32) -
                            orig_frames[i*2+1].astype(np.float32)).mean()
            inter_frame_motion.append(motion)

            if i % 100 == 0:
                print(f"  Pair {i}/{n_pairs}", flush=True)

    seg_dists = np.array(seg_dists)
    pose_dists = np.array(pose_dists)
    inter_frame_motion = np.array(inter_frame_motion)
    orig_pose_vals = np.array(orig_pose_vals)
    codec_pose_vals = np.array(codec_pose_vals)

    # 1. How much would fixing worst pairs help?
    print("\n=== Impact analysis ===")
    pose_sorted_idx = np.argsort(pose_dists)[::-1]
    total_pose = pose_dists.mean()
    for top_k in [5, 10, 20, 50, 100]:
        worst_k = pose_sorted_idx[:top_k]
        # If we could reduce these to the median
        median_pose = np.median(pose_dists)
        fixed = pose_dists.copy()
        fixed[worst_k] = median_pose
        new_total = fixed.mean()
        improvement = math.sqrt(10 * total_pose) - math.sqrt(10 * new_total)
        print(f"  Fix worst {top_k} pose pairs to median: "
              f"pose {math.sqrt(10*total_pose):.4f} → {math.sqrt(10*new_total):.4f} "
              f"(-{improvement:.4f} score)")

    # 2. Temporal clustering
    print("\n=== Temporal clustering of worst pairs ===")
    worst_20 = pose_sorted_idx[:20]
    worst_20_sorted = np.sort(worst_20)
    print(f"  Worst 20 pose pairs (temporal order): {worst_20_sorted}")
    
    # Check for clusters (pairs within 5 frames of each other)
    clusters = []
    current = [worst_20_sorted[0]]
    for i in range(1, len(worst_20_sorted)):
        if worst_20_sorted[i] - worst_20_sorted[i-1] <= 5:
            current.append(worst_20_sorted[i])
        else:
            clusters.append(current)
            current = [worst_20_sorted[i]]
    clusters.append(current)
    
    print(f"  Clusters (within 5 frames): {len(clusters)}")
    for c in clusters:
        print(f"    Pairs {c[0]}-{c[-1]} ({len(c)} pairs)")

    # 3. Correlation with motion
    print("\n=== Motion correlation ===")
    motion_corr = np.corrcoef(pose_dists, inter_frame_motion)[0, 1]
    print(f"  Pose distortion vs inter-frame motion: r={motion_corr:.4f}")
    
    worst_motion = inter_frame_motion[worst_20]
    all_motion = inter_frame_motion
    print(f"  Motion: worst 20 mean={worst_motion.mean():.2f} "
          f"vs all mean={all_motion.mean():.2f}")

    # 4. Pose vector analysis: what component contributes most?
    print("\n=== Pose vector component analysis ===")
    pose_diff = orig_pose_vals - codec_pose_vals
    component_mse = (pose_diff**2).mean(axis=0)
    labels = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    for j in range(6):
        print(f"  {labels[j]}: MSE={component_mse[j]:.6f}")
    
    print(f"\n  Worst pairs pose vector diff:")
    for rank, idx in enumerate(worst_20[:5]):
        diff = pose_diff[idx]
        print(f"    Pair {idx}: " + 
              " ".join(f"{labels[j]}={diff[j]:+.4f}" for j in range(6)))

    # 5. Brightness analysis of worst pairs
    print("\n=== Brightness of worst pairs ===")
    for idx in worst_20[:10]:
        f0_bright = orig_frames[idx*2].mean()
        f1_bright = orig_frames[idx*2+1].mean()
        print(f"  Pair {idx}: brightness f0={f0_bright:.1f} f1={f1_bright:.1f} "
              f"pose_d={pose_dists[idx]:.4f} motion={inter_frame_motion[idx]:.1f}")

    # 6. Save worst pair frames for visual inspection
    out_dir = ROOT / 'viz_analysis'
    out_dir.mkdir(exist_ok=True)
    for rank, idx in enumerate(worst_20[:3]):
        pair_vis = np.concatenate([
            np.concatenate([orig_frames[idx*2], orig_frames[idx*2+1]], axis=1),
            np.concatenate([codec_raw[idx*2], codec_raw[idx*2+1]], axis=1),
        ], axis=0)
        pair_vis_small = cv2.resize(pair_vis, (pair_vis.shape[1]//2, pair_vis.shape[0]//2))
        cv2.imwrite(str(out_dir / f'worst_pose_{rank}_{idx}.png'),
                    cv2.cvtColor(pair_vis_small, cv2.COLOR_RGB2BGR))

    print(f"\nDone.", flush=True)
