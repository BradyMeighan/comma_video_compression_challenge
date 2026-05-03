#!/usr/bin/env python
"""
Quick eval: measure distortion of pre-computed adversarial frames against teachers.
This is the "lossless baseline" — what we'd get with zero compression loss.

Usage: python -m submissions.phase3.eval_baseline
"""
import sys, math, time
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from submissions.phase2.inflate import posenet_preprocess_diff

DATA_DIR = Path('submissions/phase3/0')
MODEL_H, MODEL_W = segnet_model_input_size[1], segnet_model_input_size[0]
W_CAM, H_CAM = camera_size


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    frames_f0 = np.load(DATA_DIR / 'frames_f0.npy')        # (600, 3, 384, 512)
    frames_f1 = np.load(DATA_DIR / 'frames_f1.npy')        # (600, 3, 384, 512)
    seg_maps = np.load(DATA_DIR / 'seg_maps.npy')          # (600, 384, 512)
    pose_vectors = np.load(DATA_DIR / 'pose_vectors.npy')  # (600, 6)
    N = frames_f0.shape[0]
    print(f"Loaded {N} pairs: f0={frames_f0.shape}, f1={frames_f1.shape}")

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)

    seg_targets = torch.from_numpy(seg_maps).long().to(device)
    pose_targets = torch.from_numpy(pose_vectors.copy()).float().to(device)

    all_seg, all_pose = [], []
    batch_size = 32
    t0 = time.time()

    with torch.no_grad():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            f0 = torch.from_numpy(frames_f0[i:end]).float().to(device)
            f1 = torch.from_numpy(frames_f1[i:end]).float().to(device)

            # Round-trip: model_res -> camera_res -> uint8 -> model_res
            f0_up = F.interpolate(f0, (H_CAM, W_CAM), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round()
            f1_up = F.interpolate(f1, (H_CAM, W_CAM), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round()
            f0_down = F.interpolate(f0_up, (MODEL_H, MODEL_W),
                                    mode='bilinear', align_corners=False)
            f1_down = F.interpolate(f1_up, (MODEL_H, MODEL_W),
                                    mode='bilinear', align_corners=False)

            # Seg (uses frame_1 only)
            seg_pred = segnet(f1_down).argmax(1)
            seg_dist = (seg_pred != seg_targets[i:end]).float().mean(dim=(1, 2))
            all_seg.extend(seg_dist.cpu().tolist())

            # Pose (uses both frames)
            pair = torch.stack([f0_down, f1_down], dim=1)
            pn_in = posenet_preprocess_diff(pair)
            pose_out = posenet(pn_in)['pose'][:, :6]
            pose_dist = (pose_out - pose_targets[i:end]).pow(2).mean(dim=1)
            all_pose.extend(pose_dist.cpu().tolist())

            print(f"  Batch {i//batch_size+1}/{(N+batch_size-1)//batch_size} | "
                  f"seg={np.mean(all_seg[-len(seg_dist):]):.6f} "
                  f"pose={np.mean(all_pose[-len(pose_dist):]):.6f}",
                  flush=True)

    avg_seg = np.mean(all_seg)
    avg_pose = np.mean(all_pose)
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"LOSSLESS BASELINE (pre-computed frames, no compression loss)")
    print(f"{'='*60}")
    print(f"  seg_dist:  {avg_seg:.6f}  (100*seg = {100*avg_seg:.4f})")
    print(f"  pose_mse:  {avg_pose:.6f}  (sqrt(10*pose) = {math.sqrt(10*avg_pose):.4f})")
    print(f"  distortion total: {100*avg_seg + math.sqrt(10*avg_pose):.4f}")
    print(f"  eval time: {elapsed:.1f}s")

    # Score estimates at various archive sizes
    uncompressed = 37_545_489
    print(f"\n  Score estimates:")
    for size_kb in [200, 500, 1000, 2000, 3000, 5000]:
        rate = (size_kb * 1024) / uncompressed
        score = 100 * avg_seg + math.sqrt(10 * avg_pose) + 25 * rate
        print(f"    archive={size_kb:>5} KB -> 25*rate={25*rate:.3f} -> score={score:.4f}")


if __name__ == '__main__':
    main()
