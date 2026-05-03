#!/usr/bin/env python
"""
Test lossy video compression of pre-computed adversarial frames.
Encodes with ffmpeg at various quality levels, then evaluates distortion.

Usage: python -m submissions.phase3.test_compression
"""
import subprocess, struct, time, math, os, sys
import numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path

DATA_DIR = Path('submissions/phase3/0')
MODEL_H, MODEL_W = segnet_model_input_size[1], segnet_model_input_size[0]
W_CAM, H_CAM = camera_size


def frames_to_raw_video(frames_np, path):
    """Write (N, 3, H, W) uint8 numpy array as raw RGB planar → packed RGB for ffmpeg."""
    # ffmpeg wants rawvideo in HWC format
    hwc = np.ascontiguousarray(frames_np.transpose(0, 2, 3, 1))  # (N, H, W, 3)
    hwc.tofile(str(path))
    return hwc.shape


def encode_h264(raw_path, out_path, N, H, W, crf=18, preset='veryslow'):
    """Encode raw RGB frames to H.264."""
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', '20', '-i', str(raw_path),
        '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
        '-pix_fmt', 'yuv444p',  # preserve colors better than yuv420p
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return os.path.getsize(out_path)


def encode_h265(raw_path, out_path, N, H, W, crf=18, preset='veryslow'):
    """Encode raw RGB frames to H.265."""
    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}', '-r', '20', '-i', str(raw_path),
        '-c:v', 'libx265', '-preset', preset, '-crf', str(crf),
        '-pix_fmt', 'yuv444p',
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return os.path.getsize(out_path)


def decode_video(video_path, N, H, W):
    """Decode video back to raw RGB frames."""
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        'pipe:1',
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(N, H, W, 3)
    return np.ascontiguousarray(frames.transpose(0, 3, 1, 2))  # (N, 3, H, W)


def evaluate_frames(frames_uint8, seg_maps, pose_vectors, device):
    """Evaluate adversarial frames against teacher models with round-trip."""
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)

    from submissions.phase2.inflate import posenet_preprocess_diff

    N = frames_uint8.shape[0]
    seg_targets = torch.from_numpy(seg_maps).long().to(device)
    pose_targets = torch.from_numpy(pose_vectors.copy()).float().to(device)

    all_seg_dist = []
    all_pose_dist = []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            f1 = torch.from_numpy(frames_uint8[i:end]).float().to(device)

            # Simulate round-trip: model_res → camera_res → uint8 → model_res
            f1_up = F.interpolate(f1, (H_CAM, W_CAM), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round()
            f1_down = F.interpolate(f1_up, (MODEL_H, MODEL_W), mode='bilinear',
                                    align_corners=False)

            # Seg eval
            seg_pred = segnet(f1_down).argmax(1)
            seg_dist = (seg_pred != seg_targets[i:end]).float().mean(dim=(1, 2))
            all_seg_dist.extend(seg_dist.cpu().tolist())

            # Pose eval — use mean-color frame_0 (same as adversarial decode init)
            f0 = f1.mean(dim=(-2, -1), keepdim=True).expand_as(f1)
            f0_up = F.interpolate(f0, (H_CAM, W_CAM), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round()
            f0_down = F.interpolate(f0_up, (MODEL_H, MODEL_W), mode='bilinear',
                                    align_corners=False)
            pair = torch.stack([f0_down, f1_down], dim=1)
            pn_in = posenet_preprocess_diff(pair)
            pose_out = posenet(pn_in)['pose'][:, :6]
            pose_dist = (pose_out - pose_targets[i:end]).pow(2).mean(dim=1)
            all_pose_dist.extend(pose_dist.cpu().tolist())

            if (i // batch_size) % 5 == 0:
                print(f"    Eval batch {i//batch_size + 1}: "
                      f"seg={np.mean(all_seg_dist[-len(seg_dist):]):.6f} "
                      f"pose={np.mean(all_pose_dist[-len(pose_dist):]):.6f}",
                      flush=True)

    del segnet, posenet
    torch.cuda.empty_cache()

    avg_seg = np.mean(all_seg_dist)
    avg_pose = np.mean(all_pose_dist)
    return avg_seg, avg_pose


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load pre-computed frames
    print("Loading pre-computed adversarial frames...")
    frames = np.load(DATA_DIR / 'frames_model.npy')  # (600, 3, 384, 512) uint8
    seg_maps = np.load(DATA_DIR / 'seg_maps.npy')      # (600, 384, 512)
    pose_vectors = np.load(DATA_DIR / 'pose_vectors.npy')  # (600, 6)
    N, C, H, W = frames.shape
    print(f"Frames: {frames.shape} uint8")

    # Write raw video for ffmpeg
    raw_path = DATA_DIR / 'frames_raw.rgb'
    print("Writing raw RGB for ffmpeg...")
    frames_to_raw_video(frames, raw_path)

    # First, evaluate the ORIGINAL frames (before any compression)
    print("\n--- Evaluating ORIGINAL adversarial frames (no compression) ---")
    seg_orig, pose_orig = evaluate_frames(frames, seg_maps, pose_vectors, device)
    print(f"  Original: seg={seg_orig:.6f} pose={pose_orig:.6f}")
    print(f"  Score components: 100*seg={100*seg_orig:.4f} "
          f"sqrt(10*pose)={math.sqrt(10*pose_orig):.4f}")

    # Test various codec settings
    print(f"\n{'='*70}")
    print(f"{'Codec':<20} {'CRF':>4} {'Size':>10} {'KB':>8} {'seg':>10} "
          f"{'pose':>10} {'100*seg':>8} {'sqrt10p':>8} {'25*rate':>8} {'SCORE':>8}")
    print(f"{'='*70}")

    uncompressed_size = 37_545_489
    data_payload = 0  # phase3 doesn't need seg.bin/pose.bin in archive

    results = []

    for codec, encode_fn, crfs in [
        ('h264', encode_h264, [10, 14, 18, 22, 26, 30]),
        ('h265', encode_h265, [10, 14, 18, 22, 26, 30]),
    ]:
        for crf in crfs:
            vid_path = DATA_DIR / f'test_{codec}_crf{crf}.mkv'
            try:
                size = encode_fn(raw_path, vid_path, N, H, W, crf=crf)
            except Exception as e:
                print(f"  {codec} crf={crf}: FAILED ({e})")
                continue

            # Decode and evaluate
            try:
                decoded = decode_video(vid_path, N, H, W)
            except Exception as e:
                print(f"  {codec} crf={crf}: decode FAILED ({e})")
                continue

            seg_d, pose_d = evaluate_frames(decoded, seg_maps, pose_vectors, device)

            archive_size = data_payload + size
            rate = archive_size / uncompressed_size
            score = 100 * seg_d + math.sqrt(10 * pose_d) + 25 * rate

            print(f"  {codec:<18} {crf:>4} {size:>10,} {size/1024:>8.1f} "
                  f"{seg_d:>10.6f} {pose_d:>10.4f} {100*seg_d:>8.4f} "
                  f"{math.sqrt(10*pose_d):>8.4f} {25*rate:>8.4f} {score:>8.4f}",
                  flush=True)

            results.append((codec, crf, size, seg_d, pose_d, score))

            # Clean up
            vid_path.unlink(missing_ok=True)

    # Summary
    if results:
        results.sort(key=lambda x: x[5])
        print(f"\n{'='*70}")
        print(f"BEST: {results[0][0]} crf={results[0][1]} "
              f"size={results[0][2]/1024:.1f}KB score={results[0][5]:.4f}")
        print(f"Current best on leaderboard: 1.95")
        if results[0][5] < 1.95:
            print(f"*** BEATS THE LEADERBOARD ***")
        else:
            print(f"Gap to beat: {results[0][5] - 1.95:.4f}")

    # Cleanup
    raw_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
