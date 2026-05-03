#!/usr/bin/env python
"""
Test H.264/H.265 compression of pre-computed adversarial frames.
Encodes f0+f1 as interleaved video, decodes, runs real evaluate.py pipeline.

Usage: python -m submissions.phase3.test_codec
"""
import subprocess, os, sys, struct, math, time
import numpy as np
from pathlib import Path

DATA_DIR = Path('submissions/phase3/0')
PHASE3 = Path('submissions/phase3')

W_CAM, H_CAM = 1164, 874
MODEL_H, MODEL_W = 384, 512


def encode_video(frames_f0, frames_f1, out_path, codec='libx264',
                 crf=18, preset='veryslow', pix_fmt='yuv444p'):
    """Encode interleaved f0,f1 pairs as video."""
    N = frames_f0.shape[0]
    # Interleave: f0_0, f1_0, f0_1, f1_1, ...
    # Shape: (2*N, 3, H, W) -> write as rawvideo
    interleaved = np.empty((2*N, 3, MODEL_H, MODEL_W), dtype=np.uint8)
    interleaved[0::2] = frames_f0
    interleaved[1::2] = frames_f1
    # ffmpeg wants HWC
    hwc = np.ascontiguousarray(interleaved.transpose(0, 2, 3, 1))

    cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{MODEL_W}x{MODEL_H}', '-r', '20',
        '-i', 'pipe:0',
        '-c:v', codec, '-preset', preset, '-crf', str(crf),
        '-pix_fmt', pix_fmt,
        '-g', '60',  # GOP size
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=hwc.tobytes(), capture_output=True)
    if proc.returncode != 0:
        print(f"  ffmpeg error: {proc.stderr.decode()[-200:]}")
        return None
    return os.path.getsize(out_path)


def decode_video(vid_path, expected_frames):
    """Decode video back to numpy array."""
    cmd = [
        'ffmpeg', '-y', '-i', str(vid_path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        'pipe:1',
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        print(f"  decode error: {proc.stderr.decode()[-200:]}")
        return None
    data = proc.stdout
    frames = np.frombuffer(data, dtype=np.uint8).reshape(expected_frames, MODEL_H, MODEL_W, 3)
    return np.ascontiguousarray(frames.transpose(0, 3, 1, 2))  # NCHW


def build_raw_output(frames_f0, frames_f1, raw_path):
    """Upscale and write .raw file for evaluate.py."""
    import torch, torch.nn.functional as F
    N = frames_f0.shape[0]
    with open(raw_path, 'wb') as fout:
        bs = 32
        for i in range(0, N, bs):
            end = min(i + bs, N)
            f0 = torch.from_numpy(frames_f0[i:end]).float()
            f1 = torch.from_numpy(frames_f1[i:end]).float()
            f0_up = F.interpolate(f0, (H_CAM, W_CAM), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round().byte()
            f1_up = F.interpolate(f1, (H_CAM, W_CAM), mode='bicubic',
                                  align_corners=False).clamp(0, 255).round().byte()
            for b in range(end - i):
                fout.write(f0_up[b].permute(1, 2, 0).contiguous().numpy().tobytes())
                fout.write(f1_up[b].permute(1, 2, 0).contiguous().numpy().tobytes())


def run_evaluate(archive_size):
    """Run evaluate.py and parse results."""
    cmd = [
        sys.executable, 'evaluate.py',
        '--submission-dir', str(PHASE3),
        '--uncompressed-dir', './videos',
        '--report', str(PHASE3 / 'report.txt'),
        '--video-names-file', './public_test_video_names.txt',
        '--device', 'cpu',
    ]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = proc.stdout + proc.stderr

    # Parse results
    seg_dist = pose_dist = None
    for line in output.split('\n'):
        if 'SegNet Distortion' in line:
            seg_dist = float(line.split(':')[1].strip())
        if 'PoseNet Distortion' in line:
            pose_dist = float(line.split(':')[1].strip())

    if seg_dist is not None and pose_dist is not None:
        rate = archive_size / 37_545_489
        score = 100 * seg_dist + math.sqrt(10 * pose_dist) + 25 * rate
        return seg_dist, pose_dist, score
    return None, None, None


def main():
    print("Loading frames...")
    f0 = np.load(DATA_DIR / 'frames_f0.npy')
    f1 = np.load(DATA_DIR / 'frames_f1.npy')
    N = f0.shape[0]
    print(f"Loaded {N} pairs")

    # Test configs
    configs = [
        ('libx264', 'yuv444p', 10),
        ('libx264', 'yuv444p', 14),
        ('libx264', 'yuv444p', 18),
        ('libx264', 'yuv444p', 22),
        ('libx264', 'yuv444p', 26),
        ('libx265', 'yuv444p', 10),
        ('libx265', 'yuv444p', 14),
        ('libx265', 'yuv444p', 18),
        ('libx265', 'yuv444p', 22),
        ('libx265', 'yuv444p', 26),
    ]

    print(f"\n{'Codec':<10} {'PxFmt':<10} {'CRF':>4} {'Size':>10} "
          f"{'seg':>10} {'pose':>10} {'SCORE':>8}")
    print("=" * 72)

    os.makedirs(PHASE3 / 'inflated', exist_ok=True)

    for codec, pix_fmt, crf in configs:
        vid_path = DATA_DIR / f'test_{codec}_{crf}.mkv'

        # Encode
        size = encode_video(f0, f1, vid_path, codec=codec, crf=crf, pix_fmt=pix_fmt)
        if size is None:
            print(f"  {codec:<10} {pix_fmt:<10} {crf:>4} ENCODE FAILED")
            continue

        # Decode
        decoded = decode_video(vid_path, 2 * N)
        if decoded is None:
            print(f"  {codec:<10} {pix_fmt:<10} {crf:>4} DECODE FAILED")
            continue

        dec_f0 = decoded[0::2]  # even frames
        dec_f1 = decoded[1::2]  # odd frames

        # Build .raw and evaluate
        raw_path = PHASE3 / 'inflated' / '0.raw'
        build_raw_output(dec_f0, dec_f1, raw_path)

        # Fake archive.zip size for scoring
        # (we just care about the actual video file size)
        seg_d, pose_d, score = run_evaluate(size)

        if score is not None:
            print(f"  {codec:<10} {pix_fmt:<10} {crf:>4} {size:>10,} "
                  f"{seg_d:>10.6f} {pose_d:>10.6f} {score:>8.4f}",
                  flush=True)
        else:
            rate = size / 37_545_489
            print(f"  {codec:<10} {pix_fmt:<10} {crf:>4} {size:>10,} "
                  f"{'eval err':>10} {'':>10} {25*rate:>8.2f}+dist",
                  flush=True)

        vid_path.unlink(missing_ok=True)

    print(f"\nLeaderboard best: 1.95")


if __name__ == '__main__':
    main()
