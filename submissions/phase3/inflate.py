#!/usr/bin/env python
"""
Phase 3 Inflate: Decompress pre-computed adversarial frames → upscale → write .raw
No neural networks used. Just data decompression + image resize.

Usage: python submissions/phase3/inflate.py <data_dir> <output_path>
"""
import sys, struct, subprocess, tempfile
import numpy as np
from pathlib import Path

# Camera resolution from frame_utils
W_CAM, H_CAM = 1164, 874
MODEL_H, MODEL_W = 384, 512


def main():
    data_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find the video file (H.264/H.265 encoded frames)
    video_file = data_dir / 'frames.mkv'
    if not video_file.exists():
        # Fallback: try raw npy (for testing before compression)
        frames_f0 = np.load(data_dir / 'frames_f0.npy')
        frames_f1 = np.load(data_dir / 'frames_f1.npy')
        N = frames_f0.shape[0]
        print(f"Loaded {N} pairs from npy (uncompressed)")

        # Upscale and write
        import torch, torch.nn.functional as F
        with open(output_path, 'wb') as fout:
            batch = 32
            for i in range(0, N, batch):
                end = min(i + batch, N)
                f0 = torch.from_numpy(frames_f0[i:end]).float()
                f1 = torch.from_numpy(frames_f1[i:end]).float()
                f0_up = F.interpolate(f0, (H_CAM, W_CAM), mode='bicubic',
                                      align_corners=False).clamp(0, 255).round().byte()
                f1_up = F.interpolate(f1, (H_CAM, W_CAM), mode='bicubic',
                                      align_corners=False).clamp(0, 255).round().byte()
                for b in range(end - i):
                    fout.write(f0_up[b].permute(1, 2, 0).contiguous().numpy().tobytes())
                    fout.write(f1_up[b].permute(1, 2, 0).contiguous().numpy().tobytes())
                print(f"  Written pairs {i}-{end-1}", flush=True)

        print(f"Done: {N*2} frames -> {output_path}")
        return

    # TODO: ffmpeg decode path for compressed frames
    # (will implement once we pick the best codec/CRF)
    print(f"ERROR: video decode not yet implemented")
    sys.exit(1)


if __name__ == '__main__':
    main()
