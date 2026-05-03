#!/usr/bin/env python
"""
Phase 3 Encoder: Run adversarial decode with FULL teacher models offline,
then compress the resulting frames for storage in the archive.

No neural networks needed at decode time -- just decompress and write.

Usage: python -m submissions.phase3.encode <video_path> <output_dir>
"""
import sys, struct, bz2, lzma, time, math
import torch, torch.nn.functional as F
import einops, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import AVVideoDataset, camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path

# Reuse phase2's adversarial decode machinery
from submissions.phase2.inflate import (
    load_targets, optimize_batch, margin_loss,
    rgb_to_yuv6_diff, posenet_preprocess_diff,
)


IDEAL_COLORS = np.array([
    [52.3731, 66.0825, 53.4251],
    [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493],
    [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=np.float32)


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <video_path> <output_dir>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"Device: {device}, AMP: {use_amp}")
    t0 = time.time()

    # ── Step 1: Extract targets (same as phase2 encode) ──────────────
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)
    print(f"Models loaded in {time.time() - t0:.1f}s")

    ds = AVVideoDataset([video_path.name], data_dir=video_path.parent,
                        batch_size=16, device=torch.device('cpu'))
    ds.prepare_data()

    all_seg_maps, all_pose_vectors = [], []
    print("Extracting targets...")
    with torch.inference_mode():
        for path, idx, batch in ds:
            batch = batch.to(device)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
            seg_in = segnet.preprocess_input(x)
            seg_out = segnet(seg_in)
            all_seg_maps.append(seg_out.argmax(dim=1).cpu().numpy().astype(np.uint8))
            pn_in = posenet.preprocess_input(x)
            all_pose_vectors.append(posenet(pn_in)['pose'][:, :6].cpu().numpy())
            print(f"  Batch {idx}: {batch.shape[0]} pairs")

    seg_maps = np.concatenate(all_seg_maps, 0)        # (N, 384, 512)
    pose_vectors = np.concatenate(all_pose_vectors, 0)  # (N, 6)
    num_pairs = seg_maps.shape[0]
    print(f"Extracted {num_pairs} pairs")

    # ── Step 2: Run adversarial decode with teachers ─────────────────
    W_cam, H_cam = camera_size
    model_H, model_W = segnet_model_input_size[1], segnet_model_input_size[0]
    ideal_colors_t = torch.from_numpy(IDEAL_COLORS).float().to(device)

    seg_t = torch.from_numpy(seg_maps).long().to(device)
    pose_t = torch.from_numpy(pose_vectors.copy()).float().to(device)

    batch_size = 32  # 3090 has plenty of VRAM
    TARGET_ITERS = 200  # Fast iteration — test pipeline first

    # Pre-allocate arrays for BOTH frames and save incrementally
    frames_f0 = np.zeros((num_pairs, 3, model_H, model_W), dtype=np.uint8)
    frames_f1 = np.zeros((num_pairs, 3, model_H, model_W), dtype=np.uint8)

    # Save seg/pose immediately (they're already computed)
    np.save(output_dir / 'seg_maps.npy', seg_maps)
    np.save(output_dir / 'pose_vectors.npy', pose_vectors)

    # Check for partial progress (resume support)
    progress_file = output_dir / '_progress.txt'
    start_pair = 0
    if progress_file.exists() and (output_dir / 'frames_f1.npy').exists():
        start_pair = int(progress_file.read_text().strip())
        frames_f0 = np.load(output_dir / 'frames_f0.npy')
        frames_f1 = np.load(output_dir / 'frames_f1.npy')
        print(f"Resuming from pair {start_pair} (loaded existing frames)")

    print(f"\nRunning adversarial decode ({TARGET_ITERS} iters, batch={batch_size})...")
    for batch_start in range(start_pair, num_pairs, batch_size):
        batch_end = min(batch_start + batch_size, num_pairs)
        B = batch_end - batch_start
        t_b = time.time()

        target_seg = seg_t[batch_start:batch_end]
        target_pose = pose_t[batch_start:batch_end]

        f0, f1, stats = optimize_batch(
            segnet, posenet, target_seg, target_pose, ideal_colors_t,
            H_cam, W_cam, device, use_amp,
            num_iters=TARGET_ITERS, lr=1.2, seg_margin=0.1,
            alpha=120.0, beta=0.20,
        )
        # f0, f1 are at camera resolution (H_cam, W_cam), uint8
        # Downscale both back to model resolution for compact storage
        with torch.no_grad():
            f0_model = F.interpolate(
                f0[:B].float(), size=(model_H, model_W), mode='bilinear',
                align_corners=False
            ).round().clamp(0, 255).to(torch.uint8)
            f1_model = F.interpolate(
                f1[:B].float(), size=(model_H, model_W), mode='bilinear',
                align_corners=False
            ).round().clamp(0, 255).to(torch.uint8)
        frames_f0[batch_start:batch_end] = f0_model.cpu().numpy()
        frames_f1[batch_start:batch_end] = f1_model.cpu().numpy()

        # Save after every batch — never lose progress
        np.save(output_dir / 'frames_f0.npy', frames_f0)
        np.save(output_dir / 'frames_f1.npy', frames_f1)
        progress_file.write_text(str(batch_end))

        bi = batch_start // batch_size + 1
        total_b = (num_pairs + batch_size - 1) // batch_size
        elapsed = time.time() - t0
        eta = elapsed / batch_end * (num_pairs - batch_end)
        print(f"  [{bi}/{total_b}] pairs {batch_start}-{batch_end-1} | "
              f"{time.time()-t_b:.1f}s | seg={stats['seg_loss']:.4f} "
              f"pose={stats['pose_loss']:.6f} | saved | ETA {eta:.0f}s",
              flush=True)

    # Cleanup progress file
    progress_file.unlink(missing_ok=True)

    del segnet, posenet, seg_t, pose_t
    torch.cuda.empty_cache()

    print(f"\nAdversarial decode done in {time.time()-t0:.1f}s")
    print(f"Frames f0: {frames_f0.shape}, f1: {frames_f1.shape} uint8")

    # ── Step 3: Compute deltas from flat colors ──────────────────────
    # flat_frame[i] = ideal_colors[seg_maps[i]] at model resolution
    flat_frames = IDEAL_COLORS[seg_maps]  # (N, H, W, 3)
    flat_frames = np.transpose(flat_frames, (0, 3, 1, 2))  # (N, 3, H, W)
    flat_frames = np.round(flat_frames).clip(0, 255).astype(np.uint8)

    deltas = frames_f1.astype(np.int16) - flat_frames.astype(np.int16)  # (N, 3, H, W)

    print(f"\nDelta statistics:")
    print(f"  Range: [{deltas.min()}, {deltas.max()}]")
    print(f"  Mean abs: {np.abs(deltas).mean():.2f}")
    print(f"  Zeros: {(deltas == 0).mean()*100:.1f}%")
    print(f"  |delta| <= 1: {(np.abs(deltas) <= 1).mean()*100:.1f}%")
    print(f"  |delta| <= 3: {(np.abs(deltas) <= 3).mean()*100:.1f}%")
    print(f"  |delta| <= 7: {(np.abs(deltas) <= 7).mean()*100:.1f}%")
    print(f"  |delta| <= 15: {(np.abs(deltas) <= 15).mean()*100:.1f}%")

    # Raw sizes
    raw_f0 = frames_f0.nbytes
    raw_f1 = frames_f1.nbytes
    print(f"\nRaw sizes: f0={raw_f0/1e6:.1f} MB, f1={raw_f1/1e6:.1f} MB, "
          f"total={( raw_f0+raw_f1)/1e6:.1f} MB")
    print(f"Use test_compression.py to test H.264/H.265 lossy encoding.")

    elapsed = time.time() - t0
    print(f"Total encode time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
