#!/usr/bin/env python
"""
Phase 2 Encoder: Extract SegNet argmax maps and PoseNet pose vectors
from the original video, delta-encode, compress, and compute ideal
class colors for initialization at decode time.

Usage: python -m submissions.phase2.encode <video_path> <output_dir>
"""
import sys, struct, bz2, time
import torch, einops, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import AVVideoDataset, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <video_path> <output_dir>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    t0 = time.time()

    # Load models
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)

    print(f"Models loaded in {time.time() - t0:.1f}s")

    # Load video (AVVideoDataset requires CPU device)
    video_name = video_path.name
    data_dir = video_path.parent
    ds = AVVideoDataset(
        [video_name], data_dir=data_dir,
        batch_size=16, device=torch.device('cpu')
    )
    ds.prepare_data()

    all_seg_maps = []
    all_pose_vectors = []

    print("Extracting targets...")
    with torch.inference_mode():
        for path, idx, batch in ds:
            batch = batch.to(device)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()

            # SegNet: takes last frame, downscales, forward, argmax
            seg_in = segnet.preprocess_input(x)
            seg_out = segnet(seg_in)
            seg_maps = seg_out.argmax(dim=1).cpu().numpy().astype(np.uint8)
            all_seg_maps.append(seg_maps)

            # PoseNet: takes both frames, converts to YUV6, forward, take first 6 dims
            pn_in = posenet.preprocess_input(x)
            pn_out = posenet(pn_in)['pose'][:, :6].cpu().numpy()
            all_pose_vectors.append(pn_out)

            print(f"  Batch {idx}: {seg_maps.shape[0]} pairs processed")

    seg_maps = np.concatenate(all_seg_maps, axis=0)       # (N, 384, 512)
    pose_vectors = np.concatenate(all_pose_vectors, axis=0)  # (N, 6)
    num_pairs, orig_H, orig_W = seg_maps.shape

    print(f"Extracted {num_pairs} pairs: seg_maps {seg_maps.shape}, "
          f"pose_vectors {pose_vectors.shape}")
    print(f"Seg map class distribution: {np.bincount(seg_maps.flatten(), minlength=5)}")

    # Ideal class colors (pre-computed via gradient ascent, deterministic)
    ideal_colors = torch.tensor([
        [52.3731, 66.0825, 53.4251],
        [132.6272, 139.2837, 154.6401],
        [0.0000, 58.3693, 200.9493],
        [200.2360, 213.4126, 201.8910],
        [26.8595, 41.0758, 46.1465],
    ], device=device)
    print("Using hardcoded ideal class colors")

    BEST_PERM = (2, 4, 1, 3, 0)
    perm_lut = np.array(BEST_PERM, dtype=np.uint8)

    # Left-prediction residual encoding: store first column raw, then
    # (actual - left_neighbor) % 5 for the rest. Row-interleave + perm + bz2.
    first_col = seg_maps[:, :, 0]  # (n, H)
    residual = ((seg_maps[:, :, 1:].astype(np.int16) - seg_maps[:, :, :-1].astype(np.int16)) % 5).astype(np.uint8)

    fc_ri = perm_lut[np.ascontiguousarray(first_col.T)]  # (H, n)
    res_ri = perm_lut[np.ascontiguousarray(residual.transpose(1, 0, 2))]  # (H, n, W-1)

    # Dominant-value RLE: runs of the most common residual (permuted 0="no change")
    # get shorter encoding before bz2. For the dominant value, emit [dom, count].
    # For all other values, just emit the raw byte.
    dom_val = perm_lut[0]  # class 0 (no change) after permutation
    res_flat = res_ri.tobytes()
    rle_buf = bytearray()
    i = 0
    while i < len(res_flat):
        if res_flat[i] == dom_val:
            count = 0
            while i < len(res_flat) and res_flat[i] == dom_val and count < 255:
                count += 1
                i += 1
            rle_buf.append(dom_val)
            rle_buf.append(count)
        else:
            rle_buf.append(res_flat[i])
            i += 1
    res_rle = bytes(rle_buf)

    fc_compressed = bz2.compress(fc_ri.tobytes(), 2)
    res_compressed = bz2.compress(res_rle, 1)

    print(f"Seg map compression: {seg_maps.nbytes:,} -> fc={len(fc_compressed):,} + res={len(res_compressed):,} bytes")
    print(f"  Dominant-value RLE: {len(res_flat):,} -> {len(res_rle):,} pre-bz2")

    # seg.bin: [n:u32, H:u32, W:u32, flags:u32, perm:5xu8, fc_len:u32] + fc_data + res_data
    # flags: bit 0-1 = compressor, bit 3 = row-interleaved, bit 4 = perm,
    #        bit 5 = dominant-value RLE, bit 6 = left-pred
    seg_flags = 2 | (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6)
    with open(output_dir / 'seg.bin', 'wb') as f:
        f.write(struct.pack('<IIII', num_pairs, orig_H, orig_W, seg_flags))
        f.write(bytes(BEST_PERM))
        f.write(struct.pack('<I', len(fc_compressed)))
        f.write(fc_compressed)
        f.write(res_compressed)

    # pose.bin: byte-plane split (high/low bytes of float16) + zlib
    import zlib as zl
    pose_f16 = pose_vectors.astype(np.float16).tobytes()
    f16_arr = np.frombuffer(pose_f16, dtype=np.uint8)
    hi_bytes = zl.compress(f16_arr[1::2].tobytes(), 9)
    lo_bytes = zl.compress(f16_arr[0::2].tobytes(), 9)

    # pose.bin: [n:u32, dims:u32, flags:u32, hi_len:u32] + hi_data + lo_data
    # flags: bit 0-1 = compressor (0=zlib), bit 2 = float16, bit 5 = byte-plane
    pose_flags = 0 | (1 << 2) | (1 << 5)
    with open(output_dir / 'pose.bin', 'wb') as f:
        f.write(struct.pack('<III', num_pairs, 6, pose_flags))
        f.write(struct.pack('<I', len(hi_bytes)))
        f.write(hi_bytes)
        f.write(lo_bytes)

    seg_file_size = 16 + 5 + 4 + len(fc_compressed) + len(res_compressed)
    pose_file_size = 12 + 4 + len(hi_bytes) + len(lo_bytes)
    total_data = seg_file_size + pose_file_size
    elapsed = time.time() - t0
    uncompressed_size = 37_545_489

    print(f"\n{'='*50}")
    print(f"Encoding complete in {elapsed:.1f}s")
    print(f"  Seg maps (left-pred, dom-RLE, bz2, perm, ri): {seg_file_size:,} bytes")
    print(f"  Pose vectors (byte-plane, zlib, float16): {pose_file_size:,} bytes")
    print(f"  Total data: {total_data:,} bytes ({total_data/1024:.1f} KB)")
    print(f"  Expected rate: {total_data / uncompressed_size:.6f}")
    print(f"  Expected rate contribution: {25 * total_data / uncompressed_size:.3f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
