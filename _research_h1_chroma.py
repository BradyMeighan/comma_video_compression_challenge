#!/usr/bin/env python
"""
Hypothesis 1: PoseNet chroma insensitivity research.

Tests whether PoseNet is invariant to per-pixel chroma variations within 2x2 blocks
as long as the block average is preserved (since rgb_to_yuv6 subsamples U,V by averaging 2x2).
"""
import torch, numpy as np, time
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size, rgb_to_yuv6
from modules import PoseNet, posenet_sd_path

device = torch.device('cuda')
W_cam, H_cam = camera_size  # 1164, 874

# Load PoseNet
posenet = PoseNet().eval().to(device)
posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
for p in posenet.parameters():
    p.requires_grad_(False)

# Load original video frames
orig_raw = np.memmap('videos/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))
# Load compressed frames (av1_repro as baseline)
comp_raw = np.memmap('submissions/av1_repro/inflated/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))

print("=" * 70)
print("HYPOTHESIS 1: PoseNet Chroma Insensitivity")
print("=" * 70)

# ── Test 1: Verify that rgb_to_yuv6 destroys per-pixel chroma info ──
print("\n--- Test 1: Mathematical verification of chroma subsampling ---")

# Take a sample frame, convert to float
sample = torch.from_numpy(orig_raw[0].copy()).float().permute(2, 0, 1).unsqueeze(0).to(device)  # 1,3,H,W

# Compute YUV6
yuv6_orig = rgb_to_yuv6(sample)
print(f"Input shape: {sample.shape} -> YUV6 shape: {yuv6_orig.shape}")
print(f"YUV6 channels: y00, y10, y01, y11 (4 luma at half-res), U_sub, V_sub (chroma at half-res)")

# Perturb chroma within 2x2 blocks: change individual R,G,B pixels but keep 2x2 block averages the same
# Strategy: for each 2x2 block, add [+d, -d, -d, +d] pattern to blue channel
perturbed = sample.clone()
d = 20.0  # significant perturbation
# Apply to B channel (index 2) which affects U strongly
perturbed[:, 2, 0::2, 0::2] += d
perturbed[:, 2, 1::2, 0::2] -= d
perturbed[:, 2, 0::2, 1::2] -= d
perturbed[:, 2, 1::2, 1::2] += d
perturbed = perturbed.clamp(0, 255)

yuv6_perturbed = rgb_to_yuv6(perturbed)

# Check if U_sub and V_sub channels changed
u_diff = (yuv6_orig[:, 4] - yuv6_perturbed[:, 4]).abs().max().item()
v_diff = (yuv6_orig[:, 5] - yuv6_perturbed[:, 5]).abs().max().item()
# Check Y channels (these WILL change because Y depends on B)
y_diffs = [(yuv6_orig[:, i] - yuv6_perturbed[:, i]).abs().max().item() for i in range(4)]

print(f"\nPerturbation: +/-{d} to B channel in checkerboard within 2x2 blocks")
print(f"U_sub max diff: {u_diff:.6f}")
print(f"V_sub max diff: {v_diff:.6f}")
print(f"Y channel max diffs: {[f'{d:.6f}' for d in y_diffs]}")
print(f"NOTE: Y channels change because Y = 0.299R + 0.587G + 0.114B depends on per-pixel B")

# ── Test 2: The correct chroma-preserving perturbation ──
print("\n--- Test 2: Chroma-only perturbation (keep Y constant, vary U/V within blocks) ---")
# To keep U_sub unchanged but vary per-pixel U:
# U = (B - Y) / 1.772 + 128
# If we add delta_R and delta_B to keep Y constant: 0.299*dR + 0.114*dB = 0
# So dR = -(0.114/0.299)*dB
# Then dU = (dB - dY) / 1.772 = dB / 1.772  (since dY=0 not achievable without affecting all channels)
#
# Actually, the key insight: PoseNet's input goes through rgb_to_yuv6 AFTER bilinear downscale to 384x512.
# Let me trace the full preprocessing pipeline.

print("\n--- Test 3: Full PoseNet preprocessing pipeline trace ---")
print("PoseNet preprocessing steps:")
print("  1. Input: (B, 2, 3, 874, 1164)")
print("  2. Reshape to (B*2, 3, 874, 1164)")
print("  3. Bilinear interpolate to (B*2, 3, 384, 512)")
print("  4. rgb_to_yuv6 -> (B*2, 6, 192, 256)")
print("  5. Rearrange to (B, 12, 192, 256)")
print("  6. Normalize: (x - 127.5) / 63.75")

# So the bilinear downscale happens BEFORE yuv6 conversion.
# At model resolution (384x512), the 2x2 block averaging in yuv6 produces 192x256 chroma.

# ── Test 4: Run PoseNet on original vs perturbed frames ──
print("\n--- Test 4: PoseNet output sensitivity to chroma perturbations ---")

# Process 20 frame pairs
n_test = 20
pose_diffs = []
max_pixel_diffs = []

for pair_idx in range(n_test):
    f0_idx = pair_idx * 2
    f1_idx = pair_idx * 2 + 1

    # Original pair
    f0 = torch.from_numpy(orig_raw[f0_idx].copy()).float().to(device)  # H,W,3
    f1 = torch.from_numpy(orig_raw[f1_idx].copy()).float().to(device)

    # Stack as (1, 2, 3, H, W)
    pair_orig = torch.stack([f0, f1]).unsqueeze(0).permute(0, 1, 4, 2, 3)  # 1,2,3,H,W

    # Preprocess and run
    with torch.no_grad():
        pn_in_orig = posenet.preprocess_input(pair_orig)
        out_orig = posenet(pn_in_orig)['pose']

    # Perturb chroma at MODEL resolution (post-bilinear-downscale)
    # First do the bilinear downscale
    flat = pair_orig.reshape(2, 3, H_cam, W_cam)
    downscaled = F.interpolate(flat, size=(384, 512), mode='bilinear')  # 2,3,384,512

    # Now perturb B channel within 2x2 blocks (keeping block average constant)
    perturbed_ds = downscaled.clone()
    d = 30.0
    perturbed_ds[:, 2, 0::2, 0::2] += d
    perturbed_ds[:, 2, 1::2, 0::2] -= d
    perturbed_ds[:, 2, 0::2, 1::2] -= d
    perturbed_ds[:, 2, 1::2, 1::2] += d
    perturbed_ds = perturbed_ds.clamp(0, 255)

    # Convert both to yuv6
    with torch.no_grad():
        yuv6_clean = rgb_to_yuv6(downscaled)
        yuv6_pert = rgb_to_yuv6(perturbed_ds)

    # Check how much YUV6 changed
    yuv6_diff = (yuv6_clean - yuv6_pert).abs()

    # Now run PoseNet on perturbed version (need to upscale back to camera_size, then re-preprocess)
    # Actually, let's inject at the right point: after preprocessing
    perturbed_yuv6 = rgb_to_yuv6(perturbed_ds)  # 2,6,192,256
    pn_in_pert = perturbed_yuv6.reshape(1, 12, 192, 256)

    with torch.no_grad():
        out_pert = posenet(pn_in_pert)['pose']

    pose_diff = (out_orig[:, :6] - out_pert[:, :6]).pow(2).sum().sqrt().item()
    pose_diffs.append(pose_diff)

    pixel_diff = yuv6_diff.max().item()
    max_pixel_diffs.append(pixel_diff)

print(f"\nResults over {n_test} frame pairs:")
print(f"  Chroma perturbation: +/-{d} to B channel in 2x2 checkerboard pattern")
print(f"  Max YUV6 pixel difference: {max(max_pixel_diffs):.4f}")
print(f"  Mean pose L2 distance: {np.mean(pose_diffs):.8f}")
print(f"  Max pose L2 distance: {np.max(pose_diffs):.8f}")
print(f"  Min pose L2 distance: {np.min(pose_diffs):.8f}")

# ── Test 5: What if we zero-mean ONLY the U,V within 2x2 blocks? ──
print("\n--- Test 5: Chroma quantized to block averages (exact invariance test) ---")

exact_diffs = []
for pair_idx in range(n_test):
    f0_idx = pair_idx * 2
    f1_idx = pair_idx * 2 + 1

    f0 = torch.from_numpy(orig_raw[f0_idx].copy()).float().to(device)
    f1 = torch.from_numpy(orig_raw[f1_idx].copy()).float().to(device)

    pair = torch.stack([f0, f1]).unsqueeze(0).permute(0, 1, 4, 2, 3)

    with torch.no_grad():
        pn_in_orig = posenet.preprocess_input(pair)
        out_orig = posenet(pn_in_orig)['pose']

    # Downscale to model res
    flat = pair.reshape(2, 3, H_cam, W_cam)
    ds = F.interpolate(flat, size=(384, 512), mode='bilinear')

    # Compute YUV
    R, G, B = ds[:, 0], ds[:, 1], ds[:, 2]
    Y = R * 0.299 + G * 0.587 + B * 0.114
    U = (B - Y) / 1.772 + 128.0
    V = (R - Y) / 1.402 + 128.0

    # Replace per-pixel U,V with 2x2 block average (this is what yuv6 does)
    U_avg = (U[:, 0::2, 0::2] + U[:, 1::2, 0::2] + U[:, 0::2, 1::2] + U[:, 1::2, 1::2]) * 0.25
    V_avg = (V[:, 0::2, 0::2] + V[:, 1::2, 0::2] + V[:, 0::2, 1::2] + V[:, 1::2, 1::2]) * 0.25

    # Reconstruct RGB from Y (per-pixel) + U_avg, V_avg (block average)
    # Expand averages back to full res
    U_flat = U_avg.unsqueeze(-1).unsqueeze(-2).expand(-1, -1, 1, -1, 1).reshape(2, 192, 1, 256, 1)
    # Actually simpler: just repeat
    U_block = U_avg.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)  # 2,384,512
    V_block = V_avg.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)

    # Now reconstruct RGB: R = Y + 1.402*(V - 128), B = Y + 1.772*(U - 128), G computed from Y
    R_new = Y + 1.402 * (V_block - 128.0)
    B_new = Y + 1.772 * (U_block - 128.0)
    G_new = (Y - 0.299 * R_new - 0.114 * B_new) / 0.587

    ds_reconstructed = torch.stack([R_new, G_new, B_new], dim=1).clamp(0, 255)

    # Check the YUV6 of reconstructed vs original
    yuv6_orig = rgb_to_yuv6(ds)
    yuv6_recon = rgb_to_yuv6(ds_reconstructed)

    u_ch_diff = (yuv6_orig[:, 4] - yuv6_recon[:, 4]).abs().max().item()
    v_ch_diff = (yuv6_orig[:, 5] - yuv6_recon[:, 5]).abs().max().item()
    y_ch_diff = max((yuv6_orig[:, i] - yuv6_recon[:, i]).abs().max().item() for i in range(4))

    # Run PoseNet on reconstructed
    pn_in_recon = yuv6_recon.reshape(1, 12, 192, 256)
    with torch.no_grad():
        out_recon = posenet(pn_in_recon)['pose']

    exact_diff = (out_orig[:, :6] - out_recon[:, :6]).pow(2).sum().sqrt().item()
    exact_diffs.append(exact_diff)

print(f"\nBlock-average chroma reconstruction:")
print(f"  Mean pose L2 diff: {np.mean(exact_diffs):.10f}")
print(f"  Max pose L2 diff: {np.max(exact_diffs):.10f}")
print(f"  (If ~0, PoseNet is completely invariant to within-block chroma)")

# ── Test 6: Direct YUV6 channel sensitivity ──
print("\n--- Test 6: PoseNet sensitivity per YUV6 channel ---")

# For a frame pair, perturb each of the 6 YUV6 channels and measure output change
pair_idx = 0
f0 = torch.from_numpy(orig_raw[0].copy()).float().to(device)
f1 = torch.from_numpy(orig_raw[1].copy()).float().to(device)
pair = torch.stack([f0, f1]).unsqueeze(0).permute(0, 1, 4, 2, 3)

with torch.no_grad():
    pn_in = posenet.preprocess_input(pair)
    out_base = posenet(pn_in)['pose']

channel_names = ['y00', 'y10', 'y01', 'y11', 'U_sub', 'V_sub']
for pert_mag in [1.0, 5.0, 10.0]:
    print(f"\n  Perturbation magnitude: {pert_mag}")
    for ch in range(6):
        pn_pert = pn_in.clone()
        # Perturb channel ch for frame 0 (channels 0-5) and frame 1 (channels 6-11)
        pn_pert[:, ch] += pert_mag
        pn_pert[:, ch + 6] += pert_mag
        with torch.no_grad():
            out_pert = posenet(pn_pert)['pose']
        diff = (out_base[:, :6] - out_pert[:, :6]).pow(2).sum().sqrt().item()
        print(f"    {channel_names[ch]:6s}: pose L2 = {diff:.8f}")

# ── Test 7: Entropy savings estimation ──
print("\n\n--- Test 7: Entropy savings from chroma subsampling ---")

# If PoseNet only sees 2x2 block-averaged chroma, and SegNet doesn't use chroma differently...
# Actually SegNet takes RGB input (no YUV conversion), so it sees full chroma.
# But the question is about VIDEO ENCODING: if we subsample chroma before encoding, how much do we save?

# Let's measure entropy of full chroma vs 2x2 block chroma at model resolution
from collections import Counter
import math as mathlib

# Sample 50 frames
entropies_full = []
entropies_block = []
for i in range(0, 100, 2):  # 50 odd frames
    frame = torch.from_numpy(orig_raw[i].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)
    ds = F.interpolate(frame, size=(384, 512), mode='bilinear')

    R, G, B = ds[0, 0], ds[0, 1], ds[0, 2]
    Y = R * 0.299 + G * 0.587 + B * 0.114
    U = ((B - Y) / 1.772 + 128.0).clamp(0, 255).round().to(torch.uint8)

    # Full chroma entropy
    u_flat = U.cpu().numpy().flatten()
    counts = Counter(u_flat)
    total = len(u_flat)
    entropy = -sum((c/total) * mathlib.log2(c/total) for c in counts.values())
    entropies_full.append(entropy)

    # Block-averaged chroma entropy (192x256 resolution)
    U_float = U.float()
    U_avg = (U_float[0::2, 0::2] + U_float[1::2, 0::2] + U_float[0::2, 1::2] + U_float[1::2, 1::2]) * 0.25
    U_avg_q = U_avg.round().to(torch.uint8)
    u_block_flat = U_avg_q.cpu().numpy().flatten()
    counts_b = Counter(u_block_flat)
    total_b = len(u_block_flat)
    entropy_b = -sum((c/total_b) * mathlib.log2(c/total_b) for c in counts_b.values())
    entropies_block.append(entropy_b)

print(f"Full chroma (384x512) - Mean entropy: {np.mean(entropies_full):.4f} bits/pixel, {384*512} pixels")
print(f"Block chroma (192x256) - Mean entropy: {np.mean(entropies_block):.4f} bits/pixel, {192*256} pixels")
full_bits = np.mean(entropies_full) * 384 * 512
block_bits = np.mean(entropies_block) * 192 * 256
print(f"Full chroma: ~{full_bits:.0f} bits/frame for U channel")
print(f"Block chroma: ~{block_bits:.0f} bits/frame for U channel")
print(f"Savings per U channel: {(full_bits - block_bits)/full_bits*100:.1f}%")
print(f"(Same savings apply to V channel)")

del orig_raw, comp_raw
print("\n--- Hypothesis 1 research complete ---")
