#!/usr/bin/env python
"""
Hypothesis 2: SegNet resolution exploit research.

Tests whether corrections at 384x512 resolution propagate correctly through
bilinear resize, and quantifies the cost savings.
"""
import torch, numpy as np, time
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, segnet_sd_path

device = torch.device('cuda')
W_cam, H_cam = camera_size  # 1164, 874

# Load SegNet
segnet = SegNet().eval().to(device)
segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
for p in segnet.parameters():
    p.requires_grad_(False)

# Load frames
orig_raw = np.memmap('videos/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))
comp_raw = np.memmap('submissions/av1_repro/inflated/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))

model_H, model_W = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512

print("=" * 70)
print("HYPOTHESIS 2: SegNet Resolution Exploit")
print("=" * 70)

# ── Test 1: Verify SegNet preprocessing is just bilinear downscale ──
print("\n--- Test 1: SegNet preprocessing verification ---")
print(f"  Camera: {W_cam}x{H_cam}")
print(f"  Model input: {model_W}x{model_H}")
print(f"  Downscale factor: {W_cam/model_W:.2f}x width, {H_cam/model_H:.2f}x height")
print(f"  Pixels at model res: {model_W * model_H:,}")
print(f"  Pixels at camera res: {W_cam * H_cam:,}")
print(f"  Ratio: {(W_cam * H_cam) / (model_W * model_H):.2f}x")

# ── Test 2: Apply correction at model res, verify it propagates ──
print("\n--- Test 2: Model-res correction propagation ---")

# Take an odd frame (SegNet evaluates frame index 1 in each pair, which is odd in video)
frame_idx = 1  # second frame (odd)
frame_orig = torch.from_numpy(orig_raw[frame_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
frame_comp = torch.from_numpy(comp_raw[frame_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)

# SegNet preprocessing: bilinear downscale to model res
with torch.no_grad():
    seg_in_orig = F.interpolate(frame_orig, size=(model_H, model_W), mode='bilinear')
    seg_in_comp = F.interpolate(frame_comp, size=(model_H, model_W), mode='bilinear')

    out_orig = segnet(seg_in_orig)
    out_comp = segnet(seg_in_comp)

    pred_orig = out_orig.argmax(dim=1)
    pred_comp = out_comp.argmax(dim=1)

    mismatch = (pred_orig != pred_comp).float()
    print(f"  Frame {frame_idx}: {mismatch.sum().item():.0f} mismatched pixels out of {model_H*model_W}")
    print(f"  Mismatch rate: {mismatch.mean().item():.6f}")

# Now: apply a correction at model res, upscale, and re-run through full pipeline
print("\n  Testing: correction at model-res -> upscale -> full pipeline")

# Find mismatched pixels
mismatch_coords = torch.nonzero(mismatch[0])  # N, 2 (row, col)
print(f"  {len(mismatch_coords)} mismatched pixel locations")

if len(mismatch_coords) > 0:
    # Apply a +5 correction to R channel at model-res for a few mismatched pixels
    correction = torch.zeros(1, 3, model_H, model_W, device=device)
    test_pixels = min(100, len(mismatch_coords))
    for i in range(test_pixels):
        r, c = mismatch_coords[i]
        correction[0, 0, r, c] = 5.0  # small R correction

    # Method A: Apply at model-res directly
    seg_in_corrected_A = seg_in_comp + correction
    with torch.no_grad():
        out_A = segnet(seg_in_corrected_A)
    pred_A = out_A.argmax(dim=1)
    mismatches_A = (pred_orig != pred_A).float().sum().item()

    # Method B: Upscale correction to camera-res, add to camera-res frame, re-downsample
    correction_up = F.interpolate(correction, size=(H_cam, W_cam), mode='bilinear')
    frame_comp_corrected = frame_comp + correction_up
    seg_in_corrected_B = F.interpolate(frame_comp_corrected, size=(model_H, model_W), mode='bilinear')
    with torch.no_grad():
        out_B = segnet(seg_in_corrected_B)
    pred_B = out_B.argmax(dim=1)
    mismatches_B = (pred_orig != pred_B).float().sum().item()

    # Method C: Apply correction directly at model-res (bypassing upscale/downscale)
    # This is the "ideal" case - correction is lossless at model res

    print(f"\n  Baseline mismatches: {mismatch.sum().item():.0f}")
    print(f"  After model-res direct correction: {mismatches_A:.0f}")
    print(f"  After upscale-downscale round-trip: {mismatches_B:.0f}")
    print(f"  Difference (round-trip loss): {abs(mismatches_B - mismatches_A):.0f} pixels")

# ── Test 3: Verify bilinear downscale is exactly invertible ──
print("\n--- Test 3: Bilinear round-trip fidelity ---")

# At model res, create a known correction
correction = torch.zeros(1, 3, model_H, model_W, device=device)
correction[0, :, 100, 100] = 10.0

# Upscale to camera res, then downscale back
correction_up = F.interpolate(correction, size=(H_cam, W_cam), mode='bilinear')
correction_roundtrip = F.interpolate(correction_up, size=(model_H, model_W), mode='bilinear')

rt_error = (correction - correction_roundtrip).abs()
print(f"  Single pixel correction at (100,100): {correction[0, 0, 100, 100].item():.4f}")
print(f"  After round-trip: {correction_roundtrip[0, 0, 100, 100].item():.4f}")
print(f"  Max round-trip error: {rt_error.max().item():.6f}")
print(f"  Mean round-trip error: {rt_error.mean().item():.8f}")
print(f"  Nonzero pixels after round-trip: {(rt_error > 0.001).sum().item()}")

# The correction spreads due to bilinear interpolation!
# Let's measure how much a model-res correction spreads at camera-res
nonzero_up = (correction_up.abs() > 0.001).sum().item()
print(f"\n  1 model-res pixel correction -> {nonzero_up} camera-res pixels affected")
print(f"  Spread factor: {nonzero_up:.0f}x (expected ~{(H_cam/model_H)*(W_cam/model_W):.1f}x)")

# ── Test 4: What matters is the correction at MODEL resolution ──
print("\n--- Test 4: SegNet sees ONLY model-res input ---")
print(f"  Key insight: SegNet's preprocess_input() downscales to ({model_W}x{model_H}) with bilinear")
print(f"  So any correction we apply, SegNet only sees its {model_W}x{model_H} projection")
print(f"  A correction stored at {model_W}x{model_H} = {model_W*model_H*3:,} bytes per frame (3 channels)")
print(f"  A correction stored at {W_cam}x{H_cam} = {W_cam*H_cam*3:,} bytes per frame (3 channels)")
print(f"  Storage savings: {(W_cam*H_cam)/(model_W*model_H):.2f}x")

# ── Test 5: Quantify misclassified pixels across frames ──
print("\n--- Test 5: Misclassified pixel statistics (sample of 50 frames) ---")

n_test = 50
all_mismatches = []
all_mismatch_fracs = []

for pair_idx in range(n_test):
    odd_idx = pair_idx * 2 + 1  # SegNet evaluates odd frames (frame 1 of each pair)

    frame_o = torch.from_numpy(orig_raw[odd_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)
    frame_c = torch.from_numpy(comp_raw[odd_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        seg_o = F.interpolate(frame_o, size=(model_H, model_W), mode='bilinear')
        seg_c = F.interpolate(frame_c, size=(model_H, model_W), mode='bilinear')
        pred_o = segnet(seg_o).argmax(dim=1)
        pred_c = segnet(seg_c).argmax(dim=1)

    mm = (pred_o != pred_c).float()
    all_mismatches.append(mm.sum().item())
    all_mismatch_fracs.append(mm.mean().item())

print(f"  Over {n_test} frames:")
print(f"  Mean mismatched pixels: {np.mean(all_mismatches):.1f} / {model_H*model_W}")
print(f"  Mean mismatch fraction: {np.mean(all_mismatch_fracs):.6f}")
print(f"  Min mismatches: {np.min(all_mismatches):.0f}")
print(f"  Max mismatches: {np.max(all_mismatches):.0f}")
print(f"  Std mismatches: {np.std(all_mismatches):.1f}")
print(f"  Total model-res pixels to store corrections for: {np.sum(all_mismatches):.0f} (across {n_test} frames)")

# Storage calculation
total_correction_pixels = np.sum(all_mismatches)
model_res_bytes = total_correction_pixels * 3  # RGB delta per pixel
full_res_equiv_bytes = total_correction_pixels * 3 * (W_cam*H_cam) / (model_W*model_H)
print(f"\n  Model-res correction storage (raw, uncompressed):")
print(f"    Per mismatch pixel: 3 bytes (RGB) + 3 bytes (coordinates at model-res)")
print(f"    Total: ~{total_correction_pixels:.0f} * 6 = {total_correction_pixels * 6 / 1024:.1f} KB for {n_test} frames")
print(f"    Extrapolated to 600 frames: ~{total_correction_pixels * 6 / 1024 / n_test * 600:.1f} KB")

del orig_raw, comp_raw
print("\n--- Hypothesis 2 research complete ---")
