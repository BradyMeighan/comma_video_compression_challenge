#!/usr/bin/env python
"""
Hypothesis 3: SegNet boundary concentration research.

Analyzes where SegNet misclassifications occur and whether they concentrate
at class boundaries. Quantifies the potential score improvement from fixing
targeted pixels.
"""
import torch, numpy as np, time
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, segnet_sd_path
from collections import Counter

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

# Also load adversarial_decode frames
adv_raw = np.memmap('submissions/adversarial_decode/inflated/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))

model_H, model_W = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512

print("=" * 70)
print("HYPOTHESIS 3: SegNet Boundary Concentration")
print("=" * 70)

# ── Full analysis over ALL 600 odd frames ──
print("\n--- Running SegNet on all 600 frame pairs ---")

batch_size = 32
all_mismatch_counts = []
all_mismatch_fracs = []
all_boundary_counts = []
all_boundary_mismatch_counts = []
all_interior_mismatch_counts = []
class_pair_confusion = Counter()  # which class pairs are confused most

# For spatial heatmap
mismatch_heatmap = torch.zeros(model_H, model_W, device=device)

# For per-frame sorted pixel analysis
per_frame_mismatch_pixels = []

# Track adversarial decode performance too
adv_mismatch_counts = []
adv_mismatch_fracs = []

t0 = time.time()

# Process in batches for efficiency
n_pairs = 600
for batch_start in range(0, n_pairs, batch_size):
    batch_end = min(batch_start + batch_size, n_pairs)
    B = batch_end - batch_start

    # Load odd frames (frame index 1 in each pair)
    orig_frames = []
    comp_frames = []
    adv_frames = []
    for pair_idx in range(batch_start, batch_end):
        odd_idx = pair_idx * 2 + 1

        orig_frames.append(torch.from_numpy(orig_raw[odd_idx].copy()))
        comp_frames.append(torch.from_numpy(comp_raw[odd_idx].copy()))
        adv_frames.append(torch.from_numpy(adv_raw[odd_idx].copy()))

    orig_batch = torch.stack(orig_frames).float().to(device).permute(0, 3, 1, 2)  # B,3,H,W
    comp_batch = torch.stack(comp_frames).float().to(device).permute(0, 3, 1, 2)
    adv_batch = torch.stack(adv_frames).float().to(device).permute(0, 3, 1, 2)

    with torch.no_grad():
        # Downscale to model res
        orig_ds = F.interpolate(orig_batch, size=(model_H, model_W), mode='bilinear')
        comp_ds = F.interpolate(comp_batch, size=(model_H, model_W), mode='bilinear')
        adv_ds = F.interpolate(adv_batch, size=(model_H, model_W), mode='bilinear')

        # Run SegNet
        pred_orig = segnet(orig_ds).argmax(dim=1)  # B, H, W
        pred_comp = segnet(comp_ds).argmax(dim=1)
        pred_adv = segnet(adv_ds).argmax(dim=1)

        # Mismatches
        mismatch = (pred_orig != pred_comp)  # B, H, W
        mismatch_adv = (pred_orig != pred_adv)

        # Boundary detection: pixel is at boundary if any of its 4 neighbors has different class (original)
        # Pad to handle edges
        padded = F.pad(pred_orig.float().unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
        center = pred_orig
        up = padded[:, :-2, 1:-1].long()
        down = padded[:, 2:, 1:-1].long()
        left = padded[:, 1:-1, :-2].long()
        right = padded[:, 1:-1, 2:].long()
        is_boundary = ((center != up) | (center != down) | (center != left) | (center != right))

        for b in range(B):
            mm = mismatch[b]
            mm_adv = mismatch_adv[b]
            bnd = is_boundary[b]

            n_mismatch = mm.sum().item()
            n_boundary = bnd.sum().item()
            n_boundary_mismatch = (mm & bnd).sum().item()
            n_interior_mismatch = (mm & ~bnd).sum().item()

            all_mismatch_counts.append(n_mismatch)
            all_mismatch_fracs.append(mm.float().mean().item())
            all_boundary_counts.append(n_boundary)
            all_boundary_mismatch_counts.append(n_boundary_mismatch)
            all_interior_mismatch_counts.append(n_interior_mismatch)

            adv_mismatch_counts.append(mm_adv.sum().item())
            adv_mismatch_fracs.append(mm_adv.float().mean().item())

            # Accumulate spatial heatmap
            mismatch_heatmap += mm.float()

            # Track class pair confusion
            if n_mismatch > 0:
                orig_classes = pred_orig[b][mm].cpu().numpy()
                comp_classes = pred_comp[b][mm].cpu().numpy()
                for oc, cc in zip(orig_classes, comp_classes):
                    class_pair_confusion[(int(oc), int(cc))] += 1

    if (batch_start // batch_size) % 5 == 0:
        elapsed = time.time() - t0
        print(f"  Processed {batch_end}/{n_pairs} pairs ({elapsed:.1f}s)")

total_time = time.time() - t0
print(f"  Total processing time: {total_time:.1f}s")

# ── Report results ──
print("\n" + "=" * 70)
print("RESULTS: av1_repro vs original")
print("=" * 70)

total_pixels = model_H * model_W
print(f"\nTotal pixels per frame (at model res): {total_pixels:,}")
print(f"Total frames analyzed: {n_pairs}")

print(f"\n--- Mismatch statistics ---")
print(f"  Mean mismatches/frame: {np.mean(all_mismatch_counts):.1f}")
print(f"  Median mismatches/frame: {np.median(all_mismatch_counts):.1f}")
print(f"  Std: {np.std(all_mismatch_counts):.1f}")
print(f"  Min: {np.min(all_mismatch_counts)}")
print(f"  Max: {np.max(all_mismatch_counts)}")
print(f"  Mean mismatch fraction (seg_dist): {np.mean(all_mismatch_fracs):.8f}")
print(f"  Total mismatched pixels across all frames: {sum(all_mismatch_counts):,}")

print(f"\n--- Boundary analysis ---")
print(f"  Mean boundary pixels/frame: {np.mean(all_boundary_counts):.1f}")
print(f"  Mean boundary mismatch pixels/frame: {np.mean(all_boundary_mismatch_counts):.1f}")
print(f"  Mean interior mismatch pixels/frame: {np.mean(all_interior_mismatch_counts):.1f}")
boundary_mismatch_frac = sum(all_boundary_mismatch_counts) / max(sum(all_mismatch_counts), 1)
print(f"  Fraction of mismatches at boundaries: {boundary_mismatch_frac:.4f} ({boundary_mismatch_frac*100:.1f}%)")
boundary_frac = np.mean(all_boundary_counts) / total_pixels
print(f"  Boundaries are {boundary_frac*100:.1f}% of all pixels but contain {boundary_mismatch_frac*100:.1f}% of errors")
enrichment = boundary_mismatch_frac / boundary_frac if boundary_frac > 0 else 0
print(f"  Enrichment factor: {enrichment:.1f}x")

print(f"\n--- Class pair confusion (top 10) ---")
for (orig_cls, comp_cls), count in class_pair_confusion.most_common(10):
    print(f"  Class {orig_cls} -> {comp_cls}: {count:,} pixels")

print(f"\n--- Adversarial decode performance ---")
print(f"  Mean mismatches/frame: {np.mean(adv_mismatch_counts):.1f}")
print(f"  Mean seg_dist: {np.mean(adv_mismatch_fracs):.8f}")
print(f"  Improvement over av1_repro: {(np.mean(all_mismatch_fracs) - np.mean(adv_mismatch_fracs)) / np.mean(all_mismatch_fracs) * 100:.1f}%")

# ── Spatial distribution ──
print(f"\n--- Spatial heatmap statistics ---")
hm = mismatch_heatmap.cpu().numpy()
print(f"  Max mismatches at any pixel location: {hm.max():.0f} (out of {n_pairs} frames)")
print(f"  Pixels that are mismatched in >50% of frames: {(hm > n_pairs/2).sum()}")
print(f"  Pixels that are mismatched in >10% of frames: {(hm > n_pairs/10).sum()}")
print(f"  Pixels that are mismatched in >1% of frames: {(hm > n_pairs/100).sum()}")
print(f"  Pixels never mismatched: {(hm == 0).sum()} ({(hm == 0).sum()/total_pixels*100:.1f}%)")

# Spatial row distribution
row_mismatches = hm.sum(axis=1)
print(f"\n  Top 10 rows by total mismatches:")
top_rows = np.argsort(-row_mismatches)[:10]
for r in top_rows:
    print(f"    Row {r}: {row_mismatches[r]:.0f} total mismatches")

# ── Score impact analysis ──
print("\n" + "=" * 70)
print("SCORE IMPACT ANALYSIS")
print("=" * 70)

current_seg_dist = np.mean(all_mismatch_fracs)
adv_seg_dist = np.mean(adv_mismatch_fracs)

# If we could fix the top K mismatched pixels per frame
print("\n--- Impact of fixing top-K pixels per frame ---")
for k in [10, 50, 100, 200, 500, 1000]:
    # Each frame has some mismatches; fixing K of them reduces count by min(K, count)
    fixed_counts = [max(0, count - k) for count in all_mismatch_counts]
    fixed_seg_dist = np.mean([c / total_pixels for c in fixed_counts])
    seg_term_before = 100 * current_seg_dist
    seg_term_after = 100 * fixed_seg_dist
    improvement = seg_term_before - seg_term_after
    print(f"  Fix top {k:4d} pixels/frame: seg_dist {current_seg_dist:.6f} -> {fixed_seg_dist:.6f}, "
          f"score improvement: {improvement:.4f}")

print(f"\n--- If ALL mismatches were fixed ---")
print(f"  seg_dist: {current_seg_dist:.6f} -> 0.000000")
print(f"  Score term change: -{100 * current_seg_dist:.4f}")

# For adversarial decode
print(f"\n--- Adversarial decode analysis ---")
for k in [10, 50, 100, 200, 500]:
    fixed_counts = [max(0, count - k) for count in adv_mismatch_counts]
    fixed_seg_dist = np.mean([c / total_pixels for c in fixed_counts])
    seg_term_before = 100 * adv_seg_dist
    seg_term_after = 100 * fixed_seg_dist
    improvement = seg_term_before - seg_term_after
    print(f"  Fix top {k:4d} pixels/frame: seg_dist {adv_seg_dist:.8f} -> {fixed_seg_dist:.8f}, "
          f"score improvement: {improvement:.6f}")

# ── Spatial clustering of errors ──
print("\n--- Error clustering analysis ---")
# Use connected components to find "problem regions"
from scipy import ndimage

n_regions_per_frame = []
region_sizes = []
# Sample 100 frames
for pair_idx in range(0, min(100, n_pairs)):
    odd_idx = pair_idx * 2 + 1
    frame_o = torch.from_numpy(orig_raw[odd_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)
    frame_c = torch.from_numpy(comp_raw[odd_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred_o = segnet(F.interpolate(frame_o, size=(model_H, model_W), mode='bilinear')).argmax(dim=1)
        pred_c = segnet(F.interpolate(frame_c, size=(model_H, model_W), mode='bilinear')).argmax(dim=1)

    mm = (pred_o != pred_c)[0].cpu().numpy().astype(np.uint8)
    labeled, n_regions = ndimage.label(mm)
    n_regions_per_frame.append(n_regions)

    if n_regions > 0:
        for region_id in range(1, n_regions + 1):
            region_sizes.append((labeled == region_id).sum())

print(f"  Over 100 frames:")
print(f"  Mean regions/frame: {np.mean(n_regions_per_frame):.1f}")
print(f"  Median regions/frame: {np.median(n_regions_per_frame):.1f}")
print(f"  Max regions in a frame: {np.max(n_regions_per_frame)}")
if region_sizes:
    print(f"  Mean region size: {np.mean(region_sizes):.1f} pixels")
    print(f"  Median region size: {np.median(region_sizes):.1f} pixels")
    print(f"  Max region size: {np.max(region_sizes)} pixels")
    # Size distribution
    sizes = np.array(region_sizes)
    for thresh in [1, 5, 10, 50, 100]:
        pct = (sizes <= thresh).sum() / len(sizes) * 100
        print(f"  Regions <= {thresh:3d} pixels: {pct:.1f}%")

# ── Cost to store model-res corrections ──
print("\n--- Cost analysis for model-res corrections ---")
mean_mm_per_frame = np.mean(all_mismatch_counts)
# For each mismatched pixel we need: position (2 bytes at model-res) + correct class (1 byte)
# Or we could store a sparse map
bytes_per_correction = 3  # row(9bit) + col(9bit) + class(3bit) ~ 3 bytes packed
total_correction_bytes = mean_mm_per_frame * bytes_per_correction * n_pairs
print(f"  Mean mismatched pixels per frame: {mean_mm_per_frame:.0f}")
print(f"  Bytes per correction (packed): {bytes_per_correction}")
print(f"  Total for {n_pairs} frames: {total_correction_bytes/1024:.1f} KB uncompressed")
print(f"  With entropy coding (~50% reduction): ~{total_correction_bytes/1024/2:.1f} KB")

# For adversarial decode
adv_mean_mm = np.mean(adv_mismatch_counts)
total_adv_correction = adv_mean_mm * bytes_per_correction * n_pairs
print(f"\n  Adversarial decode mismatched pixels per frame: {adv_mean_mm:.0f}")
print(f"  Total for {n_pairs} frames: {total_adv_correction/1024:.1f} KB uncompressed")

# How much score would we save?
print(f"\n--- Score budget analysis ---")
# Current score ≈ 1.23
# score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate
# If we reduce seg_dist to 0: save 100 * seg_dist points
# Cost: need to store correction data in archive
# Additional bytes in archive -> higher rate -> higher 25*rate term

# Currently for adversarial decode approach:
print(f"  Current seg_dist (av1_repro): {current_seg_dist:.8f}")
print(f"  100 * seg_dist component: {100*current_seg_dist:.4f}")
print(f"  Current seg_dist (adversarial): {adv_seg_dist:.8f}")
print(f"  100 * seg_dist component: {100*adv_seg_dist:.4f}")

# Original uncompressed size for rate calculation
orig_size = 1200 * H_cam * W_cam * 3  # raw RGB
print(f"\n  Original uncompressed size: {orig_size:,} bytes")
print(f"  25 * rate per KB additional: {25 * 1024 / orig_size:.6f}")

# If we store full correction map at model-res:
correction_sizes = [1024, 5*1024, 10*1024, 50*1024, 100*1024]
for cs in correction_sizes:
    rate_cost = 25 * cs / orig_size
    print(f"  +{cs/1024:.0f}KB archive -> +{rate_cost:.6f} score (rate term)")

del orig_raw, comp_raw, adv_raw
print("\n--- Hypothesis 3 research complete ---")
