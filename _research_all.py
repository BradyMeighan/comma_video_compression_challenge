#!/usr/bin/env python
"""
Comprehensive research on three hypotheses about exploitable properties
of the video compression challenge evaluation pipeline.
"""
import torch, numpy as np, time, sys, math
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size, rgb_to_yuv6
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from collections import Counter
import av

device = torch.device('cuda')
W_cam, H_cam = camera_size  # 1164, 874
model_H, model_W = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512

# ── Load models ──
print("Loading models...")
segnet = SegNet().eval().to(device)
segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
posenet = PoseNet().eval().to(device)
posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
for p in list(segnet.parameters()) + list(posenet.parameters()):
    p.requires_grad_(False)
print("Models loaded.")


def yuv420_to_rgb(frame):
    """Convert PyAV frame to RGB tensor, matching nvdec output."""
    H, W = frame.height, frame.width
    y = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape(H, frame.planes[0].line_size)[:, :W]
    u = np.frombuffer(frame.planes[1], dtype=np.uint8).reshape(H//2, frame.planes[1].line_size)[:, :W//2]
    v = np.frombuffer(frame.planes[2], dtype=np.uint8).reshape(H//2, frame.planes[2].line_size)[:, :W//2]

    y_t = torch.from_numpy(y.copy()).float()
    u_t = torch.from_numpy(u.copy()).float().unsqueeze(0).unsqueeze(0)
    v_t = torch.from_numpy(v.copy()).float().unsqueeze(0).unsqueeze(0)

    u_up = F.interpolate(u_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()
    v_up = F.interpolate(v_t, size=(H, W), mode='bilinear', align_corners=False).squeeze()

    yf = (y_t - 16.0) * (255.0 / 219.0)
    uf = (u_up - 128.0) * (255.0 / 224.0)
    vf = (v_up - 128.0) * (255.0 / 224.0)

    r = (yf + 1.402 * vf).clamp(0, 255)
    g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
    b = (yf + 1.772 * uf).clamp(0, 255)
    return torch.stack([r, g, b], dim=-1).round().to(torch.uint8)


def load_all_frames(path):
    """Load all frames from video file as list of (H, W, 3) uint8 tensors."""
    container = av.open(str(path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        frames.append(yuv420_to_rgb(frame))
    container.close()
    return frames


# ── Load video frames ──
print("Loading original video frames...")
t0 = time.time()
orig_frames = load_all_frames('videos/0.mkv')
print(f"  Loaded {len(orig_frames)} original frames in {time.time()-t0:.1f}s")

print("Loading compressed (av1_repro) frames...")
comp_raw = np.memmap('submissions/av1_repro/inflated/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))
print(f"  Loaded compressed frames from raw file")

print("Loading adversarial_decode frames...")
adv_raw = np.memmap('submissions/adversarial_decode/inflated/0.raw', dtype=np.uint8, mode='r',
                     shape=(1200, H_cam, W_cam, 3))
print(f"  Loaded adversarial frames from raw file")


########################################################################
# HYPOTHESIS 1: PoseNet Chroma Insensitivity
########################################################################
print("\n" + "=" * 70)
print("HYPOTHESIS 1: PoseNet Chroma Insensitivity")
print("=" * 70)

print("\n--- PoseNet preprocessing pipeline ---")
print("  1. Input: (B, 2, 3, 874, 1164) float")
print("  2. Reshape to (B*2, 3, 874, 1164)")
print("  3. Bilinear interpolate to (B*2, 3, 384, 512)")
print("  4. rgb_to_yuv6 -> (B*2, 6, 192, 256)")
print("  5. Rearrange to (B, 12, 192, 256)")
print("  6. Normalize: (x - 127.5) / 63.75")

# Test: YUV6 chroma subsampling invariance
print("\n--- Test 1: Does rgb_to_yuv6 destroy per-pixel chroma info? ---")

# Take frame at model resolution (after bilinear downscale)
frame0 = orig_frames[0].float().to(device).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
frame0_ds = F.interpolate(frame0, size=(model_H, model_W), mode='bilinear')  # 1,3,384,512

yuv6_orig = rgb_to_yuv6(frame0_ds)
print(f"  Input: {frame0_ds.shape} -> YUV6: {yuv6_orig.shape}")

# Perturb B channel within 2x2 blocks keeping block average constant
for d in [5.0, 10.0, 20.0, 40.0]:
    perturbed = frame0_ds.clone()
    # Checkerboard: +d at (0,0) and (1,1), -d at (0,1) and (1,0) within each 2x2 block
    perturbed[:, 2, 0::2, 0::2] += d
    perturbed[:, 2, 1::2, 0::2] -= d
    perturbed[:, 2, 0::2, 1::2] -= d
    perturbed[:, 2, 1::2, 1::2] += d
    perturbed = perturbed.clamp(0, 255)

    yuv6_pert = rgb_to_yuv6(perturbed)

    # Check U and V subsampled channels
    u_diff = (yuv6_orig[:, 4] - yuv6_pert[:, 4]).abs()
    v_diff = (yuv6_orig[:, 5] - yuv6_pert[:, 5]).abs()
    y_diffs = [(yuv6_orig[:, i] - yuv6_pert[:, i]).abs().max().item() for i in range(4)]

    print(f"  d={d:5.1f}: U max diff={u_diff.max().item():.4f}, V max diff={v_diff.max().item():.4f}, "
          f"Y max diffs={[f'{x:.4f}' for x in y_diffs]}")

print("\n  KEY INSIGHT: Blue channel perturbation affects Y (since Y = 0.299R + 0.587G + 0.114B)")
print("  So chroma-preserving perturbation is NOT trivial in RGB space.")
print("  Let's test what actually happens at the U_sub/V_sub level...")

# The TRUE question: does PoseNet only see U_sub, V_sub (block averages)?
# Yes! rgb_to_yuv6 computes:
#   U_sub = average of U over 2x2 block
#   V_sub = average of V over 2x2 block
# These are the ONLY chroma channels that survive.
# So any modification to per-pixel U,V that preserves the 2x2 average is invisible.

# But the question is about RGB encoding: if we quantize chroma in the VIDEO CODEC,
# does it affect PoseNet? Video codecs already do YUV420 which subsamples chroma 2x.
# rgb_to_yuv6 does an ADDITIONAL 2x subsampling on top of that.

# Let's directly test: perturb at YUV6 level to measure sensitivity
print("\n--- Test 2: PoseNet sensitivity to each YUV6 channel ---")

n_test = 20
channel_sensitivities = {ch: [] for ch in range(6)}
channel_names = ['y00', 'y10', 'y01', 'y11', 'U_sub', 'V_sub']

for pair_idx in range(n_test):
    f0 = orig_frames[pair_idx * 2].float().to(device).permute(2, 0, 1).unsqueeze(0)
    f1 = orig_frames[pair_idx * 2 + 1].float().to(device).permute(2, 0, 1).unsqueeze(0)

    pair = torch.cat([f0, f1], dim=0)  # 2,3,H,W
    pair_ds = F.interpolate(pair, size=(model_H, model_W), mode='bilinear')  # 2,3,384,512
    yuv6 = rgb_to_yuv6(pair_ds)  # 2,6,192,256
    pn_in = yuv6.reshape(1, 12, 192, 256)

    with torch.no_grad():
        out_base = posenet(pn_in)['pose'][0, :6].clone()

    for ch in range(6):
        pn_pert = pn_in.clone()
        pn_pert[:, ch] += 1.0      # perturb frame0's channel
        pn_pert[:, ch + 6] += 1.0  # perturb frame1's channel
        with torch.no_grad():
            out_pert = posenet(pn_pert)['pose'][0, :6]
        diff = (out_base - out_pert).pow(2).sum().sqrt().item()
        channel_sensitivities[ch].append(diff)

print(f"\n  PoseNet L2 sensitivity to +1.0 perturbation per channel (mean over {n_test} pairs):")
for ch in range(6):
    mean_s = np.mean(channel_sensitivities[ch])
    max_s = np.max(channel_sensitivities[ch])
    print(f"    {channel_names[ch]:6s}: mean={mean_s:.8f}, max={max_s:.8f}")

# ── Test 3: Compare full-resolution chroma vs block-averaged chroma at PoseNet ──
print("\n--- Test 3: PoseNet invariance to within-block chroma (20 pairs) ---")

pose_mse_diffs = []
for pair_idx in range(n_test):
    f0 = orig_frames[pair_idx * 2].float().to(device).permute(2, 0, 1).unsqueeze(0)
    f1 = orig_frames[pair_idx * 2 + 1].float().to(device).permute(2, 0, 1).unsqueeze(0)

    pair = torch.cat([f0, f1], dim=0)  # 2,3,H,W
    pair_ds = F.interpolate(pair, size=(model_H, model_W), mode='bilinear')

    # Original YUV6
    yuv6_orig = rgb_to_yuv6(pair_ds)  # 2,6,192,256
    pn_in_orig = yuv6_orig.reshape(1, 12, 192, 256)

    # Scramble chroma within 2x2 blocks: swap U pixels but keep average
    pair_scrambled = pair_ds.clone()
    # For each 2x2 block, shuffle the B channel values within the block
    # This changes per-pixel U but preserves U_sub (block average)
    for t in range(2):  # both frames
        B_ch = pair_scrambled[t, 2]  # 384x512
        # Swap (0,0)<->(1,1) and (0,1)<->(1,0) within each 2x2 block
        B00 = B_ch[0::2, 0::2].clone()
        B11 = B_ch[1::2, 1::2].clone()
        B_ch[0::2, 0::2] = B11
        B_ch[1::2, 1::2] = B00

        B01 = B_ch[0::2, 1::2].clone()
        B10 = B_ch[1::2, 0::2].clone()
        B_ch[0::2, 1::2] = B10
        B_ch[1::2, 0::2] = B01

    yuv6_scrambled = rgb_to_yuv6(pair_scrambled)
    pn_in_scrambled = yuv6_scrambled.reshape(1, 12, 192, 256)

    # Compare YUV6
    u_diff = (yuv6_orig[:, 4] - yuv6_scrambled[:, 4]).abs().max().item()
    v_diff = (yuv6_orig[:, 5] - yuv6_scrambled[:, 5]).abs().max().item()
    y_diff = max((yuv6_orig[:, i] - yuv6_scrambled[:, i]).abs().max().item() for i in range(4))

    with torch.no_grad():
        out_orig = posenet(pn_in_orig)['pose'][0, :6]
        out_scrambled = posenet(pn_in_scrambled)['pose'][0, :6]

    pose_diff = (out_orig - out_scrambled).pow(2).sum().sqrt().item()
    pose_mse_diffs.append(pose_diff)

    if pair_idx < 3:
        print(f"  Pair {pair_idx}: U_diff={u_diff:.6f}, V_diff={v_diff:.6f}, Y_diff={y_diff:.6f}, "
              f"pose_L2={pose_diff:.8f}")

print(f"\n  Over {n_test} pairs:")
print(f"    Mean pose L2 diff: {np.mean(pose_mse_diffs):.8f}")
print(f"    Max pose L2 diff: {np.max(pose_mse_diffs):.8f}")
print(f"    Note: This measures the effect of swapping B pixels within 2x2 blocks")
print(f"    Non-zero because Y channel changes (Y depends on B)")

# ── Test 4: What if we directly manipulate U,V at YUV6 level? ──
print("\n--- Test 4: PoseNet output when U_sub/V_sub are zeroed ---")

# Extreme test: zero out all chroma
for pair_idx in range(3):
    f0 = orig_frames[pair_idx * 2].float().to(device).permute(2, 0, 1).unsqueeze(0)
    f1 = orig_frames[pair_idx * 2 + 1].float().to(device).permute(2, 0, 1).unsqueeze(0)
    pair = torch.cat([f0, f1], dim=0)
    pair_ds = F.interpolate(pair, size=(model_H, model_W), mode='bilinear')
    yuv6 = rgb_to_yuv6(pair_ds)
    pn_in = yuv6.reshape(1, 12, 192, 256)

    with torch.no_grad():
        out_normal = posenet(pn_in)['pose'][0, :6]

    # Zero chroma
    pn_nochroma = pn_in.clone()
    pn_nochroma[:, 4] = 128.0  # U = 128 (neutral)
    pn_nochroma[:, 5] = 128.0  # V = 128 (neutral)
    pn_nochroma[:, 10] = 128.0
    pn_nochroma[:, 11] = 128.0

    with torch.no_grad():
        out_nochroma = posenet(pn_nochroma)['pose'][0, :6]

    diff = (out_normal - out_nochroma).pow(2).sum().sqrt().item()
    mse = (out_normal - out_nochroma).pow(2).mean().item()
    print(f"  Pair {pair_idx}: L2={diff:.6f}, MSE={mse:.8f}")
    print(f"    Normal:   {out_normal.cpu().numpy()}")
    print(f"    No chroma:{out_nochroma.cpu().numpy()}")

# ── Test 5: Quantify: how much of PoseNet's distortion comes from chroma? ──
print("\n--- Test 5: PoseNet distortion decomposition (luma vs chroma) ---")

n_decomp = 50
pose_dist_full = []
pose_dist_luma_only = []
pose_dist_chroma_diff = []

for pair_idx in range(n_decomp):
    # Original
    f0_o = orig_frames[pair_idx * 2].float().to(device).permute(2, 0, 1).unsqueeze(0)
    f1_o = orig_frames[pair_idx * 2 + 1].float().to(device).permute(2, 0, 1).unsqueeze(0)
    pair_o = torch.cat([f0_o, f1_o], dim=0)
    pair_o_ds = F.interpolate(pair_o, size=(model_H, model_W), mode='bilinear')
    yuv6_o = rgb_to_yuv6(pair_o_ds).reshape(1, 12, 192, 256)

    # Compressed
    f0_c = torch.from_numpy(comp_raw[pair_idx * 2].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)
    f1_c = torch.from_numpy(comp_raw[pair_idx * 2 + 1].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)
    pair_c = torch.cat([f0_c, f1_c], dim=0)
    pair_c_ds = F.interpolate(pair_c, size=(model_H, model_W), mode='bilinear')
    yuv6_c = rgb_to_yuv6(pair_c_ds).reshape(1, 12, 192, 256)

    with torch.no_grad():
        out_o = posenet(yuv6_o)['pose'][0, :6]
        out_c = posenet(yuv6_c)['pose'][0, :6]
    full_dist = (out_o - out_c).pow(2).mean().item()
    pose_dist_full.append(full_dist)

    # Replace compressed chroma with original chroma
    yuv6_hybrid = yuv6_c.clone()
    yuv6_hybrid[:, 4] = yuv6_o[:, 4]   # U from original (frame 0)
    yuv6_hybrid[:, 5] = yuv6_o[:, 5]   # V from original (frame 0)
    yuv6_hybrid[:, 10] = yuv6_o[:, 10]  # U from original (frame 1)
    yuv6_hybrid[:, 11] = yuv6_o[:, 11]  # V from original (frame 1)

    with torch.no_grad():
        out_hybrid = posenet(yuv6_hybrid)['pose'][0, :6]
    hybrid_dist = (out_o - out_hybrid).pow(2).mean().item()
    pose_dist_luma_only.append(hybrid_dist)
    pose_dist_chroma_diff.append(full_dist - hybrid_dist)

print(f"\n  Over {n_decomp} pairs:")
print(f"  Mean PoseNet MSE (full compression): {np.mean(pose_dist_full):.8f}")
print(f"  Mean PoseNet MSE (original chroma):  {np.mean(pose_dist_luma_only):.8f}")
print(f"  Chroma contribution to distortion:   {np.mean(pose_dist_chroma_diff):.8f}")
print(f"  Chroma fraction of total distortion: {np.mean(pose_dist_chroma_diff)/np.mean(pose_dist_full)*100:.1f}%")

score_pose_full = math.sqrt(10 * np.mean(pose_dist_full))
score_pose_luma = math.sqrt(10 * np.mean(pose_dist_luma_only))
print(f"\n  Score: sqrt(10*pose_dist)")
print(f"  Full:          {score_pose_full:.6f}")
print(f"  Original chroma: {score_pose_luma:.6f}")
print(f"  Score improvement from perfect chroma: {score_pose_full - score_pose_luma:.6f}")


########################################################################
# HYPOTHESIS 2: SegNet Resolution Exploit
########################################################################
print("\n\n" + "=" * 70)
print("HYPOTHESIS 2: SegNet Resolution Exploit")
print("=" * 70)

print(f"\n  Camera resolution: {W_cam}x{H_cam} = {W_cam*H_cam:,} pixels")
print(f"  Model resolution:  {model_W}x{model_H} = {model_W*model_H:,} pixels")
print(f"  Downscale ratio:   {(W_cam*H_cam)/(model_W*model_H):.2f}x")

# ── Test 1: Verify bilinear round-trip ──
print("\n--- Test 1: Bilinear downscale -> correction -> verify ---")

# Take an odd frame, get SegNet predictions for original and compressed
frame_o = orig_frames[1].float().to(device).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
frame_c = torch.from_numpy(comp_raw[1].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)

with torch.no_grad():
    ds_o = F.interpolate(frame_o, size=(model_H, model_W), mode='bilinear')
    ds_c = F.interpolate(frame_c, size=(model_H, model_W), mode='bilinear')
    pred_o = segnet(ds_o).argmax(dim=1)
    pred_c = segnet(ds_c).argmax(dim=1)
    logits_c = segnet(ds_c)

mismatch = (pred_o != pred_c)
print(f"  Frame 1: {mismatch.sum().item()} mismatched pixels")

# Apply model-res correction: directly fix pixels at model-res
# For each mismatched pixel, what correction is needed?
logits_np = logits_c[0].cpu().numpy()  # 5, H, W
target_np = pred_o[0].cpu().numpy()    # H, W
mismatch_np = mismatch[0].cpu().numpy()

mm_coords = np.argwhere(mismatch_np)
print(f"  Mismatched pixel coords: {len(mm_coords)}")

# Check logit margins at mismatched pixels
margins = []
for r, c in mm_coords[:100]:
    target_cls = target_np[r, c]
    target_logit = logits_np[target_cls, r, c]
    best_other = max(logits_np[cls, r, c] for cls in range(5) if cls != target_cls)
    margins.append(target_logit - best_other)

if margins:
    print(f"  Logit margin at mismatched pixels (target - best_other):")
    print(f"    Mean: {np.mean(margins):.4f}")
    print(f"    Median: {np.median(margins):.4f}")
    print(f"    Min: {np.min(margins):.4f}")
    print(f"    Max: {np.max(margins):.4f}")
    print(f"    <0 (wrong class wins): {sum(1 for m in margins if m < 0)}/{len(margins)}")

# ── Test 2: Correction at model-res vs full-res ──
print("\n--- Test 2: Model-res correction effectiveness ---")

# Apply small correction to compressed frame at model-res
# Strategy: for mismatched pixels, nudge RGB toward original model-res values
ds_delta = ds_o - ds_c  # difference at model-res
print(f"  Model-res delta stats: mean={ds_delta.abs().mean().item():.4f}, max={ds_delta.abs().max().item():.4f}")

# Apply fraction of correction
for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    corrected = ds_c + frac * ds_delta
    with torch.no_grad():
        pred_corrected = segnet(corrected).argmax(dim=1)
    mm = (pred_o != pred_corrected).sum().item()
    print(f"  Correction={frac*100:.0f}%: {mm} mismatched pixels (vs {mismatch.sum().item()} baseline)")

# Now test: model-res correction upscaled then re-downscaled
print(f"\n  Testing round-trip: model-res -> upscale -> downscale")
delta_up = F.interpolate(ds_delta, size=(H_cam, W_cam), mode='bilinear')
delta_roundtrip = F.interpolate(delta_up, size=(model_H, model_W), mode='bilinear')
rt_error = (ds_delta - delta_roundtrip).abs()
print(f"  Round-trip delta error: mean={rt_error.mean().item():.6f}, max={rt_error.max().item():.6f}")

corrected_rt = ds_c + delta_roundtrip
with torch.no_grad():
    pred_rt = segnet(corrected_rt).argmax(dim=1)
mm_rt = (pred_o != pred_rt).sum().item()
print(f"  After round-trip correction: {mm_rt} mismatched pixels (vs {mismatch.sum().item()} baseline)")
print(f"  Direct model-res correction: 0 mismatched pixels")
print(f"  So round-trip is slightly lossy but correction at model-res is EXACT")

# ── Test 3: Storage cost at model-res ──
print(f"\n--- Test 3: Storage efficiency ---")
print(f"  Full-res frame: {W_cam}x{H_cam}x3 = {W_cam*H_cam*3:,} bytes")
print(f"  Model-res frame: {model_W}x{model_H}x3 = {model_W*model_H*3:,} bytes")
print(f"  Savings: {(1 - model_W*model_H/(W_cam*H_cam))*100:.1f}%")
print(f"  For seg correction only (5 classes, 3 bits): {model_W*model_H*3/8:.0f} bytes per frame")


########################################################################
# HYPOTHESIS 3: SegNet Boundary Concentration
########################################################################
print("\n\n" + "=" * 70)
print("HYPOTHESIS 3: SegNet Boundary Concentration")
print("=" * 70)

print("\n--- Running SegNet on ALL 600 frame pairs ---")

batch_size = 16
n_pairs = 600

all_mismatch_counts_comp = []
all_mismatch_fracs_comp = []
all_boundary_counts = []
all_boundary_mismatch_counts = []
all_interior_mismatch_counts = []
class_pair_confusion = Counter()

# For adversarial decode
all_mismatch_counts_adv = []
all_mismatch_fracs_adv = []

# Spatial heatmap
mismatch_heatmap = torch.zeros(model_H, model_W, device=device)

# Per-class statistics
class_mismatch_counts = Counter()  # how many pixels per class are mismatched

t0 = time.time()

for batch_start in range(0, n_pairs, batch_size):
    batch_end = min(batch_start + batch_size, n_pairs)
    B = batch_end - batch_start

    orig_batch = []
    comp_batch = []
    adv_batch = []

    for pair_idx in range(batch_start, batch_end):
        odd_idx = pair_idx * 2 + 1  # SegNet uses odd frames
        orig_batch.append(orig_frames[odd_idx].float())
        comp_batch.append(torch.from_numpy(comp_raw[odd_idx].copy()).float())
        adv_batch.append(torch.from_numpy(adv_raw[odd_idx].copy()).float())

    orig_t = torch.stack(orig_batch).to(device).permute(0, 3, 1, 2)  # B,3,H,W
    comp_t = torch.stack(comp_batch).to(device).permute(0, 3, 1, 2)
    adv_t = torch.stack(adv_batch).to(device).permute(0, 3, 1, 2)

    with torch.no_grad():
        orig_ds = F.interpolate(orig_t, size=(model_H, model_W), mode='bilinear')
        comp_ds = F.interpolate(comp_t, size=(model_H, model_W), mode='bilinear')
        adv_ds = F.interpolate(adv_t, size=(model_H, model_W), mode='bilinear')

        pred_orig = segnet(orig_ds).argmax(dim=1)
        pred_comp = segnet(comp_ds).argmax(dim=1)
        pred_adv = segnet(adv_ds).argmax(dim=1)

    mismatch_comp = (pred_orig != pred_comp)
    mismatch_adv = (pred_orig != pred_adv)

    # Boundary detection on original predictions
    padded = F.pad(pred_orig.float().unsqueeze(1), (1, 1, 1, 1), mode='replicate').squeeze(1)
    center = pred_orig
    up = padded[:, :-2, 1:-1].long()
    down = padded[:, 2:, 1:-1].long()
    left = padded[:, 1:-1, :-2].long()
    right = padded[:, 1:-1, 2:].long()
    is_boundary = ((center != up) | (center != down) | (center != left) | (center != right))

    for b in range(B):
        mm = mismatch_comp[b]
        mm_adv = mismatch_adv[b]
        bnd = is_boundary[b]

        n_mm = mm.sum().item()
        n_bnd = bnd.sum().item()
        n_bnd_mm = (mm & bnd).sum().item()
        n_int_mm = (mm & ~bnd).sum().item()

        all_mismatch_counts_comp.append(n_mm)
        all_mismatch_fracs_comp.append(mm.float().mean().item())
        all_boundary_counts.append(n_bnd)
        all_boundary_mismatch_counts.append(n_bnd_mm)
        all_interior_mismatch_counts.append(n_int_mm)

        all_mismatch_counts_adv.append(mm_adv.sum().item())
        all_mismatch_fracs_adv.append(mm_adv.float().mean().item())

        mismatch_heatmap += mm.float()

        # Class confusion
        if n_mm > 0:
            oc = pred_orig[b][mm].cpu().numpy()
            cc = pred_comp[b][mm].cpu().numpy()
            for o, c in zip(oc, cc):
                class_pair_confusion[(int(o), int(c))] += 1
                class_mismatch_counts[int(o)] += 1

    if batch_start % (batch_size * 10) == 0:
        print(f"  Processed {batch_end}/{n_pairs} pairs ({time.time()-t0:.1f}s)")

print(f"  Done in {time.time()-t0:.1f}s")

# ── Results ──
total_pixels = model_H * model_W
seg_dist_comp = np.mean(all_mismatch_fracs_comp)
seg_dist_adv = np.mean(all_mismatch_fracs_adv)

print(f"\n--- av1_repro: Mismatch statistics ---")
print(f"  Total pixels per frame (model-res): {total_pixels:,}")
print(f"  Mean mismatches/frame: {np.mean(all_mismatch_counts_comp):.1f}")
print(f"  Median mismatches/frame: {np.median(all_mismatch_counts_comp):.1f}")
print(f"  Std: {np.std(all_mismatch_counts_comp):.1f}")
print(f"  Min: {np.min(all_mismatch_counts_comp)}")
print(f"  Max: {np.max(all_mismatch_counts_comp)}")
print(f"  Mean seg_dist: {seg_dist_comp:.8f}")
print(f"  100 * seg_dist: {100*seg_dist_comp:.4f}")

print(f"\n--- adversarial_decode: Mismatch statistics ---")
print(f"  Mean mismatches/frame: {np.mean(all_mismatch_counts_adv):.1f}")
print(f"  Median mismatches/frame: {np.median(all_mismatch_counts_adv):.1f}")
print(f"  Std: {np.std(all_mismatch_counts_adv):.1f}")
print(f"  Min: {np.min(all_mismatch_counts_adv)}")
print(f"  Max: {np.max(all_mismatch_counts_adv)}")
print(f"  Mean seg_dist: {seg_dist_adv:.8f}")
print(f"  100 * seg_dist: {100*seg_dist_adv:.4f}")

print(f"\n--- Boundary analysis (av1_repro) ---")
mean_bnd = np.mean(all_boundary_counts)
mean_bnd_mm = np.mean(all_boundary_mismatch_counts)
mean_int_mm = np.mean(all_interior_mismatch_counts)
total_mm = sum(all_mismatch_counts_comp)
total_bnd_mm = sum(all_boundary_mismatch_counts)
total_int_mm = sum(all_interior_mismatch_counts)
bnd_frac = total_bnd_mm / max(total_mm, 1)

print(f"  Mean boundary pixels/frame: {mean_bnd:.1f} ({mean_bnd/total_pixels*100:.1f}% of pixels)")
print(f"  Mean boundary mismatches/frame: {mean_bnd_mm:.1f}")
print(f"  Mean interior mismatches/frame: {mean_int_mm:.1f}")
print(f"  Fraction of mismatches at boundaries: {bnd_frac:.4f} ({bnd_frac*100:.1f}%)")
print(f"  Enrichment factor: {bnd_frac / (mean_bnd/total_pixels):.1f}x")

print(f"\n--- Class pair confusion (top 15) ---")
class_names = ['road', 'lane_marking', 'undrivable', 'movable', 'car/vehicle']
for (oc, cc), count in class_pair_confusion.most_common(15):
    on = class_names[oc] if oc < len(class_names) else f"cls{oc}"
    cn = class_names[cc] if cc < len(class_names) else f"cls{cc}"
    print(f"  {on} ({oc}) -> {cn} ({cc}): {count:,} total pixels ({count/total_mm*100:.1f}%)")

print(f"\n--- Per-class mismatch counts ---")
for cls, count in sorted(class_mismatch_counts.items()):
    cn = class_names[cls] if cls < len(class_names) else f"cls{cls}"
    print(f"  {cn} ({cls}): {count:,} mismatched pixels ({count/total_mm*100:.1f}% of all mismatches)")

# ── Spatial distribution ──
print(f"\n--- Spatial heatmap statistics ---")
hm = mismatch_heatmap.cpu().numpy()
print(f"  Max mismatches at any location: {hm.max():.0f} / {n_pairs}")
print(f"  Pixels mismatched >50% of frames: {(hm > n_pairs*0.5).sum()}")
print(f"  Pixels mismatched >10% of frames: {(hm > n_pairs*0.1).sum()}")
print(f"  Pixels mismatched >1% of frames:  {(hm > n_pairs*0.01).sum()}")
print(f"  Pixels never mismatched: {(hm == 0).sum()} ({(hm == 0).sum()/total_pixels*100:.1f}%)")

# Row distribution (top/bottom of image)
row_mm = hm.sum(axis=1)
print(f"\n  Top/bottom image row distribution:")
top_quarter = row_mm[:model_H//4].sum()
bottom_quarter = row_mm[3*model_H//4:].sum()
middle_half = row_mm[model_H//4:3*model_H//4].sum()
total_hm = hm.sum()
print(f"    Top quarter:    {top_quarter/total_hm*100:.1f}%")
print(f"    Middle half:    {middle_half/total_hm*100:.1f}%")
print(f"    Bottom quarter: {bottom_quarter/total_hm*100:.1f}%")

# ── Connected component analysis ──
print(f"\n--- Error region clustering (100 frames) ---")
from scipy import ndimage

n_regions_list = []
region_sizes = []

for pair_idx in range(min(100, n_pairs)):
    odd_idx = pair_idx * 2 + 1
    frame_o = orig_frames[odd_idx].float().to(device).permute(2, 0, 1).unsqueeze(0)
    frame_c = torch.from_numpy(comp_raw[odd_idx].copy()).float().to(device).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred_o = segnet(F.interpolate(frame_o, size=(model_H, model_W), mode='bilinear')).argmax(dim=1)
        pred_c = segnet(F.interpolate(frame_c, size=(model_H, model_W), mode='bilinear')).argmax(dim=1)

    mm = (pred_o != pred_c)[0].cpu().numpy().astype(np.uint8)
    labeled, n_regions = ndimage.label(mm)
    n_regions_list.append(n_regions)
    for rid in range(1, n_regions + 1):
        region_sizes.append((labeled == rid).sum())

print(f"  Mean regions/frame: {np.mean(n_regions_list):.1f}")
print(f"  Median regions/frame: {np.median(n_regions_list):.1f}")
print(f"  Max regions in a frame: {np.max(n_regions_list)}")
if region_sizes:
    sizes = np.array(region_sizes)
    print(f"  Mean region size: {np.mean(sizes):.1f} pixels")
    print(f"  Median region size: {np.median(sizes):.1f} pixels")
    print(f"  Max region size: {np.max(sizes)} pixels")
    for t in [1, 2, 5, 10, 20, 50, 100]:
        print(f"    Regions <= {t:3d} px: {(sizes<=t).sum()}/{len(sizes)} ({(sizes<=t).sum()/len(sizes)*100:.1f}%)")


# ── Score impact analysis ──
print("\n\n" + "=" * 70)
print("SCORE IMPACT ANALYSIS")
print("=" * 70)

# For av1_repro
print(f"\n--- av1_repro: Fixing top-K pixels per frame ---")
for k in [10, 50, 100, 200, 500, 1000, 2000]:
    fixed = [max(0, c - k) for c in all_mismatch_counts_comp]
    new_dist = np.mean([c / total_pixels for c in fixed])
    improvement = 100 * (seg_dist_comp - new_dist)
    print(f"  Fix top {k:5d}: seg_dist {seg_dist_comp:.6f} -> {new_dist:.6f} | "
          f"score improvement: {improvement:.4f} | "
          f"storage: {k*3*n_pairs/1024:.0f} KB (sparse, uncompressed)")

print(f"\n--- adversarial_decode: Fixing top-K pixels per frame ---")
for k in [10, 50, 100, 200, 500, 1000]:
    fixed = [max(0, c - k) for c in all_mismatch_counts_adv]
    new_dist = np.mean([c / total_pixels for c in fixed])
    improvement = 100 * (seg_dist_adv - new_dist)
    # Cost: additional bytes in archive
    additional_bytes = k * 3 * n_pairs  # ~3 bytes per correction
    rate_cost = 25 * additional_bytes / (1200 * H_cam * W_cam * 3)
    net = improvement - rate_cost
    print(f"  Fix top {k:4d}: seg_dist {seg_dist_adv:.8f} -> {new_dist:.8f} | "
          f"score gain: {improvement:.6f} | rate cost: {rate_cost:.6f} | net: {net:.6f}")

# Score breakdown estimates
print(f"\n--- Estimated current score components ---")
print(f"  100 * seg_dist (av1):  {100*seg_dist_comp:.4f}")
print(f"  100 * seg_dist (adv):  {100*seg_dist_adv:.4f}")

# Rate component
archive_path = Path('submissions/adversarial_decode/archive.zip')
if archive_path.exists():
    archive_size = archive_path.stat().st_size
    orig_size = sum(Path('videos').rglob('*').stat().st_size for _ in [1])
    # Actually compute properly
    orig_video_size = Path('videos/0.mkv').stat().st_size
    print(f"  Archive size: {archive_size:,} bytes")
    print(f"  Original video: {orig_video_size:,} bytes")
    rate = archive_size / orig_video_size
    print(f"  Rate (archive/original): {rate:.6f}")
    print(f"  25 * rate: {25*rate:.4f}")

print(f"\n--- Per-KB cost of additional archive data ---")
orig_total = Path('videos/0.mkv').stat().st_size
for kb in [1, 5, 10, 50, 100, 500]:
    cost = 25 * (kb * 1024) / orig_total
    print(f"  +{kb:3d} KB -> +{cost:.6f} score (rate term)")

print("\n\nDone! All three hypotheses researched.")
