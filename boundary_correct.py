#!/usr/bin/env python
"""
Boundary color correction: store compact seg maps in archive, use at decode
to push boundary pixels toward correct class colors.

Strategy:
1. Encode: Extract SegNet argmax at model resolution (384x512), downsample to
   compact resolution, compress with bz2, store alongside video in archive.
2. Decode: After standard Lanczos upscale + unsharp, load seg maps,
   upsample to camera resolution, find boundary pixels, and adjust their
   colors to reduce seg argmax flips.

The correction is pure signal processing — no models needed at decode time.
"""
import subprocess, sys, os, time, math, zipfile, bz2, struct, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file
import einops

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_gt_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def extract_seg_maps(gt_frames, segnet):
    """Extract SegNet argmax maps at model resolution (384x512) for each pair."""
    n_pairs = len(gt_frames) // 2
    seg_maps = []
    batch_size = 16

    with torch.inference_mode():
        for bs in range(0, n_pairs, batch_size):
            be = min(bs + batch_size, n_pairs)
            pairs = []
            for i in range(bs, be):
                pairs.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
            batch = torch.stack(pairs).to(DEVICE)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w')
            seg_in = segnet.preprocess_input(x)
            seg_out = segnet(seg_in)
            seg_labels = seg_out.argmax(dim=1).cpu().numpy().astype(np.uint8)
            seg_maps.append(seg_labels)

    return np.concatenate(seg_maps, axis=0)  # (600, 384, 512)


def compress_seg_maps(seg_maps, compact_h=96, compact_w=128):
    """Compress seg maps to compact resolution and bz2 compress.

    Returns compressed bytes and the compact resolution used.
    """
    n = seg_maps.shape[0]

    # Downsample using nearest-neighbor (preserves class labels)
    compact = np.zeros((n, compact_h, compact_w), dtype=np.uint8)
    for i in range(n):
        compact[i] = cv2.resize(seg_maps[i], (compact_w, compact_h),
                                 interpolation=cv2.INTER_NEAREST)

    raw = compact.tobytes()
    compressed = bz2.compress(raw, compresslevel=9)
    return compressed, compact_h, compact_w


def compute_class_colors(gt_frames, seg_maps):
    """Compute average RGB color per seg class from the ground truth frames."""
    n_pairs = len(gt_frames) // 2
    model_h, model_w = segnet_model_input_size[1], segnet_model_input_size[0]

    class_sum = np.zeros((5, 3), dtype=np.float64)
    class_count = np.zeros(5, dtype=np.float64)

    for i in range(min(n_pairs, 100)):
        frame = gt_frames[i*2+1].numpy().astype(np.float64)
        frame_resized = cv2.resize(frame, (model_w, model_h))
        for c in range(5):
            mask = seg_maps[i] == c
            if mask.any():
                class_sum[c] += frame_resized[mask].sum(axis=0)
                class_count[c] += mask.sum()

    class_colors = np.zeros((5, 3), dtype=np.float32)
    for c in range(5):
        if class_count[c] > 0:
            class_colors[c] = class_sum[c] / class_count[c]

    return class_colors


def boundary_correction(raw_bytes, seg_maps_compressed, compact_h, compact_w,
                         class_colors, correction_strength=0.1):
    """Apply boundary color correction to inflated frames.

    For each pair, upsample the compact seg map to camera resolution,
    detect boundary pixels, and nudge their colors toward the correct class.
    """
    n_pairs = len(seg_maps_compressed) // (compact_h * compact_w)
    seg_compact = np.frombuffer(seg_maps_compressed, dtype=np.uint8).reshape(
        n_pairs, compact_h, compact_w)

    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        n_pairs * 2, H_CAM, W_CAM, 3)
    result = raw.copy()

    for i in range(n_pairs):
        seg = seg_compact[i]

        # Upsample to camera resolution
        seg_full = cv2.resize(seg, (W_CAM, H_CAM), interpolation=cv2.INTER_NEAREST)

        # Detect boundary pixels (where adjacent pixels have different classes)
        kernel = np.ones((3, 3), dtype=np.uint8)
        dilated = cv2.dilate(seg_full, kernel)
        eroded = cv2.erode(seg_full, kernel)
        boundary = (dilated != eroded)  # H, W boolean

        if not boundary.any():
            continue

        # For the last frame of the pair (what SegNet evaluates)
        frame_idx = i * 2 + 1
        frame = result[frame_idx].astype(np.float32)

        # At boundary pixels, nudge color toward the correct class color
        for c in range(5):
            mask = boundary & (seg_full == c)
            if mask.any():
                target_color = class_colors[c]
                frame[mask] = frame[mask] * (1 - correction_strength) + \
                              target_color * correction_strength

        result[frame_idx] = frame.clip(0, 255).astype(np.uint8)

    return result.tobytes()


def make_sky_mask():
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * 0.15)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * 0.03)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t)
    return cv2.GaussianBlur(mask, (31, 31), 10)


def encode_piped(frames_np, out_mkv, crf=33, film_grain=22, preset=0, keyint=180):
    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', f'film-grain={film_grain}:keyint={keyint}:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames_np:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size if proc.returncode == 0 else None


def inflate_standard(mkv_path, strength=0.45):
    import av
    from PIL import Image
    container = av.open(str(mkv_path))
    raw_bytes = bytearray()
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL_9, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())
    container.close()
    return bytes(raw_bytes)


def fast_eval(raw_bytes, archive_size, segnet, posenet):
    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)
    seg_dists, pose_dists = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, 'b t h w c -> b t c h w')
            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            gt_seg = gt['seg'][i:end].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())
            pn_out = posenet(posenet.preprocess_input(x))['pose'][:, :6]
            gt_pose = gt['pose'][i:end].to(DEVICE)
            pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())
    s, p = np.mean(seg_dists), np.mean(pose_dists)
    r = archive_size / 37_545_489
    return 100*s + math.sqrt(10*p) + 25*r, s, p, r


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)

    # Step 1: Extract seg maps
    print("Extracting seg maps...", flush=True)
    t0 = time.time()
    seg_maps = extract_seg_maps(gt_frames, segnet)
    print(f"  Shape: {seg_maps.shape}, {time.time()-t0:.0f}s", flush=True)

    # Step 2: Compute class colors
    print("Computing class colors...", flush=True)
    class_colors = compute_class_colors(gt_frames, seg_maps)
    for c in range(5):
        print(f"  Class {c}: RGB = ({class_colors[c][0]:.0f}, "
              f"{class_colors[c][1]:.0f}, {class_colors[c][2]:.0f})")

    # Step 3: Test different compact resolutions and their compressed sizes
    print("\n=== Seg map compression tests ===", flush=True)
    for res in [(48, 64), (96, 128), (192, 256), (384, 512)]:
        ch, cw = res
        compressed, _, _ = compress_seg_maps(seg_maps, ch, cw)
        print(f"  {ch}x{cw}: {len(compressed)/1024:.1f} KB "
              f"(raw: {seg_maps.shape[0]*ch*cw/1024:.1f} KB)")

    # Step 4: Prepare frames with sky blur
    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    frames_np = [f.numpy() for f in gt_frames]
    frames_sky = [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - sky_alpha)
                   ).astype(np.uint8) for f in frames_np]

    # Step 5: Encode baseline
    print("\n=== Encoding ===", flush=True)
    tmp_dir = ROOT / '_bound_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'

    encode_piped(frames_sky, out_mkv)
    video_size = out_mkv.stat().st_size
    print(f"  Video: {video_size/1024:.1f} KB")

    # Step 6: Inflate
    print("Inflating...", flush=True)
    raw_baseline = inflate_standard(out_mkv)

    # Step 7: Evaluate baseline (video only)
    print("\n=== Baseline (video only) ===", flush=True)
    archive_zip = tmp_dir / 'archive.zip'
    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    base_archive = archive_zip.stat().st_size
    score_base, s_b, p_b, r_b = fast_eval(raw_baseline, base_archive, segnet, posenet)
    print(f"[baseline] score={score_base:.4f} seg={100*s_b:.4f} "
          f"pose={math.sqrt(10*p_b):.4f} rate={25*r_b:.4f} "
          f"size={base_archive/1024:.1f}KB")

    # Step 8: Test boundary correction with different configs
    print("\n=== Boundary correction tests ===", flush=True)
    compact_configs = [
        (48, 64),
        (96, 128),
        (192, 256),
    ]
    strength_configs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    for ch, cw in compact_configs:
        seg_compressed, _, _ = compress_seg_maps(seg_maps, ch, cw)
        seg_size = len(seg_compressed)

        # Build archive with video + seg data
        archive_with_seg = tmp_dir / 'archive_seg.zip'
        seg_bin = tmp_dir / 'seg.bin'
        seg_bin.write_bytes(seg_compressed)

        # Also store compact resolution as header
        meta = struct.pack('HH', ch, cw)
        meta_path = tmp_dir / 'seg_meta.bin'
        meta_path.write_bytes(meta)

        # Store class colors
        colors_path = tmp_dir / 'class_colors.bin'
        colors_path.write_bytes(class_colors.tobytes())

        with zipfile.ZipFile(archive_with_seg, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(out_mkv, '0.mkv')
            zf.write(seg_bin, 'seg.bin')
            zf.write(meta_path, 'seg_meta.bin')
            zf.write(colors_path, 'class_colors.bin')

        archive_size = archive_with_seg.stat().st_size
        rate_delta = 25 * (archive_size - base_archive) / 37_545_489

        print(f"\n  Compact res: {ch}x{cw}, seg={seg_size/1024:.1f}KB, "
              f"total={archive_size/1024:.1f}KB, rate_delta={rate_delta:+.4f}")

        # Decompress seg maps for correction
        seg_raw = bz2.decompress(seg_compressed)

        for strength in strength_configs:
            name = f"bc_{ch}x{cw}_s{strength}"
            corrected = boundary_correction(raw_baseline, seg_raw, ch, cw,
                                            class_colors, strength)
            score, s, p, r = fast_eval(corrected, archive_size, segnet, posenet)
            delta = score - score_base
            print(f"    [{name}] score={score:.4f} seg={100*s:.4f} "
                  f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
                  f"(delta={delta:+.4f})", flush=True)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("\nDone.", flush=True)
