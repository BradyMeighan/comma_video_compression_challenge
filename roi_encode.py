#!/usr/bin/env python
"""
ROI-aware encoding pipeline using SegNet/PoseNet gradient sensitivity maps.

Computes per-pixel model sensitivity via backward passes, converts to 64x64
block-level QP offsets for SVT-AV1's ROI map feature, and encodes with
spatially-adaptive quality.

Usage: python roi_encode.py [--sweep]
"""
import subprocess, sys, os, time, math, zipfile, shutil, struct
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


def rgb_to_yuv6_diff(rgb_chw):
    """Differentiable version of frame_utils.rgb_to_yuv6 (no @torch.no_grad)."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def posenet_preprocess_diff(x, model_size=(384, 512)):
    """Differentiable PoseNet preprocessing (bypasses @torch.no_grad in rgb_to_yuv6)."""
    batch_size, seq_len = x.shape[0], x.shape[1]
    flat = einops.rearrange(x, 'b t c h w -> (b t) c h w')
    flat = F.interpolate(flat, size=(model_size[0], model_size[1]), mode='bilinear', align_corners=False)
    yuv = rgb_to_yuv6_diff(flat)
    return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len)
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)
    for p in posenet.parameters():
        p.requires_grad_(False)
    return segnet, posenet


def load_gt_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def compute_sensitivity_maps(gt_frames, segnet, posenet):
    """Compute per-pixel gradient sensitivity for SegNet and PoseNet.

    For each frame pair, we compute how sensitive the model output is to
    each input pixel. This tells us which regions need the most bits.
    """
    n_pairs = len(gt_frames) // 2
    model_w, model_h = segnet_model_input_size  # 512, 384

    seg_sensitivity_all = []
    pose_sensitivity_all = []

    batch_size = 8
    for bs in range(0, n_pairs, batch_size):
        be = min(bs + batch_size, n_pairs)
        B = be - bs

        pairs = []
        for i in range(bs, be):
            pair = torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()])
            pairs.append(pair)

        batch_data = torch.stack(pairs).to(DEVICE)  # B, 2, H, W, 3

        # SegNet sensitivity: gradient of logit margin at all pixels.
        # Measures how much each input pixel affects confidence of top class.
        seg_input = batch_data.clone().requires_grad_(True)
        x_seg = einops.rearrange(seg_input, 'b t h w c -> b t c h w')
        seg_in = F.interpolate(x_seg[:, -1], size=(model_h, model_w),
                               mode='bilinear', align_corners=False)
        seg_out = segnet(seg_in)  # B, 5, H, W
        top_vals, top_idx = seg_out.topk(2, dim=1)
        margin = (top_vals[:, 0] - top_vals[:, 1]).mean()
        margin.backward()
        seg_sens = seg_input.grad.abs().mean(dim=(1, 4))
        seg_sensitivity_all.append(seg_sens.detach().cpu())

        # PoseNet sensitivity (differentiable preprocessing)
        pose_input = batch_data.clone().requires_grad_(True)
        x_pose = einops.rearrange(pose_input, 'b t h w c -> b t c h w')
        pn_in = posenet_preprocess_diff(x_pose)
        pn_out = posenet(pn_in)['pose'][:, :6]
        pose_loss = pn_out.pow(2).sum()
        pose_loss.backward()
        pose_sens = pose_input.grad.abs().mean(dim=(1, 4))
        pose_sensitivity_all.append(pose_sens.detach().cpu())

        del seg_input, pose_input, batch_data

        if (bs // batch_size) % 10 == 0:
            print(f"  Sensitivity batch {bs//batch_size + 1}/{(n_pairs+batch_size-1)//batch_size}",
                  flush=True)

    seg_maps = torch.cat(seg_sensitivity_all, dim=0)   # N_pairs, H, W
    pose_maps = torch.cat(pose_sensitivity_all, dim=0)  # N_pairs, H, W

    return seg_maps, pose_maps


def sensitivity_to_roi_map(seg_maps, pose_maps, enc_w, enc_h,
                            seg_weight=100.0, pose_weight=3.16,
                            qp_range=(-32, 16)):
    """Convert per-pixel sensitivity to SVT-AV1 ROI QP offset map.

    Each line: frame_number qp_offset_block_0 qp_offset_block_1 ...
    Blocks are 64x64 in the encoded resolution, row-by-row order.
    """
    n_pairs = seg_maps.shape[0]
    n_frames = n_pairs * 2

    blocks_w = math.ceil(enc_w / 64)
    blocks_h = math.ceil(enc_h / 64)

    combined = seg_weight * seg_maps + pose_weight * pose_maps
    combined = combined.numpy()

    # Normalize to [0, 1] per-frame
    for i in range(n_pairs):
        mn = combined[i].min()
        mx = combined[i].max()
        if mx > mn:
            combined[i] = (combined[i] - mn) / (mx - mn)
        else:
            combined[i] = 0.5

    # Downscale to encoded resolution and then to block grid
    roi_lines = []
    for frame_idx in range(n_frames):
        pair_idx = frame_idx // 2
        sens = combined[pair_idx]  # H_cam, W_cam

        # Downscale to encoded resolution
        sens_t = torch.from_numpy(sens).unsqueeze(0).unsqueeze(0).float()
        sens_down = F.interpolate(sens_t, size=(enc_h, enc_w),
                                   mode='bilinear', align_corners=False)
        sens_down = sens_down.squeeze().numpy()

        # Average into 64x64 blocks
        block_values = []
        for by in range(blocks_h):
            for bx in range(blocks_w):
                y0 = by * 64
                y1 = min(y0 + 64, enc_h)
                x0 = bx * 64
                x1 = min(x0 + 64, enc_w)
                block_val = sens_down[y0:y1, x0:x1].mean()
                block_values.append(float(block_val))

        # Convert importance to QP offset:
        # High importance (1.0) → large negative offset (more bits)
        # Low importance (0.0) → large positive offset (fewer bits)
        qp_min, qp_max = qp_range
        offsets = []
        for v in block_values:
            qp = int(round(qp_max - v * (qp_max - qp_min)))
            qp = max(qp_min, min(qp_max, qp))
            offsets.append(qp)

        line = f"{frame_idx} " + " ".join(str(o) for o in offsets)
        roi_lines.append(line)

    return "\n".join(roi_lines) + "\n"


def make_sky_mask(top_frac=0.15, side_frac=0.03):
    mask = np.ones((H_CAM, W_CAM), dtype=np.float32)
    sky_end = int(H_CAM * top_frac)
    for y in range(sky_end):
        mask[y, :] = (y / sky_end) ** 0.5
    side_px = int(W_CAM * side_frac)
    for x in range(side_px):
        t = (x / side_px) ** 0.5
        mask[:, x] = np.minimum(mask[:, x], t)
        mask[:, W_CAM - 1 - x] = np.minimum(mask[:, W_CAM - 1 - x], t)
    return cv2.GaussianBlur(mask, (31, 31), 10)


def encode_with_roi(frames_np, out_mkv, roi_file, crf=33, scale=0.45,
                    film_grain=22, preset=0, keyint=180):
    """Encode frames with ROI map via ffmpeg libsvtav1."""
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2

    svtav1_params = f'film-grain={film_grain}:keyint={keyint}:scd=0'
    if roi_file:
        svtav1_params += f':roi-map-file={roi_file}'

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', svtav1_params,
        '-r', '20',
        str(out_mkv),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames_np:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        print(f"  ENCODE FAILED: {stderr[:500]}", flush=True)
        return None, stderr

    return out_mkv.stat().st_size, None


def decode_and_inflate(mkv_path, strength=0.45):
    """Standard decode: Lanczos upscale + binomial unsharp."""
    import av
    from PIL import Image

    _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    kernel = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

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
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), kernel, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())
    container.close()
    return raw_bytes


def fast_eval(raw_bytes, archive_size, segnet, posenet):
    """Evaluate raw bytes against ground truth."""
    gt_cache = ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt'
    if gt_cache.exists():
        gt = torch.load(gt_cache, weights_only=True)
    else:
        raise RuntimeError("Run fast_eval once first to cache GT")

    N = gt['seg'].shape[0]
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)

    seg_dists, pose_dists = [], []
    bs = 16
    with torch.inference_mode():
        for i in range(0, N, bs):
            end = min(i + bs, N)
            f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(DEVICE).float()
            x = torch.stack([f0, f1], dim=1)
            x = einops.rearrange(x, 'b t h w c -> b t c h w')

            seg_in = segnet.preprocess_input(x)
            seg_pred = segnet(seg_in).argmax(1)
            gt_seg = gt['seg'][i:end].to(DEVICE)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())

            pn_in = posenet.preprocess_input(x)
            pn_out = posenet(pn_in)['pose'][:, :6]
            gt_pose = gt['pose'][i:end].to(DEVICE)
            pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())

    seg_d = np.mean(seg_dists)
    pose_d = np.mean(pose_dists)
    rate = archive_size / 37_545_489
    score = 100 * seg_d + math.sqrt(10 * pose_d) + 25 * rate
    return score, seg_d, pose_d, rate


def run_config(name, frames_np, segnet, posenet, roi_text=None,
               crf=33, scale=0.45, film_grain=22, preset=0, keyint=180,
               unsharp=0.45):
    """Run a complete encode→decode→eval cycle."""
    tmp_dir = ROOT / '_roi_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'
    roi_file = None

    if roi_text:
        roi_path = tmp_dir / 'roi_map.txt'
        roi_path.write_text(roi_text)
        roi_file = str(roi_path)

    t0 = time.time()

    mkv_size, err = encode_with_roi(frames_np, out_mkv, roi_file,
                                     crf=crf, scale=scale, film_grain=film_grain,
                                     preset=preset, keyint=keyint)
    if mkv_size is None:
        return None, err

    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size

    raw_bytes = decode_and_inflate(out_mkv, strength=unsharp)
    score, seg_d, pose_d, rate = fast_eval(raw_bytes, archive_size, segnet, posenet)

    elapsed = time.time() - t0
    print(f"[{name}] score={score:.4f} "
          f"seg={100*seg_d:.4f} pose={math.sqrt(10*pose_d):.4f} "
          f"rate={25*rate:.4f} size={archive_size/1024:.1f}KB "
          f"({elapsed:.0f}s)", flush=True)

    out_mkv.unlink(missing_ok=True)
    archive_zip.unlink(missing_ok=True)

    return score, None


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    print("Models loaded.", flush=True)

    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)

    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]

    def sky_blur(frames):
        ksize = int(5.0 * 6) | 1
        result = []
        for f in frames:
            blurred = cv2.GaussianBlur(f, (ksize, ksize), 5.0)
            result.append((f * sky_alpha + blurred * (1 - sky_alpha)).astype(np.uint8))
        return result

    frames_np = sky_blur([f.numpy() for f in gt_frames])
    print("Frames preprocessed with sky blur.", flush=True)

    # Step 1: Baseline without ROI
    print("\n=== Baseline (no ROI, preset=0) ===", flush=True)
    score_base, _ = run_config("baseline", frames_np, segnet, posenet, preset=0)

    # Step 2: Compute sensitivity maps
    print("\n=== Computing sensitivity maps ===", flush=True)
    t0 = time.time()
    seg_sens, pose_sens = compute_sensitivity_maps(gt_frames, segnet, posenet)
    print(f"  Sensitivity computed in {time.time()-t0:.0f}s", flush=True)
    print(f"  Seg sensitivity range: [{seg_sens.min():.6f}, {seg_sens.max():.6f}]")
    print(f"  Pose sensitivity range: [{pose_sens.min():.6f}, {pose_sens.max():.6f}]")

    # Step 3: Test ROI maps with different QP offset ranges
    enc_w = int(W_CAM * 0.45) // 2 * 2  # 522
    enc_h = int(H_CAM * 0.45) // 2 * 2  # 392

    configs = [
        ("roi_32_16", (-32, 16)),
        ("roi_24_12", (-24, 12)),
        ("roi_16_8",  (-16, 8)),
        ("roi_8_4",   (-8, 4)),
        ("roi_48_16", (-48, 16)),
        ("roi_32_0",  (-32, 0)),
        ("roi_16_0",  (-16, 0)),
    ]

    for name, qp_range in configs:
        roi_text = sensitivity_to_roi_map(seg_sens, pose_sens, enc_w, enc_h,
                                           qp_range=qp_range)
        score, err = run_config(name, frames_np, segnet, posenet,
                                roi_text=roi_text, preset=0)
        if err:
            print(f"  [{name}] ERROR (ROI might not be supported via ffmpeg): {err[:200]}")
            # If ROI via ffmpeg fails, try without ROI but test the concept
            # with per-frame denoising guided by sensitivity maps instead
            break

    # Step 4: If ROI maps work, try combined optimizations
    # If they don't, fall back to sensitivity-guided preprocessing

    # Cleanup
    tmp_dir = ROOT / '_roi_tmp'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print("\nDone.", flush=True)
