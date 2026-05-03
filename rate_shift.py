#!/usr/bin/env python
"""
Rate-shift strategy: use aggressive video compression (high CRF) to save
space, then fill the budget with seg maps for boundary correction.

The hypothesis: if we compress video harder (CRF 36-42) saving 200-400 KB,
and use that space for high-res seg maps with strong boundary correction,
the net effect might be positive because:
- Higher CRF → seg gets worse, but boundary correction fixes it
- Higher CRF → pose gets worse too (main risk)
- Lower total archive → rate improves

Best case: video CRF 40 (500KB) + seg maps (300KB) = 800KB total
vs current: video CRF 33 (870KB) + no seg = 870KB
Rate is BETTER, and seg is fixed by boundary correction.
But will pose survive CRF 40?
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

TMP_DIR = ROOT / '_rshift_tmp'


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
    n_pairs = len(gt_frames) // 2
    seg_maps = []
    with torch.inference_mode():
        for bs in range(0, n_pairs, 16):
            be = min(bs + 16, n_pairs)
            pairs = []
            for i in range(bs, be):
                pairs.append(torch.stack([gt_frames[i*2].float(), gt_frames[i*2+1].float()]))
            batch = torch.stack(pairs).to(DEVICE)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w')
            seg_in = segnet.preprocess_input(x)
            seg_out = segnet(seg_in)
            seg_maps.append(seg_out.argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.concatenate(seg_maps, axis=0)


def compute_class_colors(gt_frames, seg_maps):
    model_h, model_w = segnet_model_input_size[1], segnet_model_input_size[0]
    class_sum = np.zeros((5, 3), dtype=np.float64)
    class_count = np.zeros(5, dtype=np.float64)
    for i in range(min(len(seg_maps), 100)):
        frame = gt_frames[i*2+1].numpy().astype(np.float64)
        frame_resized = cv2.resize(frame, (model_w, model_h))
        for c in range(5):
            mask = seg_maps[i] == c
            if mask.any():
                class_sum[c] += frame_resized[mask].sum(axis=0)
                class_count[c] += mask.sum()
    return (class_sum / np.maximum(class_count[:, None], 1)).astype(np.float32)


def compress_seg_maps(seg_maps, ch, cw):
    n = seg_maps.shape[0]
    compact = np.zeros((n, ch, cw), dtype=np.uint8)
    for i in range(n):
        compact[i] = cv2.resize(seg_maps[i], (cw, ch), interpolation=cv2.INTER_NEAREST)
    return bz2.compress(compact.tobytes(), compresslevel=9)


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
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
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


def boundary_correction(raw_bytes, seg_compressed, ch, cw,
                         class_colors, strength=0.15):
    n_pairs = len(bz2.decompress(seg_compressed)) // (ch * cw)
    seg_compact = np.frombuffer(bz2.decompress(seg_compressed),
                                 dtype=np.uint8).reshape(n_pairs, ch, cw)
    raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(
        n_pairs * 2, H_CAM, W_CAM, 3).copy()

    for i in range(n_pairs):
        seg = seg_compact[i]
        seg_full = cv2.resize(seg, (W_CAM, H_CAM), interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3, 3), dtype=np.uint8)
        boundary = cv2.dilate(seg_full, kernel) != cv2.erode(seg_full, kernel)
        if not boundary.any():
            continue
        frame = raw[i*2+1].astype(np.float32)
        for c in range(5):
            mask = boundary & (seg_full == c)
            if mask.any():
                frame[mask] = frame[mask] * (1 - strength) + class_colors[c] * strength
        raw[i*2+1] = frame.clip(0, 255).astype(np.uint8)
    return raw.tobytes()


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

    seg_maps = extract_seg_maps(gt_frames, segnet)
    class_colors = compute_class_colors(gt_frames, seg_maps)
    print("Seg maps and class colors computed.", flush=True)

    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    frames_np = [f.numpy() for f in gt_frames]
    frames_sky = [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - sky_alpha)
                   ).astype(np.uint8) for f in frames_np]

    TMP_DIR.mkdir(exist_ok=True)

    # Test different CRF values with and without boundary correction
    target_budget = 870  # KB - match current archive size

    for crf in [33, 35, 37, 39, 41]:
        out_mkv = TMP_DIR / '0.mkv'
        video_size = encode_piped(frames_sky, out_mkv, crf=crf, preset=0)
        if video_size is None:
            continue
        video_kb = video_size / 1024

        print(f"\n=== CRF {crf}: video={video_kb:.0f}KB ===", flush=True)

        # Video-only baseline for this CRF
        raw_base = inflate_standard(out_mkv)
        archive_zip = TMP_DIR / 'archive.zip'
        with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(out_mkv, '0.mkv')
        base_size = archive_zip.stat().st_size
        score, s, p, r = fast_eval(raw_base, base_size, segnet, posenet)
        print(f"  [video_only] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={base_size/1024:.0f}KB", flush=True)

        # Space available for seg maps
        remaining_kb = target_budget - video_kb
        if remaining_kb < 20:
            print(f"  No room for seg maps ({remaining_kb:.0f}KB remaining)")
            continue

        # Try different seg resolutions that fit the budget
        for ch, cw in [(192, 256), (96, 128), (48, 64)]:
            seg_compressed = compress_seg_maps(seg_maps, ch, cw)
            seg_kb = len(seg_compressed) / 1024
            total_kb = video_kb + seg_kb + 1  # +1 for zip overhead

            if total_kb > target_budget + 10:
                continue

            for strength in [0.10, 0.15, 0.20]:
                corrected = boundary_correction(raw_base, seg_compressed, ch, cw,
                                                class_colors, strength)

                # Build combined archive
                seg_bin = TMP_DIR / 'seg.bin'
                seg_bin.write_bytes(seg_compressed)
                combo_zip = TMP_DIR / 'combo.zip'
                with zipfile.ZipFile(combo_zip, 'w', zipfile.ZIP_STORED) as zf:
                    zf.write(out_mkv, '0.mkv')
                    zf.write(seg_bin, 'seg.bin')
                combo_size = combo_zip.stat().st_size

                score, s, p, r = fast_eval(corrected, combo_size, segnet, posenet)
                name = f"crf{crf}_bc{ch}x{cw}_s{strength}"
                print(f"  [{name}] score={score:.4f} seg={100*s:.4f} "
                      f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
                      f"sz={combo_size/1024:.0f}KB", flush=True)

        out_mkv.unlink(missing_ok=True)

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\nDone.", flush=True)
