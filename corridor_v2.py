#!/usr/bin/env python
"""
Scene-adaptive corridor blur + enable-overlays.

Key new ideas:
1. enable-overlays=1 (tested for PR #24 at 2.05, not tested with our pipeline)
2. Wider corridor polygon that preserves road + horizon + road context
3. Scene-segmented temporal masks (analyzing where the road is per segment)
4. Only blur truly useless regions: sky interior, extreme frame margins

Also test combinations with existing sky blur.
"""
import subprocess, sys, os, time, math, zipfile, shutil
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
TMP = ROOT / '_corr2_tmp'


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


def compute_road_masks(gt_frames, segnet):
    """Run SegNet on all frames to identify road/non-road per segment."""
    n = len(gt_frames)
    # Group into temporal segments of ~60 frames (3 seconds at 20fps)
    seg_size = 60
    n_segs = (n + seg_size - 1) // seg_size

    road_masks = []
    with torch.inference_mode():
        for i in range(0, n, 16):
            batch = torch.stack(gt_frames[i:min(i+16, n)]).float().to(DEVICE)
            batch = batch.permute(0, 3, 1, 2)  # B, 3, H, W
            inp = F.interpolate(batch, size=(384, 512), mode='bilinear', align_corners=False)
            out = segnet(inp)
            labels = out.argmax(dim=1).cpu().numpy()  # B, 384, 512
            road_masks.append(labels)

    road_masks = np.concatenate(road_masks, axis=0)  # (N, 384, 512)
    return road_masks


def make_segmented_corridor_mask(road_masks, segment_size=60, blur_sigma=5.0):
    """Create per-frame corridor masks based on SegNet output.

    For each temporal segment, aggregate the road mask to create a stable
    corridor polygon. Then apply heavy blur ONLY outside the corridor.

    The corridor includes: road surface (classes that contain the driving corridor).
    """
    n = len(road_masks)
    model_h, model_w = 384, 512

    # SegNet classes: 0=road, 1=?, 2=?, 3=?, 4=?
    # We need to identify which class is "road". Let's check the class distribution.
    class_counts = np.bincount(road_masks.flatten(), minlength=5)
    print(f"  Class distribution: {class_counts / class_counts.sum()}", flush=True)

    per_frame_masks = []
    for seg_start in range(0, n, segment_size):
        seg_end = min(seg_start + segment_size, n)
        seg_labels = road_masks[seg_start:seg_end]

        # Compute per-class frequency at each pixel
        freq = np.zeros((5, model_h, model_w), dtype=np.float32)
        for c in range(5):
            freq[c] = (seg_labels == c).mean(axis=0)

        # Road classes: typically classes 0 and 2 are road/lane,
        # class 1 is vehicle, class 3 is sky, class 4 is other
        # For a driving corridor, we want to keep everything below the sky
        # and within the road region.

        # Simple approach: the corridor is the convex hull of all non-sky pixels.
        # "Sky" is the class that dominates the top of the frame.
        top_strip = road_masks[seg_start:seg_end, :int(model_h*0.1), :]
        sky_class = np.bincount(top_strip.flatten(), minlength=5).argmax()

        # Create corridor: everywhere that is NOT majority-sky
        # with temporal smoothing
        corridor = (freq[sky_class] < 0.5).astype(np.float32)

        # Dilate the corridor to include context around the road
        kernel = np.ones((15, 15), dtype=np.uint8)
        corridor_dilated = cv2.dilate(corridor, kernel, iterations=2)

        # Smooth the corridor mask
        corridor_smooth = cv2.GaussianBlur(corridor_dilated, (31, 31), 10)
        corridor_smooth = np.clip(corridor_smooth, 0, 1)

        # Upscale to camera resolution
        corridor_cam = cv2.resize(corridor_smooth, (W_CAM, H_CAM),
                                   interpolation=cv2.INTER_LINEAR)

        for _ in range(seg_end - seg_start):
            per_frame_masks.append(corridor_cam)

    return per_frame_masks


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


def apply_mask_blur(frames_np, masks, sigma=5.0):
    ksize = int(sigma * 6) | 1
    result = []
    for frame, mask in zip(frames_np, masks):
        alpha = mask[..., np.newaxis]
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        result.append((frame * alpha + blurred * (1 - alpha)).astype(np.uint8))
    return result


def encode_piped(frames_np, out_mkv, crf=33, film_grain=22, preset=0,
                 keyint=180, extra_params=''):
    w = int(W_CAM * 0.45) // 2 * 2
    h = int(H_CAM * 0.45) // 2 * 2
    params = f'film-grain={film_grain}:keyint={keyint}:scd=0'
    if extra_params:
        params += ':' + extra_params
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', params,
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames_np:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        err = proc.stderr.read().decode()
        print(f"  ENCODE ERR: {err[:200]}", flush=True)
        return None
    return out_mkv.stat().st_size


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


def run_test(name, frames_np, segnet, posenet, preset=0, extra_params=''):
    TMP.mkdir(exist_ok=True)
    out_mkv = TMP / '0.mkv'
    archive_zip = TMP / 'archive.zip'

    t0 = time.time()
    sz = encode_piped(frames_np, out_mkv, preset=preset, extra_params=extra_params)
    if sz is None:
        return None

    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size

    raw = inflate_standard(out_mkv)
    score, s, p, r = fast_eval(raw, archive_size, segnet, posenet)

    print(f"[{name}] score={score:.4f} seg={100*s:.4f} pose={math.sqrt(10*p):.4f} "
          f"rate={25*r:.4f} sz={archive_size/1024:.1f}KB ({time.time()-t0:.0f}s)",
          flush=True)

    out_mkv.unlink(missing_ok=True)
    archive_zip.unlink(missing_ok=True)
    return score


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()
    gt_frames = load_gt_frames()
    print(f"Loaded {len(gt_frames)} GT frames.", flush=True)

    frames_np = [f.numpy() for f in gt_frames]

    # Sky blur baseline frames
    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    ksize = int(5.0 * 6) | 1
    frames_sky = [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - sky_alpha)
                   ).astype(np.uint8) for f in frames_np]

    # 1. BASELINE
    print("\n=== 1. Sky blur baseline ===", flush=True)
    run_test("sky_blur", frames_sky, segnet, posenet)

    # 2. ENABLE-OVERLAYS (key new parameter)
    print("\n=== 2. Enable overlays ===", flush=True)
    run_test("sky_overlay", frames_sky, segnet, posenet,
             extra_params='enable-overlays=1')

    # 3. SCENE-ADAPTIVE CORRIDOR
    print("\n=== 3. Computing road masks ===", flush=True)
    road_masks = compute_road_masks(gt_frames, segnet)

    print("Building corridor masks...", flush=True)
    corridor_masks = make_segmented_corridor_mask(road_masks)

    # Test corridor blur with different sigmas
    for sigma in [3.0, 5.0, 7.0]:
        name = f"corridor_s{sigma}"
        corridor_frames = apply_mask_blur(frames_np, corridor_masks, sigma)
        run_test(name, corridor_frames, segnet, posenet)

    # 4. COMBINED: corridor + sky
    print("\n=== 4. Combined masks ===", flush=True)
    combined_masks = []
    for cm, sm in zip(corridor_masks, [sky_mask]*len(corridor_masks)):
        combined = np.minimum(cm, sm)
        combined_masks.append(combined)

    combined_frames = apply_mask_blur(frames_np, combined_masks, 5.0)
    run_test("corridor+sky_s5", combined_frames, segnet, posenet)

    # 5. CORRIDOR + OVERLAYS
    print("\n=== 5. Best preprocess + overlays ===", flush=True)
    run_test("corridor_s5_overlay", apply_mask_blur(frames_np, corridor_masks, 5.0),
             segnet, posenet, extra_params='enable-overlays=1')
    run_test("sky+overlay", frames_sky, segnet, posenet,
             extra_params='enable-overlays=1')

    # 6. Wider sky blur (top 20%, 25%)
    print("\n=== 6. Wider sky masks ===", flush=True)
    for top_frac in [0.20, 0.25, 0.30]:
        mask = make_sky_mask(top_frac=top_frac)
        alpha = mask[..., np.newaxis]
        wider = [(f * alpha + cv2.GaussianBlur(f, (ksize, ksize), 5.0) * (1 - alpha)
                  ).astype(np.uint8) for f in frames_np]
        run_test(f"sky_{top_frac}", wider, segnet, posenet)

    # 7. Heavy sky blur (stronger sigma)
    print("\n=== 7. Stronger sky blur ===", flush=True)
    for sigma in [8.0, 12.0, 20.0]:
        ks = int(sigma * 6) | 1
        heavy = [(f * sky_alpha + cv2.GaussianBlur(f, (ks, ks), sigma) * (1 - sky_alpha)
                  ).astype(np.uint8) for f in frames_np]
        run_test(f"sky_heavy_s{sigma}", heavy, segnet, posenet)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
