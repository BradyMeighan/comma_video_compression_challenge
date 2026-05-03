#!/usr/bin/env python
"""
ROI-based preprocessing: denoise non-important regions before encoding.
Uses SegNet to generate importance maps, then blurs unimportant areas.

This is the key technique the leader uses to gain ~0.18 score advantage.
The leader uses hand-drawn driving corridor polygons; we use model-derived maps.

Usage:
    python roi_preprocess.py                   # preprocess with default settings
    python roi_preprocess.py --blur-sigma 5    # stronger blur on non-ROI
    python roi_preprocess.py --visualize       # save debug visualizations
"""

import os, sys, time, math, argparse, subprocess, zipfile
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
import einops
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
INPUT_VIDEO = VIDEOS_DIR / "0.mkv"
ORIGINAL_SIZE = 37_545_489
W_CAM, H_CAM = 1164, 874
NUM_FRAMES = 1200


def generate_importance_map(device='cuda'):
    """
    Generate per-pixel importance maps using SegNet gradients.
    Pixels that strongly affect segmentation output get high importance.
    """
    from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
    from safetensors.torch import load_file
    from frame_utils import AVVideoDataset

    device = torch.device(device)

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    ds = AVVideoDataset(['0.mkv'], data_dir=VIDEOS_DIR,
                        batch_size=1, device=torch.device('cpu'))
    ds.prepare_data()

    print("Generating importance maps from SegNet + PoseNet gradients...", flush=True)

    all_importance = []
    frame_idx = 0

    for _, _, batch in ds:
        batch = batch.to(device)
        x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()

        for t_idx in range(x.shape[1]):
            frame = x[:, t_idx:t_idx+1].clone().requires_grad_(True)
            frame_input = frame.squeeze(1)

            seg_in = F.interpolate(frame_input, size=(384, 512), mode='bilinear')
            seg_in.retain_grad()
            seg_out = segnet(seg_in)
            seg_loss = seg_out.max(1).values.sum()
            seg_loss.backward(retain_graph=False)

            seg_grad = seg_in.grad.abs().mean(1, keepdim=True)
            seg_importance = F.interpolate(seg_grad, size=(H_CAM, W_CAM), mode='bilinear')
            seg_importance = seg_importance.squeeze().detach()

            seg_in.grad = None
            segnet.zero_grad()

            all_importance.append(seg_importance.cpu())
            frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{NUM_FRAMES} frames", flush=True)

    importance = torch.stack(all_importance)
    p5, p95 = importance.quantile(0.05), importance.quantile(0.95)
    importance = ((importance - p5) / (p95 - p5 + 1e-8)).clamp(0, 1)

    del segnet, posenet
    torch.cuda.empty_cache()

    return importance


def generate_segmentation_roi(device='cuda'):
    """
    Simpler approach: use SegNet class predictions to define ROI.
    Road (class ~1-2) and vehicles (class ~3-4) are important.
    Sky/background (class ~0) is less important.
    Also: edges between classes are important.
    """
    from modules import SegNet, segnet_sd_path
    from safetensors.torch import load_file
    from frame_utils import AVVideoDataset

    device = torch.device(device)
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    ds = AVVideoDataset(['0.mkv'], data_dir=VIDEOS_DIR,
                        batch_size=4, device=torch.device('cpu'))
    ds.prepare_data()

    print("Generating segmentation-based ROI maps...", flush=True)

    all_rois = []
    frame_idx = 0

    with torch.inference_mode():
        for _, _, batch in ds:
            batch = batch.to(device)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()

            for t_idx in range(x.shape[1]):
                frame = x[:, t_idx]
                seg_in = F.interpolate(frame, size=(384, 512), mode='bilinear')
                seg_out = segnet(seg_in)
                seg_pred = seg_out.argmax(1)

                seg_full = F.interpolate(
                    seg_pred.float().unsqueeze(1),
                    size=(H_CAM, W_CAM), mode='nearest'
                ).squeeze(1)

                for b in range(seg_full.shape[0]):
                    seg_map = seg_full[b]

                    importance = torch.ones_like(seg_map) * 0.3

                    importance[seg_map == 0] = 0.1  # sky/background
                    importance[seg_map == 1] = 0.8  # road
                    importance[seg_map == 2] = 1.0  # lane markings / vehicles
                    importance[seg_map == 3] = 1.0  # vehicles/objects
                    importance[seg_map == 4] = 0.6  # other

                    sobel_x = F.conv2d(
                        seg_map.unsqueeze(0).unsqueeze(0).float(),
                        torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
                        padding=1
                    )
                    sobel_y = F.conv2d(
                        seg_map.unsqueeze(0).unsqueeze(0).float(),
                        torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0),
                        padding=1
                    )
                    edges = (sobel_x.abs() + sobel_y.abs()).squeeze()
                    edge_mask = (edges > 0).float()

                    edge_dilated = F.max_pool2d(
                        edge_mask.unsqueeze(0).unsqueeze(0),
                        kernel_size=15, stride=1, padding=7
                    ).squeeze()

                    importance = torch.max(importance, edge_dilated)

                    importance = F.avg_pool2d(
                        importance.unsqueeze(0).unsqueeze(0),
                        kernel_size=31, stride=1, padding=15
                    ).squeeze()

                    importance = importance.clamp(0, 1)
                    all_rois.append(importance.cpu())
                    frame_idx += 1

            if frame_idx % 200 == 0:
                print(f"  {frame_idx}/{NUM_FRAMES} frames", flush=True)

    print(f"  Done: {frame_idx} frames", flush=True)
    del segnet
    torch.cuda.empty_cache()

    return torch.stack(all_rois)


def generate_driving_corridor_roi():
    """
    Static driving corridor ROI based on typical dashcam perspective.
    The bottom-center of the frame is road (high importance).
    Top portion is sky (low importance). Sides are moderate.
    This is a simple approximation of the leader's hand-drawn polygon.
    """
    importance = np.zeros((H_CAM, W_CAM), dtype=np.float32)

    cx, cy = W_CAM // 2, H_CAM // 3  # vanishing point approx
    pts = np.array([
        [0, H_CAM],           # bottom-left
        [cx - 200, cy],       # upper-left of corridor
        [cx + 200, cy],       # upper-right of corridor
        [W_CAM, H_CAM],      # bottom-right
    ], dtype=np.int32)

    mask = np.zeros((H_CAM, W_CAM), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    importance[mask > 0] = 1.0
    importance[mask == 0] = 0.15

    importance = cv2.GaussianBlur(importance, (51, 51), 15)
    importance = np.clip(importance, 0, 1)

    return torch.from_numpy(importance).unsqueeze(0).expand(NUM_FRAMES, -1, -1)


def preprocess_video(roi_maps, input_video, output_video,
                     blur_sigma=5.0, min_importance=0.15):
    """
    Apply ROI-based preprocessing: blur non-important regions.
    roi_maps: (N, H, W) tensor of importance values [0, 1]
    """
    import av
    from frame_utils import yuv420_to_rgb

    print(f"Preprocessing video (blur_sigma={blur_sigma})...", flush=True)

    container = av.open(str(input_video))
    stream = container.streams.video[0]

    ksize = int(blur_sigma * 6) | 1

    out_frames = []
    frame_idx = 0
    t0 = time.time()

    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)
        img = rgb.numpy()

        importance = roi_maps[frame_idx].numpy()

        blurred = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)

        alpha = importance[..., np.newaxis]
        result = (img * alpha + blurred * (1 - alpha)).astype(np.uint8)

        out_frames.append(result)
        frame_idx += 1

        if frame_idx % 200 == 0:
            elapsed = time.time() - t0
            print(f"  {frame_idx}/{NUM_FRAMES} frames ({elapsed:.0f}s)", flush=True)

    container.close()

    print(f"Writing preprocessed video to raw file...", flush=True)
    raw_path = str(output_video)
    with open(raw_path, 'wb') as f:
        for frame_data in out_frames:
            f.write(frame_data.tobytes())

    print(f"  Wrote {frame_idx} frames ({os.path.getsize(raw_path) / 1e6:.0f} MB)", flush=True)
    return raw_path


def encode_from_raw(raw_path, output_mkv, width, height,
                    crf=35, film_grain=22, keyint=600, preset=0):
    """Encode raw RGB frames to SVT-AV1."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', raw_path,
        '-vf', f'scale={width}:{height}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', f'film-grain={film_grain}:keyint={keyint}:scd=0',
        '-r', '20',
        str(output_mkv),
    ]
    print(f"  Encoding: crf={crf} fg={film_grain} ki={keyint} {width}x{height}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"  ENCODE FAILED: {r.stderr[:300]}", flush=True)
        return False
    return True


def inflate_and_eval(mkv_path, archive_bytes, unsharp=0.50, kernel_taps=9):
    """Inflate and evaluate using fast_eval approach."""
    import av
    from frame_utils import yuv420_to_rgb
    from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
    from safetensors.torch import load_file

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Inflate
    ktaps = kernel_taps
    if ktaps == 5: _r = torch.tensor([1., 4., 6., 4., 1.])
    elif ktaps == 7: _r = torch.tensor([1., 6., 15., 20., 15., 6., 1.])
    elif ktaps == 9: _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    elif ktaps == 11: _r = torch.tensor([1., 10., 45., 120., 210., 252., 210., 120., 45., 10., 1.])
    else: _r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
    kernel = (torch.outer(_r, _r) / (_r.sum()**2)).to(device).expand(3, 1, ktaps, ktaps)
    pad = ktaps // 2

    raw_path = str(mkv_path).replace('.mkv', '.raw')
    container = av.open(str(mkv_path))
    n = 0
    with open(raw_path, 'wb') as f:
        for frame in container.decode(container.streams.video[0]):
            t = yuv420_to_rgb(frame)
            H, W, _ = t.shape
            if H != H_CAM or W != W_CAM:
                pil = Image.fromarray(t.numpy()).resize((W_CAM, H_CAM), Image.LANCZOS)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(device)
                if unsharp > 0:
                    blur = F.conv2d(F.pad(x, (pad,pad,pad,pad), mode='reflect'), kernel, padding=0, groups=3)
                    x = x + unsharp * (x - blur)
                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()

    # Evaluate
    cache = ROOT / "_sweep_cache" / "gt.pt"
    gt = torch.load(cache, weights_only=True)
    gt_seg, gt_pose = gt['seg'], gt['pose']
    N = gt_seg.shape[0]

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    raw = np.fromfile(raw_path, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)
    sd, pd = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(device).float()
            f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(device).float()
            x = einops.rearrange(torch.stack([f0, f1], dim=1), 'b t h w c -> b t c h w')
            seg_pred = segnet(segnet.preprocess_input(x)).argmax(1)
            sd.extend((seg_pred != gt_seg[i:end].to(device)).float().mean((1,2)).cpu().tolist())
            pose_pred = posenet(posenet.preprocess_input(x))['pose'][:, :6]
            pd.extend((pose_pred - gt_pose[i:end].to(device)).pow(2).mean(1).cpu().tolist())

    seg_d, pose_d = np.mean(sd), np.mean(pd)
    rate = archive_bytes / ORIGINAL_SIZE
    score = 100*seg_d + math.sqrt(10*pose_d) + 25*rate

    del segnet, posenet
    torch.cuda.empty_cache()

    return {
        'score': score,
        'seg': 100*seg_d, 'pose': math.sqrt(10*pose_d), 'rate': 25*rate,
        'archive_bytes': archive_bytes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='segmentation',
                        choices=['segmentation', 'corridor', 'gradient'])
    parser.add_argument('--blur-sigma', type=float, default=5.0)
    parser.add_argument('--crf', type=int, default=35)
    parser.add_argument('--film-grain', type=int, default=22)
    parser.add_argument('--scale', type=float, default=0.45)
    parser.add_argument('--keyint', type=int, default=600)
    parser.add_argument('--unsharp', type=float, default=0.50)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    work = ROOT / "_roi_tmp"
    work.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    print("=" * 70, flush=True)
    print(f"ROI Preprocessing: method={args.method}, blur_sigma={args.blur_sigma}", flush=True)
    print("=" * 70, flush=True)

    # Step 1: Generate ROI maps
    t0 = time.time()
    if args.method == 'segmentation':
        roi_maps = generate_segmentation_roi(args.device)
    elif args.method == 'corridor':
        roi_maps = generate_driving_corridor_roi()
    elif args.method == 'gradient':
        roi_maps = generate_importance_map(args.device)
    print(f"ROI maps generated in {time.time()-t0:.0f}s", flush=True)
    print(f"  Shape: {roi_maps.shape}, range: [{roi_maps.min():.3f}, {roi_maps.max():.3f}]", flush=True)

    if args.visualize:
        viz_dir = ROOT / "viz_roi"
        viz_dir.mkdir(exist_ok=True)
        for i in [0, 100, 300, 600, 900, 1199]:
            if i < roi_maps.shape[0]:
                img = (roi_maps[i].numpy() * 255).astype(np.uint8)
                Image.fromarray(img).save(viz_dir / f"roi_{i:04d}.png")
        print(f"  Saved ROI visualizations to {viz_dir}", flush=True)

    # Step 2: Preprocess video
    t0 = time.time()
    preprocessed_raw = work / "preprocessed.raw"
    preprocess_video(roi_maps, INPUT_VIDEO, preprocessed_raw,
                     blur_sigma=args.blur_sigma)
    print(f"Preprocessing done in {time.time()-t0:.0f}s", flush=True)

    # Step 3: Encode preprocessed video
    t0 = time.time()
    w, h = int(1164 * args.scale) // 2 * 2, int(874 * args.scale) // 2 * 2
    encoded_mkv = work / "0.mkv"
    ok = encode_from_raw(
        str(preprocessed_raw), encoded_mkv,
        w, h, crf=args.crf, film_grain=args.film_grain,
        keyint=args.keyint, preset=0,
    )
    if not ok:
        print("ENCODING FAILED", flush=True)
        return

    # Step 4: Create archive
    zip_path = work / "archive.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(encoded_mkv, '0.mkv')
    archive_bytes = zip_path.stat().st_size
    print(f"Encoded in {time.time()-t0:.0f}s, archive: {archive_bytes/1024:.0f}KB", flush=True)

    # Step 5: Inflate and evaluate
    t0 = time.time()
    results = inflate_and_eval(encoded_mkv, archive_bytes, unsharp=args.unsharp)
    print(f"Eval in {time.time()-t0:.0f}s", flush=True)

    total_time = time.time() - t_total
    print(f"\n{'='*70}", flush=True)
    print(f"RESULT: score={results['score']:.4f}", flush=True)
    print(f"  seg={results['seg']:.4f} pose={results['pose']:.4f} rate={results['rate']:.4f}", flush=True)
    print(f"  archive: {archive_bytes/1024:.0f}KB", flush=True)
    print(f"  total time: {total_time:.0f}s", flush=True)
    print(f"  vs baseline 2.1304: delta={2.1304 - results['score']:.4f}", flush=True)
    print(f"  vs best sweep 2.1038: delta={2.1038 - results['score']:.4f}", flush=True)
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    main()
