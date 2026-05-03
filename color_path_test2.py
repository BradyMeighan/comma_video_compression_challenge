#!/usr/bin/env python
"""
Part 2: Test remaining color path ideas.

Focus on:
1. Different decode upscaling (ffmpeg lanczos vs PyAV + PIL lanczos)
2. Different CRF with piped path to find new optimal
3. "Double pipe" - decode with PyAV, re-encode, decode again
4. What specific pixel-level differences exist between piped vs direct?
"""
import subprocess, sys, os, time, math, zipfile, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, str(Path(__file__).parent))
from frame_utils import camera_size, yuv420_to_rgb
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file
import einops

ROOT = Path(__file__).parent
VIDEO = ROOT / 'videos' / '0.mkv'
W_CAM, H_CAM = camera_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_r9 = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL_9 = (torch.outer(_r9, _r9) / (_r9.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)

TMP = ROOT / '_cpath2_tmp'


def load_models():
    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    posenet = PoseNet().eval().to(DEVICE)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(DEVICE)))
    return segnet, posenet


def load_raw_frames():
    import av
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f) for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


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


def apply_sky_blur(frames_np, sigma=5.0):
    mask = make_sky_mask()
    alpha = mask[..., np.newaxis]
    ksize = int(sigma * 6) | 1
    return [(f * alpha + cv2.GaussianBlur(f, (ksize, ksize), sigma) * (1 - alpha)
             ).astype(np.uint8) for f in frames_np]


def encode_piped(frames, out_mkv, crf='33', scale=0.45):
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20', '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos', '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', '0', '-crf', crf,
        '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
        '-r', '20', str(out_mkv),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    return out_mkv.stat().st_size


def decode_pyav(mkv_path):
    import av
    from PIL import Image
    container = av.open(str(mkv_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        frames.append(f_np)
    container.close()
    return frames


def decode_ffmpeg(mkv_path, flags='lanczos'):
    raw_out = TMP / 'decoded_ff.raw'
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', str(mkv_path),
        '-vf', f'scale={W_CAM}:{H_CAM}:flags={flags}',
        '-pix_fmt', 'rgb24', '-f', 'rawvideo', str(raw_out),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    data = np.fromfile(raw_out, dtype=np.uint8).reshape(-1, H_CAM, W_CAM, 3)
    raw_out.unlink()
    return [data[i] for i in range(len(data))]


def apply_unsharp_and_eval(decoded, archive_size, segnet, posenet, unsharp=0.45):
    raw = bytearray()
    for f_np in decoded:
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + unsharp * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw.extend(t.contiguous().numpy().tobytes())

    gt = torch.load(ROOT / 'submissions' / 'av1_repro' / '_cache' / 'gt.pt',
                    weights_only=True)
    N = gt['seg'].shape[0]
    raw_np = np.frombuffer(bytes(raw), dtype=np.uint8).reshape(N*2, H_CAM, W_CAM, 3)
    seg_dists, pose_dists = [], []
    with torch.inference_mode():
        for i in range(0, N, 16):
            end = min(i + 16, N)
            f0 = torch.from_numpy(raw_np[2*i:2*end:2].copy()).to(DEVICE).float()
            f1 = torch.from_numpy(raw_np[2*i+1:2*end:2].copy()).to(DEVICE).float()
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
    score = 100*s + math.sqrt(10*p) + 25*r
    return score, s, p, r


if __name__ == '__main__':
    print(f"Device: {DEVICE}", flush=True)
    segnet, posenet = load_models()

    TMP.mkdir(exist_ok=True)
    out_mkv = TMP / '0.mkv'

    frames_raw = load_raw_frames()
    frames_np = [f.numpy() for f in frames_raw]
    frames_blur = apply_sky_blur(frames_np)
    print(f"Loaded {len(frames_raw)} frames.", flush=True)

    def test(name, decoded, sz, unsharp=0.45):
        score, s, p, r = apply_unsharp_and_eval(decoded, sz, segnet, posenet, unsharp)
        print(f"[{name:45s}] score={score:.4f} seg={100*s:.4f} "
              f"pose={math.sqrt(10*p):.4f} rate={25*r:.4f} "
              f"size={sz//1024}KB", flush=True)
        return score

    # Encode our best config
    sz = encode_piped(frames_blur, out_mkv)
    print(f"Encoded: {sz//1024} KB", flush=True)

    print("\n=== Decode method comparison ===", flush=True)
    # PyAV decode (our current approach)
    dec_pyav = decode_pyav(out_mkv)
    test("decode_pyav_lanczos", dec_pyav, sz)

    # FFmpeg decode with lanczos
    dec_ff = decode_ffmpeg(out_mkv, 'lanczos')
    test("decode_ffmpeg_lanczos", dec_ff, sz)

    # FFmpeg decode with bicubic
    dec_ff_bic = decode_ffmpeg(out_mkv, 'bicubic')
    test("decode_ffmpeg_bicubic", dec_ff_bic, sz)

    # FFmpeg decode with spline
    dec_ff_spl = decode_ffmpeg(out_mkv, 'spline')
    test("decode_ffmpeg_spline", dec_ff_spl, sz)

    # Pixel-level comparison between PyAV and FFmpeg decode
    print("\n=== Pixel difference analysis ===", flush=True)
    diff = np.abs(dec_pyav[0].astype(np.float32) - dec_ff[0].astype(np.float32))
    print(f"  PyAV vs FFmpeg Lanczos: mean_diff={diff.mean():.4f} max={diff.max():.0f} "
          f"nonzero={np.count_nonzero(diff>0)}/{diff.size}", flush=True)

    diff2 = np.abs(dec_pyav[0].astype(np.float32) - dec_ff_bic[0].astype(np.float32))
    print(f"  PyAV vs FFmpeg bicubic: mean_diff={diff2.mean():.4f} max={diff2.max():.0f}", flush=True)

    # Fine-tune CRF with piped path
    print("\n=== CRF fine-tune (piped + sky blur) ===", flush=True)
    for crf in ['31', '32', '33', '34']:
        sz2 = encode_piped(frames_blur, out_mkv, crf=crf)
        dec = decode_pyav(out_mkv)
        test(f"crf_{crf}", dec, sz2)

    # Fine-tune unsharp with best CRF
    print("\n=== Unsharp fine-tune (piped CRF=33) ===", flush=True)
    sz = encode_piped(frames_blur, out_mkv, crf='33')
    dec = decode_pyav(out_mkv)
    for ush in [0.35, 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50]:
        test(f"ush_{ush}", dec, sz, unsharp=ush)

    # What about torch.nn.functional upscale instead of PIL?
    print("\n=== Torch upscale (bilinear vs bicubic) ===", flush=True)
    import av
    container = av.open(str(out_mkv))
    small_frames = []
    for frame in container.decode(container.streams.video[0]):
        f_np = yuv420_to_rgb(frame).numpy()
        small_frames.append(f_np)
    container.close()
    print(f"  Small frames: {small_frames[0].shape}", flush=True)

    for mode in ['bilinear', 'bicubic']:
        upscaled = []
        for f_np in small_frames:
            x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float()
            x = F.interpolate(x, size=(H_CAM, W_CAM), mode=mode,
                              align_corners=False if mode != 'nearest' else None)
            upscaled.append(x.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(
                torch.uint8).numpy())
        test(f"torch_{mode}_upscale", upscaled, sz)

    shutil.rmtree(TMP, ignore_errors=True)
    print("\nDone.", flush=True)
