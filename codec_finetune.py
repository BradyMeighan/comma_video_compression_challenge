#!/usr/bin/env python
"""
Fine-grained codec-only optimizations targeting the 2.02→1.95 gap.

Tests:
1. Different upscale methods at decode (torch bicubic, PIL Lanczos, different lobes)
2. Pre-sharpening before encode (sharper input → crisper downscale)
3. Per-frame QP file (--use-q-file) for frame-level bit allocation
4. Fractional CRF values (SVT-AV1 supports 0.25 increments)
5. Different film-grain-denoise settings
6. aq-mode sweep with sky blur
7. Combinations of best findings
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


def sky_blur_frames(frames_np, sky_alpha, sigma=5.0):
    ksize = int(sigma * 6) | 1
    return [(f * sky_alpha + cv2.GaussianBlur(f, (ksize, ksize), sigma) * (1 - sky_alpha)
             ).astype(np.uint8) for f in frames_np]


def encode_piped(frames_np, out_mkv, crf=33, scale=0.45, film_grain=22,
                 preset=0, keyint=180, extra_params=''):
    w = int(W_CAM * scale) // 2 * 2
    h = int(H_CAM * scale) // 2 * 2
    params = f'film-grain={film_grain}:keyint={keyint}:scd=0'
    if extra_params:
        params += ':' + extra_params
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
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
        print(f"  ERR: {err[:200]}", flush=True)
        return None
    return out_mkv.stat().st_size


def decode_frames(mkv_path):
    import av
    container = av.open(str(mkv_path))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def inflate_lanczos_unsharp(decoded, strength=0.45, kernel=KERNEL_9, pad=4):
    """Standard: PIL Lanczos upscale + binomial unsharp."""
    from PIL import Image
    raw_bytes = bytearray()
    for f_np in decoded:
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        blur = F.conv2d(F.pad(x, (pad, pad, pad, pad), mode='reflect'),
                        kernel, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())
    return raw_bytes


def inflate_torch_bicubic(decoded, strength=0.45):
    """Torch bicubic upscale + binomial unsharp."""
    raw_bytes = bytearray()
    for f_np in decoded:
        h, w, _ = f_np.shape
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        if h != H_CAM or w != W_CAM:
            x = F.interpolate(x, size=(H_CAM, W_CAM), mode='bicubic',
                              align_corners=False)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())
    return raw_bytes


def inflate_torch_bilinear(decoded, strength=0.45):
    """Torch bilinear upscale + binomial unsharp."""
    raw_bytes = bytearray()
    for f_np in decoded:
        h, w, _ = f_np.shape
        x = torch.from_numpy(f_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
        if h != H_CAM or w != W_CAM:
            x = F.interpolate(x, size=(H_CAM, W_CAM), mode='bilinear',
                              align_corners=False)
        blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'),
                        KERNEL_9, padding=0, groups=3)
        x = x + strength * (x - blur)
        t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)
        raw_bytes.extend(t.contiguous().numpy().tobytes())
    return raw_bytes


def inflate_lanczos_no_unsharp(decoded):
    """Lanczos upscale only, no unsharp."""
    from PIL import Image
    raw_bytes = bytearray()
    for f_np in decoded:
        h, w, _ = f_np.shape
        if h != H_CAM or w != W_CAM:
            pil = Image.fromarray(f_np)
            pil = pil.resize((W_CAM, H_CAM), Image.LANCZOS)
            f_np = np.array(pil)
        raw_bytes.extend(f_np.tobytes())
    return raw_bytes


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


def run_test(name, frames_np, inflate_fn, segnet, posenet,
             crf=33, preset=0, extra_params='', film_grain=22):
    tmp_dir = ROOT / '_fine_tmp'
    tmp_dir.mkdir(exist_ok=True)
    out_mkv = tmp_dir / '0.mkv'
    archive_zip = tmp_dir / 'archive.zip'

    t0 = time.time()
    sz = encode_piped(frames_np, out_mkv, crf=crf, preset=preset,
                      film_grain=film_grain, extra_params=extra_params)
    if sz is None:
        return None

    with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
    archive_size = archive_zip.stat().st_size

    decoded = decode_frames(out_mkv)
    raw = inflate_fn(decoded)
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

    sky_mask = make_sky_mask()
    sky_alpha = sky_mask[..., np.newaxis]
    frames_np = [f.numpy() for f in gt_frames]
    frames_sky = sky_blur_frames(frames_np, sky_alpha)

    results = {}

    # 1. BASELINE
    print("\n=== 1. Baseline (sky blur + Lanczos + unsharp 0.45) ===", flush=True)
    results['baseline'] = run_test("baseline", frames_sky,
                                    lambda d: inflate_lanczos_unsharp(d, 0.45),
                                    segnet, posenet)

    # 2. UPSCALE METHOD TESTS
    print("\n=== 2. Upscale methods ===", flush=True)
    results['bicubic_0.45'] = run_test("bicubic_0.45", frames_sky,
                                        lambda d: inflate_torch_bicubic(d, 0.45),
                                        segnet, posenet)
    results['bilinear_0.45'] = run_test("bilinear_0.45", frames_sky,
                                         lambda d: inflate_torch_bilinear(d, 0.45),
                                         segnet, posenet)
    results['lanczos_no_ush'] = run_test("lanczos_no_ush", frames_sky,
                                          inflate_lanczos_no_unsharp,
                                          segnet, posenet)

    # 3. FRACTIONAL CRF
    print("\n=== 3. Fractional CRF ===", flush=True)
    for crf_val in [32.5, 33.25, 33.5, 33.75, 34]:
        name = f"crf_{crf_val}"
        results[name] = run_test(name, frames_sky,
                                  lambda d: inflate_lanczos_unsharp(d, 0.45),
                                  segnet, posenet, crf=crf_val)

    # 4. AQ MODE SWEEP
    print("\n=== 4. AQ mode ===", flush=True)
    for aq in [0, 1]:
        name = f"aq_{aq}"
        results[name] = run_test(name, frames_sky,
                                  lambda d: inflate_lanczos_unsharp(d, 0.45),
                                  segnet, posenet, extra_params=f'aq-mode={aq}')

    # 5. FILM-GRAIN DENOISE
    print("\n=== 5. Film-grain denoise ===", flush=True)
    results['fgd_1'] = run_test("fgd_1", frames_sky,
                                 lambda d: inflate_lanczos_unsharp(d, 0.45),
                                 segnet, posenet,
                                 extra_params='film-grain-denoise=1')

    # 6. TUNE MODE
    print("\n=== 6. Tune mode ===", flush=True)
    for tune in [0, 2]:
        name = f"tune_{tune}"
        results[name] = run_test(name, frames_sky,
                                  lambda d: inflate_lanczos_unsharp(d, 0.45),
                                  segnet, posenet, extra_params=f'tune={tune}')

    # 7. ENABLE-QM
    print("\n=== 7. Quantization matrices ===", flush=True)
    results['qm_on'] = run_test("qm_on", frames_sky,
                                 lambda d: inflate_lanczos_unsharp(d, 0.45),
                                 segnet, posenet, extra_params='enable-qm=1')

    # 8. SHARPNESS PARAM
    print("\n=== 8. Sharpness ===", flush=True)
    for sh in [-2, -1, 1, 2]:
        name = f"sharpness_{sh}"
        results[name] = run_test(name, frames_sky,
                                  lambda d: inflate_lanczos_unsharp(d, 0.45),
                                  segnet, posenet, extra_params=f'sharpness={sh}')

    # 9. DIFFERENT DOWNSCALE FILTERS
    print("\n=== 9. Downscale filters ===", flush=True)
    for scale_flags in ['bicubic', 'spline', 'bilinear']:
        name = f"down_{scale_flags}"
        w = int(W_CAM * 0.45) // 2 * 2
        h = int(H_CAM * 0.45) // 2 * 2

        tmp_dir = ROOT / '_fine_tmp'
        tmp_dir.mkdir(exist_ok=True)
        out_mkv = tmp_dir / '0.mkv'
        archive_zip = tmp_dir / 'archive.zip'

        t0 = time.time()
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
            '-i', 'pipe:0',
            '-vf', f'scale={w}:{h}:flags={scale_flags}',
            '-pix_fmt', 'yuv420p',
            '-c:v', 'libsvtav1', '-preset', '0', '-crf', '33',
            '-svtav1-params', 'film-grain=22:keyint=180:scd=0',
            '-r', '20', str(out_mkv),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        for frame in frames_sky:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()

        if proc.returncode != 0:
            print(f"  [{name}] FAILED", flush=True)
            continue

        with zipfile.ZipFile(archive_zip, 'w', zipfile.ZIP_STORED) as zf:
            zf.write(out_mkv, '0.mkv')
        archive_size = archive_zip.stat().st_size

        decoded = decode_frames(out_mkv)
        raw = inflate_lanczos_unsharp(decoded, 0.45)
        score, s, p, r = fast_eval(raw, archive_size, segnet, posenet)
        print(f"[{name}] score={score:.4f} seg={100*s:.4f} pose={math.sqrt(10*p):.4f} "
              f"rate={25*r:.4f} ({time.time()-t0:.0f}s)", flush=True)
        results[name] = score

        out_mkv.unlink(missing_ok=True)
        archive_zip.unlink(missing_ok=True)

    # 10. AC-BIAS
    print("\n=== 10. AC bias ===", flush=True)
    for ab in [1.0, 2.0, 4.0]:
        name = f"acbias_{ab}"
        results[name] = run_test(name, frames_sky,
                                  lambda d: inflate_lanczos_unsharp(d, 0.45),
                                  segnet, posenet,
                                  extra_params=f'ac-bias={ab}')

    # SUMMARY
    print(f"\n{'='*70}")
    print("RESULTS (sorted by score):")
    print(f"{'='*70}")
    for name, score in sorted(results.items(), key=lambda x: (x[1] is None, x[1] or 999)):
        if score is not None:
            delta = score - results.get('baseline', score)
            print(f"  {name:25s} {score:.4f} (delta={delta:+.4f})")

    shutil.rmtree(ROOT / '_fine_tmp', ignore_errors=True)
