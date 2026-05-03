#!/usr/bin/env python
"""
Optimized compression: piped RGB + sky blur + SVT-AV1 encode.
Reads frames via PyAV (matching decode color path), applies gentle sky denoising,
and pipes raw RGB to ffmpeg for encoding.

Usage: python -m submissions.av1_repro.compress
"""
import subprocess, os, zipfile, sys
from pathlib import Path

import numpy as np
import cv2

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
VIDEO = ROOT / 'videos' / '0.mkv'
ARCHIVE_DIR = HERE / 'archive'
ARCHIVE_ZIP = HERE / 'archive.zip'
DX_META = ARCHIVE_DIR / 'frame0_dx_q.bin'
DXY_META = ARCHIVE_DIR / 'frame0_dxy_q.bin'
DXYR_META = ARCHIVE_DIR / 'frame0_dxyr_q.bin'
DXYR2_META = ARCHIVE_DIR / 'frame0_dxyr2_q.bin'
AB_META = ARCHIVE_DIR / 'frame0_ab_q.bin'
AFFINE_META = ARCHIVE_DIR / 'frame0_affine_q.bin'

W_CAM, H_CAM = 1164, 874

# ── Hand-tuned driving corridor polygons (normalized coords, from PR #49) ──
SEGMENTS = [
    (0,   299, [(0.14, 0.52), (0.82, 0.48), (0.98, 1.00), (0.05, 1.00)]),
    (300, 599, [(0.10, 0.50), (0.76, 0.47), (0.92, 1.00), (0.00, 1.00)]),
    (600, 899, [(0.18, 0.50), (0.84, 0.47), (0.98, 1.00), (0.06, 1.00)]),
    (900, 1199,[(0.22, 0.52), (0.90, 0.49), (1.00, 1.00), (0.10, 1.00)]),
]


def build_roi_mask(frame_idx, feather=24):
    """Soft ROI mask: 1=inside corridor (preserve), 0=outside (denoise)."""
    poly_norm = SEGMENTS[-1][2]
    for start, end, p in SEGMENTS:
        if start <= frame_idx <= end:
            poly_norm = p
            break
    pts = np.array([(int(x * W_CAM), int(y * H_CAM)) for x, y in poly_norm], dtype=np.int32)
    mask = np.zeros((H_CAM, W_CAM), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    mask_f = mask.astype(np.float32) / 255.0
    if feather > 0:
        ksize = feather * 4 + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), feather)
    return mask_f


def denoise_outside_roi(frame, mask, luma_strength=2.5, chroma_k=5, blend=0.50):
    """Denoise outside driving corridor in YUV space (matches leader's approach)."""
    f = frame.astype(np.float32)
    # RGB → YUV
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = (b - y) / 1.772 + 128.0
    v = (r - y) / 1.402 + 128.0

    # Luma: light Gaussian denoise
    sigma = max(0.1, luma_strength * 0.35)
    ksize = 3 if luma_strength <= 2.0 else 5
    y_blur = cv2.GaussianBlur(y, (ksize, ksize), sigma)
    luma_blend = min(0.9, luma_strength / 3.0)
    y_dn = (1 - luma_blend) * y + luma_blend * y_blur

    # Chroma: avg_pool blur
    u_dn = cv2.blur(u, (chroma_k, chroma_k))
    v_dn = cv2.blur(v, (chroma_k, chroma_k))

    # YUV → RGB (denoised)
    u2, v2 = u_dn - 128.0, v_dn - 128.0
    r_dn = y_dn + 1.402 * v2
    g_dn = y_dn - 0.344136 * u2 - 0.714136 * v2
    b_dn = y_dn + 1.772 * u2
    denoised = np.stack([r_dn, g_dn, b_dn], axis=-1)

    # Blend outside corridor
    alpha = mask[..., np.newaxis]
    outside = (1.0 - alpha) * blend
    result = f * (1.0 - outside) + denoised * outside
    return np.clip(result, 0, 255).astype(np.uint8)


def load_frames():
    import av
    sys.path.insert(0, str(ROOT))
    from frame_utils import yuv420_to_rgb
    container = av.open(str(VIDEO))
    frames = [yuv420_to_rgb(f).numpy() for f in container.decode(container.streams.video[0])]
    container.close()
    return frames


def compress(crf=33, scale=0.45, film_grain=22, preset=0, keyint=180):
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_mkv = ARCHIVE_DIR / '0.mkv'

    w = int(1164 * scale) // 2 * 2
    h = int(874 * scale) // 2 * 2

    print(f"Loading and preprocessing frames...", flush=True)
    frames = load_frames()

    # ROI-aware preprocessing: denoise outside driving corridor
    print("Applying ROI denoise (hand-tuned corridors, YUV-space)...", flush=True)
    processed = []
    for i, frame in enumerate(frames):
        mask = build_roi_mask(i, feather=24)
        processed.append(denoise_outside_roi(frame, mask))
    frames = processed

    print(f"Encoding: crf={crf} scale={scale} fg={film_grain} "
          f"preset={preset} ki={keyint} res={w}x{h}", flush=True)

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{W_CAM}x{H_CAM}', '-r', '20',
        '-i', 'pipe:0',
        '-vf', f'scale={w}:{h}:flags=lanczos',
        '-pix_fmt', 'yuv420p',
        '-c:v', 'libsvtav1', '-preset', str(preset), '-crf', str(crf),
        '-svtav1-params', f'film-grain={film_grain}:keyint={keyint}:scd=0',
        '-r', '20',
        str(out_mkv),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        print(f"  ENCODE FAILED: {proc.stderr.read().decode()[:300]}", flush=True)
        return None

    mkv_size = out_mkv.stat().st_size
    print(f"  MKV: {mkv_size:,} bytes ({mkv_size/1024:.1f} KB)")

    with zipfile.ZipFile(ARCHIVE_ZIP, 'w', zipfile.ZIP_STORED) as zf:
        zf.write(out_mkv, '0.mkv')
        odd_ab_chain = ARCHIVE_DIR / 'frame1_ab_chain_q.bin'
        odd_dxyr_chain = ARCHIVE_DIR / 'frame1_dxyr_chain_q.bin'
        odd_ab_sidecars = [odd_ab_chain] if odd_ab_chain.exists() else sorted(ARCHIVE_DIR.glob('frame1_ab_q*.bin'))
        odd_dxyr_sidecars = [odd_dxyr_chain] if odd_dxyr_chain.exists() else sorted(ARCHIVE_DIR.glob('frame1_dxyr_q*.bin'))
        if DXYR_META.exists():
            zf.write(DXYR_META, DXYR_META.name)
            if AB_META.exists():
                zf.write(AB_META, AB_META.name)
            if DXYR2_META.exists():
                zf.write(DXYR2_META, DXYR2_META.name)
            if AFFINE_META.exists():
                zf.write(AFFINE_META, AFFINE_META.name)
            for p in odd_dxyr_sidecars:
                zf.write(p, p.name)
            for p in odd_ab_sidecars:
                zf.write(p, p.name)
        elif DXY_META.exists():
            zf.write(DXY_META, DXY_META.name)
            if AB_META.exists():
                zf.write(AB_META, AB_META.name)
            if AFFINE_META.exists():
                zf.write(AFFINE_META, AFFINE_META.name)
            for p in odd_dxyr_sidecars:
                zf.write(p, p.name)
            for p in odd_ab_sidecars:
                zf.write(p, p.name)
        elif DX_META.exists():
            zf.write(DX_META, DX_META.name)
            if AB_META.exists():
                zf.write(AB_META, AB_META.name)
            if AFFINE_META.exists():
                zf.write(AFFINE_META, AFFINE_META.name)
            for p in odd_dxyr_sidecars:
                zf.write(p, p.name)
            for p in odd_ab_sidecars:
                zf.write(p, p.name)

    zip_size = ARCHIVE_ZIP.stat().st_size
    rate = zip_size / 37_545_489
    print(f"  archive.zip: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"  25*rate = {25*rate:.4f}")
    return zip_size


if __name__ == '__main__':
    compress()
