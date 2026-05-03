#!/usr/bin/env python
"""
Decode AV1 video: Lanczos upscale + binomial unsharp mask → .raw
Optional: apply per-pair even-frame horizontal shifts from sidecar metadata.

Usage: python -m submissions.evenframe_meta_v1.inflate <src_mkv> <dst_raw>
"""
import sys, bz2, struct, av, cv2, torch, numpy as np
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from frame_utils import camera_size, yuv420_to_rgb

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 9-tap binomial kernel (same as leader)
_r = torch.tensor([1., 8., 28., 56., 70., 56., 28., 8., 1.])
KERNEL = (torch.outer(_r, _r) / (_r.sum()**2)).to(DEVICE).expand(3, 1, 9, 9)
STRENGTH = 0.45
MODEL_W = 512
MODEL_H = 384


def load_even_frame_shift(video_path: str):
    """
    Load optional sidecar produced offline:
    - frame0_dxyr_q.bin: bz2-compressed int8 (N,3) in 0.1 model-px / 0.1 deg
    - frame0_dxy_q.bin:  bz2-compressed int8 (N,2) in 0.1 model-px units
    - frame0_dx_q.bin:  bz2-compressed int8 (N,)  in 0.1 model-px units
    Returns (dx_full, dy_full, th_deg) arrays or (None, None, None).
    """
    base = Path(video_path).parent
    p_dxyr = base / "frame0_dxyr_q.bin"
    p_dxy = base / "frame0_dxy_q.bin"
    p_dx = base / "frame0_dx_q.bin"

    try:
        if p_dxyr.exists():
            q = np.frombuffer(bz2.decompress(p_dxyr.read_bytes()), dtype=np.int8).astype(np.float32)
            q = q.reshape(-1, 3) / 10.0
            dx = q[:, 0] * (camera_size[0] / MODEL_W)
            dy = q[:, 1] * (camera_size[1] / MODEL_H)
            th = q[:, 2]  # degrees
            return dx, dy, th
        if p_dxy.exists():
            q = np.frombuffer(bz2.decompress(p_dxy.read_bytes()), dtype=np.int8).astype(np.float32)
            q = q.reshape(-1, 2) / 10.0
            dx = q[:, 0] * (camera_size[0] / MODEL_W)
            dy = q[:, 1] * (camera_size[1] / MODEL_H)
            return dx, dy, np.zeros_like(dx)
        if p_dx.exists():
            q = np.frombuffer(bz2.decompress(p_dx.read_bytes()), dtype=np.int8).astype(np.float32)
            q = (q / 10.0) * (camera_size[0] / MODEL_W)
            return q, np.zeros_like(q), np.zeros_like(q)
        return None, None, None
    except Exception:
        return None, None, None


def load_even_frame_ab(video_path: str):
    """
    Optional photometric sidecar:
    - frame0_ab_q.bin: bz2-compressed int8 (N,2)
      a_q: scale around 1.0 in 0.01 units -> a = 1 + a_q/100
      b_q: bias in 1.0 units
    Returns (a, b) arrays or (None, None).
    """
    p = Path(video_path).parent / "frame0_ab_q.bin"
    if not p.exists():
        return None, None
    try:
        q = np.frombuffer(bz2.decompress(p.read_bytes()), dtype=np.int8).astype(np.float32).reshape(-1, 2)
        a = 1.0 + q[:, 0] / 100.0
        b = q[:, 1]
        return a, b
    except Exception:
        return None, None


def load_odd_frame_ab_chain(video_path: str):
    """
    Optional odd-frame photometric sidecars.

    Loads `frame1_ab_q*.bin` in lexical order and applies them sequentially:
      arr <- clip(arr * a_k[pair] + b_k[pair], 0, 255)
    """
    base = Path(video_path).parent

    # Preferred packed chain format: one file, many stages.
    p_chain = base / "frame1_ab_chain_q.bin"
    if p_chain.exists():
        try:
            blob = bz2.decompress(p_chain.read_bytes())
            if len(blob) < 8:
                return []
            k, n = struct.unpack("<II", blob[:8])
            arr = np.frombuffer(blob[8:], dtype=np.int8)
            arr = arr.reshape(k, n, 2).astype(np.float32)
            out = []
            for i in range(k):
                a = 1.0 + arr[i, :, 0] / 100.0
                b = arr[i, :, 1]
                out.append((a, b))
            return out
        except Exception:
            return []

    # Fallback: legacy multi-file chain.
    paths = sorted(base.glob("frame1_ab_q*.bin"))
    if not paths:
        return []
    out = []
    for p in paths:
        try:
            q = np.frombuffer(bz2.decompress(p.read_bytes()), dtype=np.int8).astype(np.float32).reshape(-1, 2)
            a = 1.0 + q[:, 0] / 100.0
            b = q[:, 1]
            out.append((a, b))
        except Exception:
            continue
    return out


def load_even_frame_shift_residual(video_path: str):
    """
    Optional second-stage residual geometry:
    - frame0_dxyr2_q.bin: bz2-compressed int8 (N,3) in 0.1 model-px / 0.1 deg
    """
    p = Path(video_path).parent / "frame0_dxyr2_q.bin"
    if not p.exists():
        return None, None, None
    try:
        q = np.frombuffer(bz2.decompress(p.read_bytes()), dtype=np.int8).astype(np.float32)
        q = q.reshape(-1, 3) / 10.0
        dx = q[:, 0] * (camera_size[0] / MODEL_W)
        dy = q[:, 1] * (camera_size[1] / MODEL_H)
        th = q[:, 2]
        return dx, dy, th
    except Exception:
        return None, None, None


def load_even_frame_affine(video_path: str):
    """
    Optional residual affine sidecar:
    - frame0_affine_q.bin: bz2-compressed int8 (N,3)
      scale_q: 0.001 units around 1.0
      shx_q, shy_q: 0.001 shear units
    """
    p = Path(video_path).parent / "frame0_affine_q.bin"
    if not p.exists():
        return None, None, None
    try:
        q = np.frombuffer(bz2.decompress(p.read_bytes()), dtype=np.int8).astype(np.float32).reshape(-1, 3)
        scale = 1.0 + (q[:, 0] / 1000.0)
        shx = q[:, 1] / 1000.0
        shy = q[:, 2] / 1000.0
        return scale, shx, shy
    except Exception:
        return None, None, None


def apply_center_affine(arr: np.ndarray, scale: float, shx: float, shy: float) -> np.ndarray:
    h, w = arr.shape[:2]
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    A = np.array([[scale, shx], [shy, scale]], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    t = c - A @ c
    M = np.array([[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]]], dtype=np.float32)
    return cv2.warpAffine(
        arr,
        M,
        dsize=(w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def decode_and_resize(video_path: str, dst: str):
    target_w, target_h = camera_size
    even_dx, even_dy, even_th = load_even_frame_shift(video_path)
    even_a, even_b = load_even_frame_ab(video_path)
    odd_ab_chain = load_odd_frame_ab_chain(video_path)
    even_dx2, even_dy2, even_th2 = load_even_frame_shift_residual(video_path)
    even_scale, even_shx, even_shy = load_even_frame_affine(video_path)
    container = av.open(video_path)
    stream = container.streams.video[0]
    n = 0
    with open(dst, 'wb') as f:
        for frame in container.decode(stream):
            t = yuv420_to_rgb(frame)  # (H, W, 3)
            H, W, _ = t.shape
            if H != target_h or W != target_w:
                pil = Image.fromarray(t.numpy())
                pil = pil.resize((target_w, target_h), Image.LANCZOS)
                x = torch.from_numpy(np.array(pil)).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                blur = F.conv2d(F.pad(x, (4, 4, 4, 4), mode='reflect'), KERNEL, padding=0, groups=3)
                x = x + STRENGTH * (x - blur)

                t = x.clamp(0, 255).squeeze(0).permute(1, 2, 0).round().cpu().to(torch.uint8)

            # Pose-correction sidecars are applied only on EVEN frames (frame0 of each pair).
            if n % 2 == 0:
                pair_idx = n // 2
                arr = t.contiguous().numpy()
                changed = False

                if even_dx is not None and pair_idx < even_dx.shape[0]:
                    dx = float(even_dx[pair_idx])
                    dy = float(even_dy[pair_idx]) if even_dy is not None else 0.0
                    th = float(even_th[pair_idx]) if even_th is not None else 0.0
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6 or abs(th) > 1e-6:
                        center = (target_w * 0.5, target_h * 0.5)
                        M = cv2.getRotationMatrix2D(center, th, 1.0)
                        M[0, 2] += dx
                        M[1, 2] += dy
                        arr = cv2.warpAffine(
                            arr,
                            M,
                            dsize=(target_w, target_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                        changed = True

                if even_a is not None and pair_idx < even_a.shape[0]:
                    arr = np.clip(
                        arr.astype(np.float32) * float(even_a[pair_idx]) + float(even_b[pair_idx]),
                        0,
                        255,
                    ).astype(np.uint8)
                    changed = True

                if even_dx2 is not None and pair_idx < even_dx2.shape[0]:
                    dx2 = float(even_dx2[pair_idx])
                    dy2 = float(even_dy2[pair_idx]) if even_dy2 is not None else 0.0
                    th2 = float(even_th2[pair_idx]) if even_th2 is not None else 0.0
                    if abs(dx2) > 1e-6 or abs(dy2) > 1e-6 or abs(th2) > 1e-6:
                        center = (target_w * 0.5, target_h * 0.5)
                        M2 = cv2.getRotationMatrix2D(center, th2, 1.0)
                        M2[0, 2] += dx2
                        M2[1, 2] += dy2
                        arr = cv2.warpAffine(
                            arr,
                            M2,
                            dsize=(target_w, target_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                        changed = True

                if even_scale is not None and pair_idx < even_scale.shape[0]:
                    s = float(even_scale[pair_idx])
                    sx = float(even_shx[pair_idx]) if even_shx is not None else 0.0
                    sy = float(even_shy[pair_idx]) if even_shy is not None else 0.0
                    if abs(s - 1.0) > 1e-6 or abs(sx) > 1e-6 or abs(sy) > 1e-6:
                        arr = apply_center_affine(arr, s, sx, sy)
                        changed = True

                if changed:
                    t = torch.from_numpy(arr)
            elif odd_ab_chain:
                pair_idx = n // 2
                arr = t.contiguous().numpy()
                changed = False
                for a_arr, b_arr in odd_ab_chain:
                    if pair_idx < a_arr.shape[0]:
                        a1 = float(a_arr[pair_idx])
                        b1 = float(b_arr[pair_idx])
                        if abs(a1 - 1.0) > 1e-6 or abs(b1) > 1e-6:
                            arr = np.clip(arr.astype(np.float32) * a1 + b1, 0, 255).astype(np.uint8)
                            changed = True
                if changed:
                    t = torch.from_numpy(arr)
            f.write(t.contiguous().numpy().tobytes())
            n += 1
    container.close()
    return n


if __name__ == '__main__':
    src, dst = sys.argv[1], sys.argv[2]
    n = decode_and_resize(src, dst)
    print(f"Decoded {n} frames")
