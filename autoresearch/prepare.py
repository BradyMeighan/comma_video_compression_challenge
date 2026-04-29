#!/usr/bin/env python
"""
prepare.py — Fixed infrastructure for the autoresearch loop.

██████████████████████████████████████████████████████████████████████████
██  DO NOT MODIFY THIS FILE. The agent only edits train.py.            ██
██████████████████████████████████████████████████████████████████████████

Provides:
  - load_data(device)     → cached dict of rgb, masks, poses, gt tensors
  - evaluate(model, data) → score dict
  - Helper functions for loss computation (differentiable eval chain)
  - GPU memory safety utilities

Data is cached to a single .pt file. First run takes ~30s, after that ~1s.
"""
import sys, os, io, math, time, gc
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import AVVideoDataset, camera_size, segnet_model_input_size, seq_len
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path

# ─── Constants (do not change) ────────────────────────────────────────

MODEL_H, MODEL_W = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512
OUT_H, OUT_W = camera_size[1], camera_size[0]  # 874, 1164
UNCOMPRESSED_SIZE = 37_545_489
MASK_BYTES = 219_588   # CRF=50 AV1 OBU + brotli (pre-measured)
POSE_BYTES = 13_194    # float32 numpy + brotli (pre-measured)

N_PROXY_PAIRS = 100    # subset for fast proxy training
PROXY_SEED = 42
TRAIN_BUDGET_SEC = 300  # 5 minutes wall-clock training

CACHE_DIR = Path(__file__).parent / "_cache"

# ─── GPU memory safety ───────────────────────────────────────────────

def gpu_cleanup():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def gpu_mem_mb():
    """Current GPU memory used in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

# ─── FP4 quantization simulation ─────────────────────────────────────

_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def fp4_round_trip(x, block_size=32):
    """Simulate FP4 blockwise quantize → dequantize."""
    orig = x.shape
    flat = x.reshape(-1)
    pad = (block_size - flat.numel() % block_size) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    ma = blocks.abs().amax(1, keepdim=True)
    sc = torch.where(ma > 0, ma / 6.0, torch.ones_like(ma))
    norm = blocks / sc
    signs = norm < 0
    lvl = _FP4_LEVELS.to(x.device, x.dtype).view(1, 1, -1)
    mag = (norm.abs().unsqueeze(-1) - lvl).abs().argmin(-1)
    q = torch.where(signs, -lvl[0, 0, mag], lvl[0, 0, mag])
    return (q * sc).view(-1)[:x.numel()].view(orig)

def fake_quant_fp4_ste(x, block_size=32):
    """STE-based fake quantization for training."""
    dq = fp4_round_trip(x, block_size)
    return x + (dq - x).detach()

def apply_fp4_to_model(model):
    """Hard-quantize all Conv2d/Embedding weights with quantize_weight=True."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and getattr(m, 'quantize_weight', True):
                m.weight.data = fp4_round_trip(m.weight.data)
            elif isinstance(m, nn.Embedding) and getattr(m, 'quantize_weight', True):
                m.weight.data = fp4_round_trip(m.weight.data)

def estimate_model_bytes(model, brotli_ratio=0.78):
    """Estimate compressed model size after FP4 + brotli."""
    bits = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight
            if getattr(m, 'quantize_weight', True):
                bits += w.numel() * 4 + (w.numel() // 32 + 1) * 16
            else:
                bits += w.numel() * 16
            if m.bias is not None:
                bits += m.bias.numel() * 16
        elif isinstance(m, nn.Embedding):
            if getattr(m, 'quantize_weight', True):
                bits += m.weight.numel() * 4 + (m.weight.numel() // 32 + 1) * 16
            else:
                bits += m.weight.numel() * 16
        elif isinstance(m, nn.Linear):
            bits += m.weight.numel() * 16
            if m.bias is not None:
                bits += m.bias.numel() * 16
        elif isinstance(m, nn.GroupNorm):
            if m.weight is not None: bits += m.weight.numel() * 16
            if m.bias is not None: bits += m.bias.numel() * 16
    return int(bits / 8 * brotli_ratio)

# ─── Differentiable eval-chain helpers ────────────────────────────────

def diff_round(x):
    return x + (x.round() - x).detach()

def diff_rgb_to_yuv6(rgb_chw):
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :2*h2, :2*w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299*r + 0.587*g + 0.114*b).clamp(0., 255.)
    u = ((b - y)/1.772 + 128.).clamp(0., 255.)
    v = ((r - y)/1.402 + 128.).clamp(0., 255.)
    return torch.stack([
        y[:, 0::2, 0::2], y[:, 1::2, 0::2], y[:, 0::2, 1::2], y[:, 1::2, 1::2],
        (u[:, 0::2, 0::2]+u[:, 1::2, 0::2]+u[:, 0::2, 1::2]+u[:, 1::2, 1::2])*0.25,
        (v[:, 0::2, 0::2]+v[:, 1::2, 0::2]+v[:, 0::2, 1::2]+v[:, 1::2, 1::2])*0.25,
    ], dim=1)

def pack_pair_yuv6(f1, f2):
    return torch.cat([diff_rgb_to_yuv6(f1), diff_rgb_to_yuv6(f2)], dim=1)

def get_pose6(posenet, pin):
    out = posenet(pin)
    return (out["pose"] if isinstance(out, dict) else out.pose)[..., :6]

def kl_on_logits(s, t, temp=2.0):
    return F.kl_div(F.log_softmax(s/temp, 1), F.softmax(t/temp, 1), reduction="batchmean") * temp**2

# ─── Data loading ─────────────────────────────────────────────────────

def load_data(device):
    """
    Load proxy training data (100 pairs). Cached after first run.

    Returns dict with:
        rgb:       (100, 2, 874, 1164, 3) uint8 - original frame pairs
        masks:     (100, 384, 512) uint8 - segnet class indices 0-4
        poses:     (100, 6) float32 - posenet 6D pose vectors
        gt_logits: (100, 5, 384, 512) float32 - segnet logits on GT frame2
        gt_pose:   (100, 6) float32 - posenet pose on GT pairs
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"proxy_{N_PROXY_PAIRS}.pt"

    if cache_file.exists():
        print(f"[prepare] Loading cached data ({cache_file.stat().st_size // 1024 // 1024}MB)")
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    print("[prepare] Building data cache (first run only, ~30s)...")
    t0 = time.time()

    # Load all 600 pairs on CPU
    files = [l.strip() for l in (ROOT / "public_test_video_names.txt").read_text().splitlines() if l.strip()]
    ds = AVVideoDataset(files, data_dir=ROOT / "videos", batch_size=16,
                        device=torch.device("cpu"), num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds.prepare_data()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    all_rgb = []
    for _, _, batch in tqdm(dl, desc="[prepare] Loading video", leave=False):
        all_rgb.append(batch)
    all_rgb = torch.cat(all_rgb, 0)
    print(f"[prepare] Loaded {all_rgb.shape[0]} pairs in {time.time()-t0:.1f}s")

    # Subsample
    g = torch.Generator()
    g.manual_seed(PROXY_SEED)
    idx = torch.randperm(all_rgb.shape[0], generator=g)[:N_PROXY_PAIRS]
    rgb = all_rgb[idx].contiguous()
    del all_rgb
    gc.collect()

    # Extract masks + poses + GT on GPU (batched, then free immediately)
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))

    masks_l, poses_l, gt_logits_l = [], [], []
    BS = 8
    with torch.inference_mode():
        for i in tqdm(range(0, N_PROXY_PAIRS, BS), desc="[prepare] Extracting", leave=False):
            b = rgb[i:i+BS].to(device).float()
            bc = einops.rearrange(b, 'b t h w c -> b t c h w')

            r2 = F.interpolate(bc[:, 1], (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            logits = segnet(r2).float()
            masks_l.append(logits.argmax(1).to(torch.uint8).cpu())
            gt_logits_l.append(logits.cpu())

            pin = posenet.preprocess_input(bc)
            p6 = get_pose6(posenet, pin).float()
            poses_l.append(p6.cpu())

    del segnet, posenet
    gpu_cleanup()

    data = {
        "rgb": rgb,
        "masks": torch.cat(masks_l, 0).contiguous(),
        "poses": torch.cat(poses_l, 0).contiguous(),
        "gt_logits": torch.cat(gt_logits_l, 0).contiguous(),
        "gt_pose": torch.cat(poses_l, 0).contiguous(),  # same as poses for GT
    }
    torch.save(data, cache_file)
    print(f"[prepare] Cached {cache_file.stat().st_size // 1024 // 1024}MB in {time.time()-t0:.1f}s")
    return data

# ─── Frozen nets for training ─────────────────────────────────────────

def load_segnet(device):
    """Load frozen SegNet for training loss. Caller must delete when done."""
    net = SegNet().eval().to(device)
    net.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    for p in net.parameters():
        p.requires_grad = False
    return net

def load_posenet(device):
    """Load frozen PoseNet for training loss. Caller must delete when done."""
    net = PoseNet().eval().to(device)
    net.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for p in net.parameters():
        p.requires_grad = False
    return net

# ─── Evaluation ───────────────────────────────────────────────────────

def evaluate(model, data, device, batch_size=16):
    """
    Ground truth evaluation. Applies FP4 quantization, runs full distortion.

    Args:
        model: nn.Module with forward(mask_long, pose_float) -> (f1, f2)
               each (B, 3, 384, 512) in [0, 255]
        data:  dict from load_data()
        device: torch device

    Returns dict with score (lower = better) and breakdown.
    """
    model = model.to(device)
    model.eval()
    apply_fp4_to_model(model)

    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    rgb = data["rgb"]
    masks = data["masks"]
    poses = data["poses"]
    n = rgb.shape[0]

    t_seg, t_pose, t_n = 0.0, 0.0, 0

    with torch.inference_mode():
        for i in range(0, n, batch_size):
            m = masks[i:i+batch_size].to(device).long()
            p = poses[i:i+batch_size].to(device).float()
            f1, f2 = model(m, p)
            comp = torch.stack([
                F.interpolate(f1, (OUT_H, OUT_W), mode="bilinear", align_corners=False),
                F.interpolate(f2, (OUT_H, OUT_W), mode="bilinear", align_corners=False),
            ], dim=1)
            comp = einops.rearrange(comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
            gt = rgb[i:i+batch_size].to(device)
            pd, sd = dist_net.compute_distortion(gt, comp)
            t_seg += sd.sum().item()
            t_pose += pd.sum().item()
            t_n += gt.shape[0]

    del dist_net
    gpu_cleanup()

    avg_seg = t_seg / max(1, t_n)
    avg_pose = t_pose / max(1, t_n)
    model_bytes = estimate_model_bytes(model)
    total_bytes = MASK_BYTES + POSE_BYTES + model_bytes
    rate = total_bytes / UNCOMPRESSED_SIZE
    score = 100 * avg_seg + math.sqrt(max(0, 10 * avg_pose)) + 25 * rate

    return {
        "score": round(score, 6),
        "seg_term": round(100 * avg_seg, 6),
        "pose_term": round(math.sqrt(max(0, 10 * avg_pose)), 6),
        "rate_term": round(25 * rate, 6),
        "model_bytes": model_bytes,
        "total_bytes": total_bytes,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
