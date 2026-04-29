#!/usr/bin/env python
"""
prepare.py — Fixed infrastructure for the autoresearch loop.

██████████████████████████████████████████████████████████████████████████
██  DO NOT MODIFY THIS FILE. The agent only edits train.py.            ██
██████████████████████████████████████████████████████████████████████████

Provides:
  - load_data(device)     → dict with train + val splits (80/20)
  - evaluate(model, data) → score on HELD-OUT val set (not training data)
  - Helper functions for loss computation

CRITICAL DESIGN: eval uses a 20-pair HELD-OUT validation set that the
model never trains on. This ensures proxy improvements transfer to full
training — we're measuring generalization, not memorization.
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

N_TOTAL_PROXY = 100     # total subset from 600
N_TRAIN = 80            # training split
N_VAL = 20              # held-out validation split
PROXY_SEED = 42
TRAIN_BUDGET_SEC = 300  # 5 minutes wall-clock training

CACHE_DIR = Path(__file__).parent / "_cache"

# ─── GPU memory safety ───────────────────────────────────────────────

def gpu_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0

# ─── FP4 quantization simulation ─────────────────────────────────────

_FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

def fp4_round_trip(x, block_size=32):
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
    dq = fp4_round_trip(x, block_size)
    return x + (dq - x).detach()

def apply_fp4_to_model(model):
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and getattr(m, 'quantize_weight', True):
                m.weight.data = fp4_round_trip(m.weight.data)
            elif isinstance(m, nn.Embedding) and getattr(m, 'quantize_weight', True):
                m.weight.data = fp4_round_trip(m.weight.data)

def estimate_model_bytes(model, brotli_ratio=0.78):
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
    Load proxy data with train/val split. Cached after first run.

    Returns dict with:
        train_rgb:    (80, 2, 874, 1164, 3) uint8
        train_masks:  (80, 384, 512) uint8
        train_poses:  (80, 6) float32
        val_rgb:      (20, 2, 874, 1164, 3) uint8
        val_masks:    (20, 384, 512) uint8
        val_poses:    (20, 6) float32
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "proxy_split_v2.pt"

    if cache_file.exists():
        print(f"[prepare] Loading cached data ({cache_file.stat().st_size // 1024 // 1024}MB)")
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    print("[prepare] Building data cache with train/val split (first run only, ~30s)...")
    t0 = time.time()

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

    g = torch.Generator()
    g.manual_seed(PROXY_SEED)
    idx = torch.randperm(all_rgb.shape[0], generator=g)[:N_TOTAL_PROXY]
    rgb = all_rgb[idx].contiguous()
    del all_rgb
    gc.collect()

    # Split: first 80 = train, last 20 = val
    train_rgb = rgb[:N_TRAIN].contiguous()
    val_rgb = rgb[N_TRAIN:].contiguous()
    del rgb

    # Extract masks + poses on GPU
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))

    def extract(rgb_subset):
        masks_l, poses_l = [], []
        BS = 8
        with torch.inference_mode():
            for i in range(0, rgb_subset.shape[0], BS):
                b = rgb_subset[i:i+BS].to(device).float()
                bc = einops.rearrange(b, 'b t h w c -> b t c h w')
                r2 = F.interpolate(bc[:, 1], (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
                logits = segnet(r2).float()
                masks_l.append(logits.argmax(1).to(torch.uint8).cpu())
                pin = posenet.preprocess_input(bc)
                p6 = get_pose6(posenet, pin).float()
                poses_l.append(p6.cpu())
        return torch.cat(masks_l, 0).contiguous(), torch.cat(poses_l, 0).contiguous()

    print("[prepare] Extracting train split...")
    train_masks, train_poses = extract(train_rgb)
    print("[prepare] Extracting val split...")
    val_masks, val_poses = extract(val_rgb)

    del segnet, posenet
    gpu_cleanup()

    data = {
        "train_rgb": train_rgb,
        "train_masks": train_masks,
        "train_poses": train_poses,
        "val_rgb": val_rgb,
        "val_masks": val_masks,
        "val_poses": val_poses,
    }
    torch.save(data, cache_file)
    print(f"[prepare] Cached {cache_file.stat().st_size // 1024 // 1024}MB in {time.time()-t0:.1f}s")
    return data

# ─── Frozen nets for training ─────────────────────────────────────────

def load_segnet(device):
    net = SegNet().eval().to(device)
    net.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    for p in net.parameters():
        p.requires_grad = False
    return net

def load_posenet(device):
    net = PoseNet().eval().to(device)
    net.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for p in net.parameters():
        p.requires_grad = False
    return net

# ─── Evaluation (on HELD-OUT val set) ─────────────────────────────────

def evaluate(model, data, device, batch_size=16):
    """
    Ground truth evaluation on HELD-OUT validation set.

    This evaluates on the 20 val pairs that the model NEVER trained on.
    This ensures proxy scores correlate with full-training generalization.
    """
    model = model.to(device)
    model.eval()
    apply_fp4_to_model(model)

    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    rgb = data["val_rgb"]
    masks = data["val_masks"]
    poses = data["val_poses"]
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
