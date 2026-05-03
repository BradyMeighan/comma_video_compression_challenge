#!/usr/bin/env python
"""
Fast proxy training & evaluation for architecture/hyperparameter search.

Trains on a SUBSET of pairs for a SHORT number of epochs, then evaluates
with the real DistortionNet to get a proxy score that correlates with
full-run performance.

Usage:
  # Single config test (~5 min on a T4/3060):
  python -m submissions.mask2frame_v1.fast_probe --config '{"c1":56,"c2":64}'

  # Automated sweep (reads configs from sweep_configs.json):
  python -m submissions.mask2frame_v1.fast_probe --sweep sweep_configs.json

  # Generate default sweep file:
  python -m submissions.mask2frame_v1.fast_probe --gen-sweep
"""
import sys, os, io, json, math, time, argparse, logging, hashlib, itertools
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file

MODEL_H, MODEL_W = 384, 512
OUT_H, OUT_W = camera_size[1], camera_size[0]
UNCOMPRESSED = 37_545_489

# ─── Config dataclass ────────────────────────────────────────────────

@dataclass
class ProbeConfig:
    # Architecture
    c1: int = 56
    c2: int = 64
    emb_dim: int = 6
    cond_dim: int = 48
    head_hidden: int = 52
    depth_mult: int = 1
    num_classes: int = 5
    pose_dim: int = 6

    # Training
    anchor_epochs: int = 40
    finetune_epochs: int = 20
    joint_epochs: int = 15
    lr: float = 5e-4
    ft_lr: float = 5e-5
    jt_lr: float = 1e-5
    batch_size: int = 4
    err_boost: float = 9.0
    err_boost_high: float = 49.0
    ce_weight: float = 1.0
    pose_weight: float = 1.0
    grad_clip: float = 1.0

    # QAT
    qat_start_frac: float = 0.5  # start QAT at this fraction of anchor epochs
    block_size: int = 32

    # Proxy settings
    n_pairs: int = 100  # subset size (out of 600)
    seed: int = 42

    # Mask CRF (pre-computed, not re-encoded per probe)
    mask_crf: int = 50

    def config_hash(self):
        d = asdict(self)
        for k in ['n_pairs', 'seed', 'mask_crf']:
            d.pop(k, None)
        return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]

# ─── Reuse architecture from compress.py ──────────────────────────────

def diff_round(x): return x + (x.round() - x).detach()

def diff_rgb_to_yuv6(rgb_chw):
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :2*h2, :2*w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299*r + 0.587*g + 0.114*b).clamp(0., 255.)
    u = ((b - y)/1.772 + 128.).clamp(0., 255.)
    v = ((r - y)/1.402 + 128.).clamp(0., 255.)
    return torch.stack([y[:, 0::2, 0::2], y[:, 1::2, 0::2], y[:, 0::2, 1::2], y[:, 1::2, 1::2],
                        (u[:, 0::2, 0::2]+u[:, 1::2, 0::2]+u[:, 0::2, 1::2]+u[:, 1::2, 1::2])*0.25,
                        (v[:, 0::2, 0::2]+v[:, 1::2, 0::2]+v[:, 0::2, 1::2]+v[:, 1::2, 1::2])*0.25], dim=1)

def pack_pair_yuv6(f1, f2): return torch.cat([diff_rgb_to_yuv6(f1), diff_rgb_to_yuv6(f2)], dim=1)

def get_pose6(posenet, pin):
    out = posenet(pin)
    return (out["pose"] if isinstance(out, dict) else out.pose)[..., :6]

def kl_on_logits(s, t, temp=2.0):
    return F.kl_div(F.log_softmax(s/temp, 1), F.softmax(t/temp, 1), reduction="batchmean") * temp**2

class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    @staticmethod
    def quantize_blockwise(x, block_size=32):
        orig = x.shape; flat = x.reshape(-1)
        pad = (block_size - flat.numel() % block_size) % block_size
        if pad: flat = F.pad(flat, (0, pad))
        blocks = flat.view(-1, block_size)
        ma = blocks.abs().amax(1, keepdim=True)
        scales = torch.where(ma > 0, ma / 6.0, torch.ones_like(ma))
        norm = blocks / scales
        signs = (norm < 0).to(torch.int16)
        levels = FP4Codebook.pos_levels.to(x.device, x.dtype).view(1, 1, -1)
        mag = (norm.abs().unsqueeze(-1) - levels).abs().argmin(-1).to(torch.int16)
        q = torch.where(signs.bool(), -levels[0, 0, mag.long()], levels[0, 0, mag.long()])
        return (q * scales).view(-1)[:x.numel()].view(orig), ((signs << 3) | mag).to(torch.uint8), scales.squeeze(1)

def fake_quant_fp4_ste(x, bs=32):
    dq, _, _ = FP4Codebook.quantize_blockwise(x, bs)
    return x + (dq - x).detach()

def pack_nibbles(nib):
    flat = nib.reshape(-1)
    if flat.numel() % 2: flat = F.pad(flat, (0, 1))
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)

# ─── Quantizable modules ─────────────────────────────────────────────

class QConv2d(nn.Conv2d):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.block_size, self.quantize_weight, self.qat_enabled = block_size, quantize_weight, False
    def set_qat(self, e): self.qat_enabled = e
    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QEmbedding(nn.Embedding):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.block_size, self.quantize_weight, self.qat_enabled = block_size, quantize_weight, False
    def set_qat(self, e): self.qat_enabled = e
    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx)

# ─── Parameterized architecture ──────────────────────────────────────

class SepConvGNAct(nn.Module):
    def __init__(self, i, o, k=3, s=1, dm=4, qw=True):
        super().__init__()
        m = i * dm
        self.dw = QConv2d(i, m, k, stride=s, padding=k//2, groups=i, bias=False, quantize_weight=qw)
        self.pw = QConv2d(m, o, 1, bias=True, quantize_weight=qw)
        self.n = nn.GroupNorm(min(2, o), o)
        self.a = nn.SiLU(inplace=True)
    def forward(self, x): return self.a(self.n(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, i, o, k=3, s=1, dm=4, qw=True):
        super().__init__()
        m = i * dm
        self.dw = QConv2d(i, m, k, stride=s, padding=k//2, groups=i, bias=False, quantize_weight=qw)
        self.pw = QConv2d(m, o, 1, bias=True, quantize_weight=qw)
    def forward(self, x): return self.pw(self.dw(x))

class SepRes(nn.Module):
    def __init__(self, ch, dm=4, qw=True):
        super().__init__()
        self.c1 = SepConvGNAct(ch, ch, 3, 1, dm, qw)
        self.c2 = SepConv(ch, ch, 3, 1, dm, qw)
        self.n = nn.GroupNorm(min(2, ch), ch)
        self.a = nn.SiLU(inplace=True)
    def forward(self, x): return self.a(x + self.n(self.c2(self.c1(x))))

class FiLMRes(nn.Module):
    def __init__(self, ch, cd, dm=4, qw=True):
        super().__init__()
        self.c1 = SepConvGNAct(ch, ch, 3, 1, dm, qw)
        self.c2 = SepConv(ch, ch, 3, 1, dm, qw)
        self.n = nn.GroupNorm(min(2, ch), ch)
        self.film = nn.Linear(cd, ch * 2)
        self.a = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        r = self.n(self.c2(self.c1(x)))
        g, b = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        return self.a(x + r * (1.0 + g) + b)

def make_coords(b, h, w, dev, dt):
    ys = (torch.arange(h, device=dev, dtype=dt) + 0.5) / h
    xs = (torch.arange(w, device=dev, dtype=dt) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx*2-1, yy*2-1], 0).unsqueeze(0).expand(b, -1, -1, -1)

class Trunk(nn.Module):
    def __init__(self, nc=5, ed=6, c1=56, c2=64, dm=1):
        super().__init__()
        self.emb = QEmbedding(nc, ed, quantize_weight=False)
        self.stem = SepConvGNAct(ed+2, c1, dm=dm)
        self.sb = SepRes(c1, dm=dm)
        self.dc = SepConvGNAct(c1, c2, stride=2, dm=dm)
        self.db = SepRes(c2, dm=dm)
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), SepConvGNAct(c2, c1, dm=dm))
        self.fuse = SepConvGNAct(c1+c1, c1, dm=dm)
        self.fb = SepRes(c1, dm=dm)
    def forward(self, mask, coords):
        e = F.interpolate(self.emb(mask.long()).permute(0,3,1,2), coords.shape[-2:], mode="bilinear", align_corners=False)
        s = self.sb(self.stem(torch.cat([e, coords], 1)))
        z = self.up(self.db(self.dc(s)))
        return self.fb(self.fuse(torch.cat([z, s], 1)))

class H2(nn.Module):
    def __init__(self, ic, hid=36, dm=4):
        super().__init__()
        self.b1 = SepRes(ic, dm=dm); self.b2 = SepRes(ic, dm=dm)
        self.pre = SepConvGNAct(ic, hid, dm=dm)
        self.head = QConv2d(hid, 3, 1, quantize_weight=False)
    def forward(self, f): return torch.sigmoid(self.head(self.pre(self.b2(self.b1(f))))) * 255.0

class H1(nn.Module):
    def __init__(self, ic, cd=48, hid=36, dm=4):
        super().__init__()
        self.b1 = FiLMRes(ic, cd, dm=dm); self.b2 = SepRes(ic, dm=dm)
        self.pre = SepConvGNAct(ic, hid, dm=dm)
        self.head = QConv2d(hid, 3, 1, quantize_weight=False)
    def forward(self, f, c): return torch.sigmoid(self.head(self.pre(self.b2(self.b1(f, c))))) * 255.0

class Gen(nn.Module):
    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.trunk = Trunk(cfg.num_classes, cfg.emb_dim, cfg.c1, cfg.c2, cfg.depth_mult)
        self.pose_mlp = nn.Sequential(nn.Linear(cfg.pose_dim, cfg.cond_dim), nn.SiLU(), nn.Linear(cfg.cond_dim, cfg.cond_dim))
        self.h1 = H1(cfg.c1, cfg.cond_dim, cfg.head_hidden, cfg.depth_mult)
        self.h2 = H2(cfg.c1, cfg.head_hidden, cfg.depth_mult)
    def set_qat(self, e):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmbedding)): m.set_qat(e)
    def forward(self, mask, pose):
        coords = make_coords(mask.shape[0], MODEL_H, MODEL_W, mask.device, torch.float32)
        f = self.trunk(mask, coords)
        return self.h1(f, self.pose_mlp(pose)), self.h2(f)
    def param_count(self): return sum(p.numel() for p in self.parameters())
    def fp4_size_estimate(self):
        total_bits = 0
        for n, m in self.named_modules():
            if isinstance(m, (QConv2d, QEmbedding)):
                w = m.weight
                if getattr(m, 'quantize_weight', True):
                    total_bits += w.numel() * 4 + (w.numel() // 32) * 16
                else:
                    total_bits += w.numel() * 16
                if hasattr(m, 'bias') and m.bias is not None:
                    total_bits += m.bias.numel() * 16
            elif isinstance(m, nn.Linear):
                total_bits += m.weight.numel() * 16
                if m.bias is not None: total_bits += m.bias.numel() * 16
            elif isinstance(m, nn.GroupNorm):
                if m.weight is not None: total_bits += m.weight.numel() * 16
                if m.bias is not None: total_bits += m.bias.numel() * 16
        return total_bits // 8  # bytes before brotli; brotli typically ~0.7-0.85x

# ─── Data caching ────────────────────────────────────────────────────

_CACHE = {}

def get_cached_data(device, n_pairs, seed, mask_crf):
    key = f"{n_pairs}_{seed}_{mask_crf}"
    if key in _CACHE:
        return _CACHE[key]

    from frame_utils import AVVideoDataset
    cache_dir = Path(__file__).parent / "_probe_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"probe_data_n{n_pairs}_s{seed}.pt"

    if cache_file.exists():
        logging.info(f"Loading cached probe data from {cache_file}")
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        _CACHE[key] = data
        return data

    logging.info("Building probe data cache (first run only)...")
    files = [l.strip() for l in (ROOT / "public_test_video_names.txt").read_text().splitlines() if l.strip()]
    ds = AVVideoDataset(files, data_dir=ROOT / "videos", batch_size=16, device=torch.device("cpu"), num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds.prepare_data()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    all_rgb = []
    for _, _, batch in tqdm(dl, desc="Loading video"):
        all_rgb.append(batch.cpu())
    all_rgb = torch.cat(all_rgb, 0)

    g = torch.Generator(); g.manual_seed(seed)
    idx = torch.randperm(all_rgb.shape[0], generator=g)[:n_pairs]
    rgb_sub = all_rgb[idx].contiguous()

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))

    masks, poses = [], []
    with torch.inference_mode():
        for i in range(0, n_pairs, 8):
            b = rgb_sub[i:i+8].to(device).float()
            b_chw = einops.rearrange(b, 'b t h w c -> b t c h w')
            odd = b_chw[:, 1]
            resized = F.interpolate(odd, (MODEL_H, MODEL_W), mode='bilinear')
            m = segnet(resized).argmax(1).to(torch.uint8)
            masks.append(m.cpu())
            pin = posenet.preprocess_input(b_chw)
            p = get_pose6(posenet, pin).cpu()
            poses.append(p)

    masks_t = torch.cat(masks, 0).contiguous()
    poses_t = torch.cat(poses, 0).float().contiguous()

    gt_seg, gt_pose = [], []
    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    with torch.inference_mode():
        for i in range(0, n_pairs, 16):
            b = rgb_sub[i:i+16].to(device)
            po, so = dist_net(b)
            gt_seg.append(so.cpu())
            gt_pose.append({k: v.cpu() for k, v in po.items()})

    del segnet, posenet, dist_net
    torch.cuda.empty_cache()

    data = {
        "rgb": rgb_sub, "masks": masks_t, "poses": poses_t, "idx": idx,
        "gt_seg": torch.cat(gt_seg, 0),
        "gt_pose_keys": list(gt_pose[0].keys()),
        **{f"gt_pose_{k}": torch.cat([p[k] for p in gt_pose], 0) for k in gt_pose[0].keys()}
    }
    torch.save(data, cache_file)
    _CACHE[key] = data
    return data

# ─── Fast training loop ──────────────────────────────────────────────

def fast_train(cfg: ProbeConfig, device, data):
    rgb = data["rgb"]
    masks = data["masks"]
    poses = data["poses"]
    n = rgb.shape[0]

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for p in segnet.parameters(): p.requires_grad = False
    for p in posenet.parameters(): p.requires_grad = False

    gen = Gen(cfg).to(device)
    n_params = gen.param_count()
    model_bytes_est = int(gen.fp4_size_estimate() * 0.78)  # ~78% brotli ratio

    def shuffled_batches(epoch):
        g = torch.Generator(); g.manual_seed(cfg.seed + epoch)
        perm = torch.randperm(n, generator=g)
        for s in range(0, n, cfg.batch_size):
            idx = perm[s:s+cfg.batch_size]
            yield (rgb.index_select(0, idx).to(device, non_blocking=True),
                   masks.index_select(0, idx).to(device, non_blocking=True),
                   poses.index_select(0, idx).to(device, non_blocking=True))

    # Stage 1: Anchor (frame2 seg only)
    for p in gen.h1.parameters(): p.requires_grad = False
    for p in gen.pose_mlp.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()), lr=cfg.lr, betas=(0.9, 0.99))
    qat_start = int(cfg.anchor_epochs * cfg.qat_start_frac)

    for ep in range(cfg.anchor_epochs):
        gen.train()
        qat = ep >= qat_start
        gen.set_qat(qat)
        boost = cfg.err_boost if ep < cfg.anchor_epochs - 5 else cfg.err_boost_high
        kl_alpha = min(1.0, ep / max(1, qat_start // 2)) if qat_start > 0 else 1.0
        s2_kl_w = 0.9 - 0.9 * kl_alpha
        s2_ce_w = 0.1 + 0.9 * kl_alpha

        for batch_rgb, in_mask, in_pose in shuffled_batches(ep):
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float()
            with torch.no_grad():
                r2 = F.interpolate(batch[:, 1], (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
                gt_logits2 = segnet(r2).float()
                gt_mask2 = gt_logits2.argmax(1)
            opt.zero_grad(set_to_none=True)
            _, p2 = gen(in_mask.long(), in_pose.float())
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            fl2 = segnet(f2d).float()
            ce2 = F.cross_entropy(fl2, gt_mask2, reduction='none')
            with torch.no_grad():
                b2 = 1.0 + (fl2.argmax(1) != gt_mask2).float() * boost
            ce_loss = (ce2 * b2).mean()
            kl_loss = kl_on_logits(fl2, gt_logits2) / (MODEL_H * MODEL_W)
            loss = 100.0 * (s2_kl_w * kl_loss + s2_ce_w * 0.5 * cfg.ce_weight * ce_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
            opt.step()

    # Stage 2: Finetune (frame1 pose)
    for p in gen.parameters(): p.requires_grad = True
    for p in gen.trunk.parameters(): p.requires_grad = False
    for p in gen.h2.parameters(): p.requires_grad = False
    gen.trunk.eval(); gen.h2.eval()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()), lr=cfg.ft_lr, betas=(0.9, 0.99))
    gen.set_qat(True)

    for ep in range(cfg.finetune_epochs):
        gen.h1.train(); gen.pose_mlp.train()
        for batch_rgb, in_mask, in_pose in shuffled_batches(1000 + ep):
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float()
            with torch.no_grad():
                gt_pose = get_pose6(posenet, posenet.preprocess_input(batch)).float()
            opt.zero_grad(set_to_none=True)
            p1, p2 = gen(in_mask.long(), in_pose.float())
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f1d = F.interpolate(diff_round(f1u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            loss = cfg.pose_weight * 10.0 * F.mse_loss(fp, gt_pose)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
            opt.step()

    # Stage 3: Joint (both)
    for p in gen.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.jt_lr, betas=(0.9, 0.99))
    gen.set_qat(True)

    for ep in range(cfg.joint_epochs):
        gen.train()
        for batch_rgb, in_mask, in_pose in shuffled_batches(2000 + ep):
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float()
            with torch.no_grad():
                r2 = F.interpolate(batch[:, 1], (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
                gt_logits2 = segnet(r2).float()
                gt_mask2 = gt_logits2.argmax(1)
                gt_pose = get_pose6(posenet, posenet.preprocess_input(batch)).float()
            opt.zero_grad(set_to_none=True)
            p1, p2 = gen(in_mask.long(), in_pose.float())
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f1d = F.interpolate(diff_round(f1u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            fl2 = segnet(f2d).float()
            ce2 = F.cross_entropy(fl2, gt_mask2, reduction='none')
            with torch.no_grad():
                b2 = 1.0 + (fl2.argmax(1) != gt_mask2).float() * cfg.err_boost
            seg_loss = 100.0 * cfg.ce_weight * (ce2 * b2).mean()
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            pose_loss = 30.0 * cfg.pose_weight * F.mse_loss(fp, gt_pose)
            loss = seg_loss + pose_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
            opt.step()

    del segnet, posenet
    torch.cuda.empty_cache()
    return gen, n_params, model_bytes_est

# ─── Evaluation ───────────────────────────────────────────────────────

def evaluate_proxy(gen, cfg, device, data):
    gen.eval()
    gen.set_qat(True)  # evaluate with quantized weights

    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    rgb = data["rgb"]
    masks = data["masks"]
    poses = data["poses"]
    n = rgb.shape[0]

    t_seg, t_pose, t_n = 0., 0., 0
    with torch.inference_mode():
        for i in range(0, n, 16):
            m = masks[i:i+16].to(device).long()
            p = poses[i:i+16].to(device).float()
            p1, p2 = gen(m, p)
            comp = torch.stack([
                F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False),
                F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            ], dim=1)
            comp = einops.rearrange(comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
            gt_batch = rgb[i:i+16].to(device)
            pd, sd = dist_net.compute_distortion(gt_batch, comp)
            t_seg += sd.sum().item()
            t_pose += pd.sum().item()
            t_n += gt_batch.shape[0]

    del dist_net
    torch.cuda.empty_cache()

    avg_seg = t_seg / max(1, t_n)
    avg_pose = t_pose / max(1, t_n)

    mask_bytes = 219588  # from our CRF=50 run; TODO: make configurable
    pose_bytes = 13194
    model_bytes = int(gen.fp4_size_estimate() * 0.78)
    total_bytes = mask_bytes + pose_bytes + model_bytes
    rate = total_bytes / UNCOMPRESSED

    score = 100 * avg_seg + math.sqrt(max(0, 10 * avg_pose)) + 25 * rate
    return {
        "score": round(score, 4),
        "seg_term": round(100 * avg_seg, 4),
        "pose_term": round(math.sqrt(max(0, 10 * avg_pose)), 4),
        "rate_term": round(25 * rate, 4),
        "avg_seg": avg_seg,
        "avg_pose": avg_pose,
        "total_bytes": total_bytes,
        "model_bytes_est": model_bytes,
        "n_params": gen.param_count(),
    }

# ─── Sweep generation ────────────────────────────────────────────────

def generate_sweep_configs():
    configs = []

    # Axis 1: Architecture width
    for c1, c2, hh in [(40, 48, 36), (48, 56, 44), (56, 64, 52), (64, 72, 60), (72, 80, 68)]:
        configs.append({"c1": c1, "c2": c2, "head_hidden": hh, "label": f"width_{c1}_{c2}"})

    # Axis 2: Embedding dimension
    for ed in [4, 6, 8, 12]:
        configs.append({"emb_dim": ed, "label": f"emb_{ed}"})

    # Axis 3: Depth multiplier (controls depthwise expansion)
    for dm in [1, 2, 3]:
        configs.append({"depth_mult": dm, "label": f"dm_{dm}"})

    # Axis 4: Conditioning dimension
    for cd in [32, 48, 64, 96]:
        configs.append({"cond_dim": cd, "label": f"cond_{cd}"})

    # Axis 5: Error boost
    for eb, ebh in [(4, 16), (9, 49), (16, 64), (25, 100)]:
        configs.append({"err_boost": eb, "err_boost_high": ebh, "label": f"boost_{eb}_{ebh}"})

    # Axis 6: Learning rate
    for lr in [1e-4, 3e-4, 5e-4, 1e-3, 2e-3]:
        configs.append({"lr": lr, "label": f"lr_{lr}"})

    # Axis 7: QAT start fraction
    for qf in [0.3, 0.5, 0.7, 0.0]:
        configs.append({"qat_start_frac": qf, "label": f"qat_{qf}"})

    # Axis 8: Pose weight
    for pw in [0.5, 1.0, 2.0, 5.0]:
        configs.append({"pose_weight": pw, "label": f"pw_{pw}"})

    # Axis 9: Block size for quantization
    for bs in [16, 32, 64]:
        configs.append({"block_size": bs, "label": f"bs_{bs}"})

    # Promising combos
    configs.append({"c1": 48, "c2": 56, "head_hidden": 44, "depth_mult": 2, "label": "narrow_deep"})
    configs.append({"c1": 64, "c2": 72, "head_hidden": 60, "depth_mult": 1, "err_boost": 16, "label": "wide_highboost"})
    configs.append({"c1": 56, "c2": 64, "head_hidden": 52, "lr": 1e-3, "qat_start_frac": 0.3, "label": "fast_qat"})
    configs.append({"c1": 40, "c2": 48, "head_hidden": 36, "depth_mult": 2, "cond_dim": 64, "label": "tiny_rich"})

    return configs

# ─── Main ─────────────────────────────────────────────────────────────

def run_single(cfg: ProbeConfig, device, log_path: Path):
    t0 = time.time()
    data = get_cached_data(device, cfg.n_pairs, cfg.seed, cfg.mask_crf)
    gen, n_params, model_est = fast_train(cfg, device, data)
    result = evaluate_proxy(gen, cfg, device, data)
    elapsed = time.time() - t0

    entry = {
        "config": asdict(cfg),
        "result": result,
        "elapsed_sec": round(elapsed, 1),
    }
    if hasattr(cfg, '_label'):
        entry["label"] = cfg._label

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    del gen
    torch.cuda.empty_cache()
    return entry

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="JSON config override")
    parser.add_argument("--sweep", type=str, default=None, help="Path to sweep configs JSON")
    parser.add_argument("--gen-sweep", action="store_true", help="Generate default sweep file")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--anchor-epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--joint-epochs", type=int, default=15)
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = torch.device(args.device)

    probe_dir = Path(__file__).parent / "_probe_results"
    probe_dir.mkdir(exist_ok=True)
    log_path = Path(args.log) if args.log else probe_dir / f"sweep_{int(time.time())}.jsonl"

    if args.gen_sweep:
        configs = generate_sweep_configs()
        out = probe_dir / "sweep_configs.json"
        with open(out, "w") as f:
            json.dump(configs, f, indent=2)
        print(f"Generated {len(configs)} configs -> {out}")
        return

    if args.sweep:
        with open(args.sweep) as f:
            configs = json.load(f)
        logging.info(f"Running sweep: {len(configs)} configs -> {log_path}")
        for i, c in enumerate(configs):
            label = c.pop("label", f"config_{i}")
            cfg = ProbeConfig(
                n_pairs=args.n_pairs,
                anchor_epochs=args.anchor_epochs,
                finetune_epochs=args.finetune_epochs,
                joint_epochs=args.joint_epochs,
            )
            for k, v in c.items():
                if hasattr(cfg, k): setattr(cfg, k, v)
            cfg._label = label
            logging.info(f"\n{'='*60}\n[{i+1}/{len(configs)}] {label}\n{'='*60}")
            try:
                entry = run_single(cfg, device, log_path)
                logging.info(f"  Score={entry['result']['score']:.4f} "
                             f"seg={entry['result']['seg_term']:.4f} "
                             f"pose={entry['result']['pose_term']:.4f} "
                             f"rate={entry['result']['rate_term']:.4f} "
                             f"params={entry['result']['n_params']:,} "
                             f"time={entry['elapsed_sec']:.0f}s")
            except Exception as e:
                logging.error(f"  FAILED: {e}")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"label": label, "error": str(e)}) + "\n")
        logging.info(f"\nSweep complete. Results: {log_path}")
        return

    # Single config run
    cfg = ProbeConfig(
        n_pairs=args.n_pairs,
        anchor_epochs=args.anchor_epochs,
        finetune_epochs=args.finetune_epochs,
        joint_epochs=args.joint_epochs,
    )
    if args.config:
        overrides = json.loads(args.config)
        for k, v in overrides.items():
            if hasattr(cfg, k): setattr(cfg, k, v)
    cfg._label = "single"

    entry = run_single(cfg, device, log_path)
    print(f"\nScore={entry['result']['score']:.4f}")
    print(f"  seg={entry['result']['seg_term']:.4f}  pose={entry['result']['pose_term']:.4f}  rate={entry['result']['rate_term']:.4f}")
    print(f"  params={entry['result']['n_params']:,}  model_est={entry['result']['model_bytes_est']:,}B")
    print(f"  time={entry['elapsed_sec']:.0f}s")

if __name__ == "__main__":
    main()
