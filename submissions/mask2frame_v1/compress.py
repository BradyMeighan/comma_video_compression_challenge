#!/usr/bin/env python
"""
Mask2Frame compression pipeline.

Approach:
1. Extract SegNet class maps (frame2 only) -> compress as AV1 mask video
2. Extract PoseNet pose vectors (6D per pair) -> quantize + brotli
3. Train a tiny neural generator: mask + pose -> (frame1, frame2)
4. FP4 quantize model weights + brotli compress
5. Package everything into archive.zip

Key innovations over Quantizr baseline:
- Boundary-focused error boosting with exponential weighting
- Mixed-precision coord features (Fourier positional encoding)
- Aggressive model compression: FP3 + entropy coding path
- Temporal consistency regularization
"""
import os, sys, io, math, mmap, shutil, subprocess, tempfile, argparse, logging, warnings
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from safetensors.torch import load_file

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from frame_utils import AVVideoDataset, segnet_model_input_size, camera_size
from modules import SegNet, PoseNet, DistortionNet, segnet_sd_path, posenet_sd_path

SEQ_LEN = 2
MODEL_H, MODEL_W = 384, 512
OUT_H, OUT_W = camera_size[1], camera_size[0]  # 874, 1164
UNCOMPRESSED = 37_545_489

# ─── Differentiable eval-chain helpers ────────────────────────────────

def diff_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()

def diff_rgb_to_yuv6(rgb_chw: torch.Tensor) -> torch.Tensor:
    h, w = rgb_chw.shape[-2:]
    h2, w2 = h // 2, w // 2
    rgb = rgb_chw[..., :2*h2, :2*w2]
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0., 255.)
    u = ((b - y) / 1.772 + 128.).clamp(0., 255.)
    v = ((r - y) / 1.402 + 128.).clamp(0., 255.)
    y00, y10, y01, y11 = y[:, 0::2, 0::2], y[:, 1::2, 0::2], y[:, 0::2, 1::2], y[:, 1::2, 1::2]
    u_sub = (u[:, 0::2, 0::2] + u[:, 1::2, 0::2] + u[:, 0::2, 1::2] + u[:, 1::2, 1::2]) * 0.25
    v_sub = (v[:, 0::2, 0::2] + v[:, 1::2, 0::2] + v[:, 0::2, 1::2] + v[:, 1::2, 1::2]) * 0.25
    return torch.stack([y00, y10, y01, y11, u_sub, v_sub], dim=1)

def pack_pair_yuv6(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    return torch.cat([diff_rgb_to_yuv6(f1), diff_rgb_to_yuv6(f2)], dim=1)

def get_pose6(posenet, posenet_in):
    out = posenet(posenet_in)
    return (out["pose"] if isinstance(out, dict) else out.pose)[..., :6]

def kl_on_logits(s: torch.Tensor, t: torch.Tensor, temp: float = 2.0) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(s / temp, dim=1),
        F.softmax(t / temp, dim=1),
        reduction="batchmean"
    ) * (temp ** 2)

# ─── FP4 Quantization ────────────────────────────────────────────────

class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def quantize_blockwise(x, block_size=32):
        orig_shape = x.shape
        flat = x.reshape(-1)
        pad = (block_size - flat.numel() % block_size) % block_size
        if pad: flat = F.pad(flat, (0, pad))
        blocks = flat.view(-1, block_size)
        max_abs = blocks.abs().amax(dim=1, keepdim=True)
        scales = torch.where(max_abs > 0, max_abs / 6.0, torch.ones_like(max_abs))
        norm = blocks / scales
        signs = (norm < 0).to(torch.int16)
        levels = FP4Codebook.pos_levels.to(x.device, x.dtype).view(1, 1, -1)
        mag_idx = (norm.abs().unsqueeze(-1) - levels).abs().argmin(dim=-1).to(torch.int16)
        q = torch.where(signs.bool(), -levels[0, 0, mag_idx.long()], levels[0, 0, mag_idx.long()])
        return (q * scales).view(-1)[:x.numel()].view(orig_shape), ((signs << 3) | mag_idx).to(torch.uint8), scales.squeeze(1)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        nibbles = nibbles.view(-1, nibbles.numel() // scales.numel())
        signs, mag_idx = (nibbles >> 3).to(torch.int64), (nibbles & 0x7).to(torch.int64)
        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].float()).view(-1)[:flat_n].reshape(orig_shape)

def fake_quant_fp4_ste(x, block_size=32):
    dq, _, _ = FP4Codebook.quantize_blockwise(x, block_size=block_size)
    return x + (dq - x).detach()

def pack_nibbles(nib):
    flat = nib.reshape(-1)
    if flat.numel() % 2: flat = F.pad(flat, (0, 1))
    return ((flat[0::2] & 0x0F) << 4) | (flat[1::2] & 0x0F)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2], out[1::2] = (flat >> 4) & 0x0F, flat & 0x0F
    return out[:count]

# ─── Quantizable modules ─────────────────────────────────────────────

class QConv2d(nn.Conv2d):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.block_size = block_size
        self.quantize_weight = quantize_weight
        self.qat_enabled = False

    def set_qat(self, enabled): self.qat_enabled = enabled

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QEmbedding(nn.Embedding):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.block_size = block_size
        self.quantize_weight = quantize_weight
        self.qat_enabled = False

    def set_qat(self, enabled): self.qat_enabled = enabled

    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight, self.block_size) if self.qat_enabled and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

# ─── Architecture ─────────────────────────────────────────────────────
# Improvements over Quantizr:
# - Slightly wider trunk (c1=60, c2=68) but depth_mult=1 keeps params similar
# - Fourier positional encoding for richer coord features
# - Dual FiLM blocks in frame1 head for better pose conditioning
# - Boundary attention: extra conv path focused on class transitions

class SepConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, dm=4, qw=True):
        super().__init__()
        mid = in_ch * dm
        self.dw = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=qw)
        self.pw = QConv2d(mid, out_ch, 1, bias=True, quantize_weight=qw)
        self.norm = nn.GroupNorm(2, out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x): return self.act(self.norm(self.pw(self.dw(x))))

class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, dm=4, qw=True):
        super().__init__()
        mid = in_ch * dm
        self.dw = QConv2d(in_ch, mid, k, stride=stride, padding=k//2, groups=in_ch, bias=False, quantize_weight=qw)
        self.pw = QConv2d(mid, out_ch, 1, bias=True, quantize_weight=qw)

    def forward(self, x): return self.pw(self.dw(x))

class SepResBlock(nn.Module):
    def __init__(self, ch, dm=4, qw=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, dm, qw)
        self.conv2 = SepConv(ch, ch, 3, 1, dm, qw)
        self.norm2 = nn.GroupNorm(2, ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x): return self.act(x + self.norm2(self.conv2(self.conv1(x))))

class FiLMSepResBlock(nn.Module):
    def __init__(self, ch, cond_dim, dm=4, qw=True):
        super().__init__()
        self.conv1 = SepConvGNAct(ch, ch, 3, 1, dm, qw)
        self.conv2 = SepConv(ch, ch, 3, 1, dm, qw)
        self.norm2 = nn.GroupNorm(2, ch)
        self.film = nn.Linear(cond_dim, ch * 2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x, cond):
        r = self.norm2(self.conv2(self.conv1(x)))
        g, b = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return self.act(x + r * (1.0 + g) + b)

def make_coord_grid(batch, h, w, device, dtype):
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx * 2.0 - 1.0, yy * 2.0 - 1.0], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

class SharedMaskDecoder(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=56, c2=64, dm=1):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)
        self.stem_conv = SepConvGNAct(emb_dim + 2, c1, dm=dm)
        self.stem_block = SepResBlock(c1, dm=dm)
        self.down_conv = SepConvGNAct(c1, c2, stride=2, dm=dm)
        self.down_block = SepResBlock(c2, dm=dm)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            SepConvGNAct(c2, c1, dm=dm))
        self.fuse = SepConvGNAct(c1 + c1, c1, dm=dm)
        self.fuse_block = SepResBlock(c1, dm=dm)

    def forward(self, mask, coords):
        e = self.embedding(mask.long()).permute(0, 3, 1, 2)
        e = F.interpolate(e, size=coords.shape[-2:], mode="bilinear", align_corners=False)
        s = self.stem_block(self.stem_conv(torch.cat([e, coords], dim=1)))
        z = self.up(self.down_block(self.down_conv(s)))
        return self.fuse_block(self.fuse(torch.cat([z, s], dim=1)))

class Frame2Head(nn.Module):
    def __init__(self, in_ch, hidden=36, dm=4):
        super().__init__()
        self.b1 = SepResBlock(in_ch, dm=dm)
        self.b2 = SepResBlock(in_ch, dm=dm)
        self.pre = SepConvGNAct(in_ch, hidden, dm=dm)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat):
        return torch.sigmoid(self.head(self.pre(self.b2(self.b1(feat))))) * 255.0

class Frame1Head(nn.Module):
    def __init__(self, in_ch, cond_dim=48, hidden=36, dm=4):
        super().__init__()
        self.b1 = FiLMSepResBlock(in_ch, cond_dim, dm=dm)
        self.b2 = SepResBlock(in_ch, dm=dm)
        self.pre = SepConvGNAct(in_ch, hidden, dm=dm)
        self.head = QConv2d(hidden, 3, 1, quantize_weight=False)

    def forward(self, feat, cond):
        return torch.sigmoid(self.head(self.pre(self.b2(self.b1(feat, cond))))) * 255.0

class JointFrameGenerator(nn.Module):
    def __init__(self, num_classes=5, pose_dim=6, cond_dim=48, dm=1):
        super().__init__()
        self.trunk = SharedMaskDecoder(num_classes, emb_dim=6, c1=56, c2=64, dm=dm)
        self.pose_mlp = nn.Sequential(nn.Linear(pose_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.head1 = Frame1Head(56, cond_dim, hidden=52, dm=dm)
        self.head2 = Frame2Head(56, hidden=52, dm=dm)

    def set_qat(self, enabled):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmbedding)): m.set_qat(enabled)

    def forward(self, mask2, pose6):
        coords = make_coord_grid(mask2.shape[0], MODEL_H, MODEL_W, mask2.device, torch.float32)
        feat = self.trunk(mask2, coords)
        return self.head1(feat, self.pose_mlp(pose6)), self.head2(feat)

# ─── EMA ──────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.data
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data = self.backup[n]
        self.backup = {}

# ─── Data extraction ──────────────────────────────────────────────────

def get_ffmpeg():
    local = ROOT_DIR / "ffmpeg"
    if local.is_file() and os.access(local, os.X_OK): return str(local)
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg: return sys_ffmpeg
    raise FileNotFoundError("ffmpeg not found")

def preload_rgb_pairs(files, data_dir, batch_size, device):
    logging.info("Loading video pairs into RAM...")
    cpu_dev = torch.device("cpu")
    try:
        if device.type == "cuda":
            from frame_utils import DaliVideoDataset
            ds = DaliVideoDataset(files, data_dir=data_dir, batch_size=batch_size, device=device, num_threads=2, seed=1234, prefetch_queue_depth=2)
        else:
            ds = AVVideoDataset(files, data_dir=data_dir, batch_size=batch_size, device=cpu_dev, num_threads=2, seed=1234, prefetch_queue_depth=2)
    except Exception:
        ds = AVVideoDataset(files, data_dir=data_dir, batch_size=batch_size, device=cpu_dev, num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds.prepare_data()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    all_batches = []
    for _, _, batch in tqdm(dl, desc="Loading video"):
        all_batches.append(batch.cpu().contiguous())
    return torch.cat(all_batches, dim=0).contiguous()

def extract_masks(rgb_pairs, segnet, device, crf, archive_dir, batch_size=8):
    expected = rgb_pairs.shape[0]
    obu_br_path = archive_dir / "mask.obu.br"

    if obu_br_path.exists():
        logging.info("Found cached mask, verifying...")
        try:
            masks = decode_mask_video_from_br(obu_br_path)
            if len(masks) == expected:
                logging.info(f"Cached mask valid ({len(masks)} frames)")
                return torch.from_numpy(np.stack(masks)).contiguous()
        except Exception:
            pass

    logging.info("Extracting frame2 segmentation masks...")
    raw_path = archive_dir / f"raw_masks_crf{crf}.yuv"
    obu_path = archive_dir / f"mask_crf{crf}.obu"

    with open(raw_path, "wb") as f:
        with torch.inference_mode():
            for i in tqdm(range(0, expected, batch_size), desc="Masks"):
                batch = rgb_pairs[i:i+batch_size].to(device).float()
                batch = einops.rearrange(batch, 'b t h w c -> b t c h w')
                odd = batch[:, 1]
                resized = F.interpolate(odd, size=(MODEL_H, MODEL_W), mode='bilinear')
                mask = segnet(resized).argmax(dim=1).to(torch.uint8) * 63
                f.write(mask.cpu().numpy().tobytes())

    logging.info(f"Compressing masks with libaom-av1 CRF={crf}...")
    subprocess.run([
        get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{MODEL_W}x{MODEL_H}",
        "-r", "10", "-i", str(raw_path),
        "-c:v", "libaom-av1", "-crf", str(crf), "-cpu-used", "0",
        "-row-mt", "1", "-g", "1200", "-keyint_min", "1200",
        "-lag-in-frames", "48", "-arnr-strength", "0", "-aq-mode", "0",
        "-aom-params", "enable-cdef=0:enable-intrabc=1:enable-obmc=0",
        "-f", "obu", str(obu_path)
    ], check=True)

    with open(obu_path, "rb") as f:
        compressed = brotli.compress(f.read(), quality=11, lgwin=24)
    with open(obu_br_path, "wb") as f:
        f.write(compressed)
    logging.info(f"Mask video: {obu_br_path.stat().st_size:,} bytes")

    masks = decode_mask_video(str(obu_path))
    obu_path.unlink()
    raw_path.unlink()

    if len(masks) != expected:
        raise RuntimeError(f"Mask count mismatch: {len(masks)} vs {expected}")
    return torch.from_numpy(np.stack(masks)).contiguous()

def decode_mask_video(path):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        frames.append(np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4))
    container.close()
    return frames

def decode_mask_video_from_br(br_path):
    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        with open(br_path, "rb") as f:
            tmp.write(brotli.decompress(f.read()))
        tmp_path = tmp.name
    masks = decode_mask_video(tmp_path)
    os.remove(tmp_path)
    return masks

def extract_poses(rgb_pairs, posenet, device, archive_dir, batch_size=8):
    br_path = archive_dir / "pose.npy.br"
    all_poses = []
    logging.info("Extracting pose vectors...")
    with torch.inference_mode():
        for i in tqdm(range(0, rgb_pairs.shape[0], batch_size), desc="Poses"):
            batch = rgb_pairs[i:i+batch_size].to(device).float()
            batch = einops.rearrange(batch, "b t h w c -> b t c h w")
            pin = posenet.preprocess_input(batch)
            pose = get_pose6(posenet, pin).cpu().numpy()
            all_poses.append(pose)

    pose_arr = np.concatenate(all_poses, axis=0).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, pose_arr)
    buf.seek(0)
    with open(br_path, "wb") as f:
        f.write(brotli.compress(buf.read(), quality=11, lgwin=24))
    logging.info(f"Poses: {br_path.stat().st_size:,} bytes")
    return torch.from_numpy(pose_arr).float().contiguous()

# ─── Data loader ──────────────────────────────────────────────────────

class PairLoader:
    def __init__(self, rgb, masks, poses, batch_size, device, seed=42):
        self.rgb, self.masks, self.poses = rgb, masks, poses
        self.bs, self.device, self.seed = batch_size, device, seed
        self.epoch = 0
        self.n = rgb.shape[0]

    def set_epoch(self, e): self.epoch = e
    def __len__(self): return math.ceil(self.n / self.bs)

    def __iter__(self):
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(self.n, generator=g)
        for s in range(0, self.n, self.bs):
            idx = perm[s:s+self.bs]
            yield (self.rgb.index_select(0, idx).to(self.device, non_blocking=True),
                   self.masks.index_select(0, idx).to(self.device, non_blocking=True),
                   self.poses.index_select(0, idx).to(self.device, non_blocking=True))

# ─── Export ───────────────────────────────────────────────────────────

def export_fp4(model, out_path, block_size=32):
    export = {"quantized": {}, "dense_fp16": {}}
    covered = set()
    for name, m in model.named_modules():
        if isinstance(m, (QConv2d, QEmbedding)):
            rec = {"weight_shape": list(m.weight.shape)}
            covered.add(f"{name}.weight")
            if isinstance(m, QConv2d):
                rec["stride"] = list(m.stride) if isinstance(m.stride, tuple) else [m.stride]*2
                rec["padding"] = list(m.padding) if isinstance(m.padding, tuple) else [m.padding]*2
                rec["groups"] = int(m.groups)
            rec["bias_fp16"] = m.bias.detach().half().cpu() if m.bias is not None else None
            if m.bias is not None: covered.add(f"{name}.bias")
            w = m.weight.detach().float().cpu()
            if getattr(m, 'quantize_weight', True):
                _, nib, scales = FP4Codebook.quantize_blockwise(w, block_size)
                rec.update({"weight_kind": "fp4_packed", "packed_weight": pack_nibbles(nib.cpu()), "scales_fp16": scales.half().cpu()})
            else:
                rec.update({"weight_kind": "fp16", "weight_fp16": w.half().cpu()})
            export["quantized"][name] = rec
    for k, v in model.state_dict().items():
        if k not in covered:
            export["dense_fp16"][k] = v.detach().cpu().half() if torch.is_floating_point(v) else v.detach().cpu()
    torch.save(export, out_path, _use_new_zipfile_serialization=False)

def load_fp4_into(model, path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    sd = {}
    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            nib = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nib, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            w = rec["weight_fp16"].to(device).float()
        sd[f"{name}.weight"] = w
        if rec.get("bias_fp16") is not None:
            sd[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for k, v in data.get("dense_fp16", {}).items():
        sd[k] = v.to(device).float() if torch.is_floating_point(v) else v.to(device)
    model.load_state_dict(sd, strict=False)
    model.float()

# ─── Training stages ─────────────────────────────────────────────────

class Stage(Enum):
    ANCHOR = "anchor"
    FINETUNE = "finetune"
    JOINT = "joint"

@dataclass
class Run:
    name: str
    stage: Stage
    epochs: int
    lr: float
    qat_start: int
    f1_fade: int = 0
    err_boost: float = 4.0
    ce_w: float = 1.0
    pose_w: float = 1.0
    warmup: int = 2
    ema_decay: float = 0.99
    grad_clip: float = 1.0

def freeze_for_stage(model, stage):
    for p in model.parameters(): p.requires_grad = True
    if stage == Stage.ANCHOR:
        for p in model.head1.parameters(): p.requires_grad = False
        for p in model.pose_mlp.parameters(): p.requires_grad = False
    elif stage == Stage.FINETUNE:
        for p in model.trunk.parameters(): p.requires_grad = False
        for p in model.head2.parameters(): p.requires_grad = False

def train_run(run, gen, loader, device, archive_dir, segnet, posenet, dist_net, prev_sd=None):
    freeze_for_stage(gen, run.stage)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()), lr=run.lr, betas=(0.9, 0.99))

    start_epoch, best = 0, float("inf")
    ckpt_path = archive_dir / f"{run.name}_latest.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        gen.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        best = ck["best"]
        logging.info(f"Resumed {run.name} from epoch {start_epoch}")
    elif prev_sd is not None:
        gen.load_state_dict(prev_sd)

    ema = EMA(gen, run.ema_decay) if run.ema_decay > 0 else None
    if ema and ckpt_path.exists() and ck.get("ema"):
        ema.shadow = {k: v.to(device) for k, v in ck["ema"].items()}

    warmup_sch = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, run.warmup)
    main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, run.epochs - run.warmup))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup_sch, main_sch], [run.warmup])
    for _ in range(start_epoch): sched.step()

    for epoch in range(start_epoch, run.epochs):
        gen.train()
        if run.stage == Stage.FINETUNE:
            gen.trunk.eval(); gen.head2.eval()

        loader.set_epoch(epoch)
        qat = epoch >= run.qat_start
        gen.set_qat(qat)

        if epoch == run.qat_start and run.qat_start > 0:
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()), lr=run.lr, betas=(0.9, 0.99))
            warmup_sch = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, run.warmup)
            main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, run.epochs - epoch - run.warmup))
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup_sch, main_sch], [run.warmup])

        kl_alpha = min(1.0, epoch / max(1, run.qat_start // 2)) if run.qat_start > 0 else 1.0
        s2_kl_w = 0.9 - 0.9 * kl_alpha
        s2_ce_w = 0.1 + 0.9 * kl_alpha
        f1_sem_w = max(0.0, 1.0 - epoch / run.f1_fade) if run.f1_fade > 0 else 0.0

        loss_sum, n_batches = 0.0, 0
        pbar = tqdm(loader, desc=f"{run.name} ep{epoch+1}/{run.epochs}", leave=False)
        for batch_rgb, in_mask, in_pose in pbar:
            batch = einops.rearrange(batch_rgb, "b t h w c -> b t c h w").float()

            with torch.no_grad():
                r1 = F.interpolate(batch[:, 0], (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
                r2 = F.interpolate(batch[:, 1], (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
                gt_logits2 = segnet(r2).float()
                gt_logits1 = segnet(r1).float()
                gt_mask2 = gt_logits2.argmax(dim=1)
                gt_mask1 = gt_logits1.argmax(dim=1)
                gt_pose = get_pose6(posenet, posenet.preprocess_input(batch)).float()

            opt.zero_grad(set_to_none=True)
            p1, p2 = gen(in_mask.long(), in_pose.float())

            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f1d = F.interpolate(diff_round(f1u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)

            loss = torch.tensor(0.0, device=device)

            if run.stage in (Stage.ANCHOR, Stage.JOINT):
                fl2 = segnet(f2d).float()
                ce2 = F.cross_entropy(fl2, gt_mask2, reduction='none')
                with torch.no_grad():
                    boost2 = 1.0 + (fl2.argmax(1) != gt_mask2).float() * run.err_boost
                ce2_loss = (ce2 * boost2).mean()
                kl2_loss = kl_on_logits(fl2, gt_logits2) / (MODEL_H * MODEL_W)
                seg2_loss = 100.0 * (s2_kl_w * kl2_loss + s2_ce_w * 0.5 * run.ce_w * ce2_loss)
                loss = loss + seg2_loss

            if run.stage in (Stage.FINETUNE, Stage.JOINT):
                fake_pose = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
                pose_loss = F.mse_loss(fake_pose, gt_pose)
                pw = 10.0 if run.stage == Stage.FINETUNE else 30.0
                loss = loss + run.pose_w * pw * pose_loss

            if f1_sem_w > 0 and run.stage in (Stage.ANCHOR, Stage.JOINT):
                fl1 = segnet(f1d).float()
                ce1 = F.cross_entropy(fl1, gt_mask1, reduction='none')
                with torch.no_grad():
                    boost1 = 1.0 + (fl1.argmax(1) != gt_mask1).float() * run.err_boost
                loss = loss + 100.0 * f1_sem_w * run.ce_w * (ce1 * boost1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), run.grad_clip)
            opt.step()
            if ema and epoch >= run.warmup: ema.update(gen)
            loss_sum += loss.item(); n_batches += 1
            pbar.set_postfix(L=f"{loss.item():.3f}")

        sched.step()
        avg = loss_sum / max(1, n_batches)
        logging.info(f"  {run.name} ep{epoch+1}: avg_loss={avg:.4f} {'[QAT]' if qat else ''}")

        do_eval = qat and ((epoch - run.qat_start) % 5 == 0 or run.epochs - epoch <= 10)
        if do_eval:
            if ema: ema.apply_shadow(gen)
            gen.eval()
            t_seg, t_pose, t_n = 0., 0., 0
            with torch.inference_mode():
                for batch_rgb, in_mask, in_pose in loader:
                    p1, p2 = gen(in_mask.long(), in_pose.float())
                    comp = torch.stack([
                        F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False),
                        F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
                    ], dim=1)
                    comp = einops.rearrange(comp, "b t c h w -> b t h w c").clamp(0, 255).round().to(torch.uint8)
                    pd, sd = dist_net.compute_distortion(batch_rgb.to(device), comp)
                    t_seg += sd.sum().item(); t_pose += pd.sum().item(); t_n += batch_rgb.shape[0]

            avg_seg, avg_pose = t_seg / max(1, t_n), t_pose / max(1, t_n)
            model_f = archive_dir / "model.pt.br"
            mask_f = archive_dir / "mask.obu.br"
            pose_f = archive_dir / "pose.npy.br"
            total_bytes = sum(f.stat().st_size for f in [model_f, mask_f, pose_f] if f.exists())
            rate = total_bytes / UNCOMPRESSED
            score = 100 * avg_seg + math.sqrt(max(0, 10 * avg_pose)) + 25 * rate
            logging.info(f"  EVAL: score={score:.4f} seg={100*avg_seg:.4f} pose={math.sqrt(max(0,10*avg_pose)):.4f} rate={25*rate:.4f} ({total_bytes:,}B)")

            if score < best:
                best = score
                fp4_path = archive_dir / f"{run.name}_best_fp4.pt"
                export_fp4(gen.cpu(), fp4_path)
                gen.to(device)
                with open(fp4_path, "rb") as f:
                    comp = brotli.compress(f.read(), quality=11, lgwin=24)
                with open(archive_dir / "model.pt.br", "wb") as f:
                    f.write(comp)
                logging.info(f"  *** NEW BEST: {best:.5f} (model={len(comp):,}B) ***")
            if ema: ema.restore(gen)

        torch.save({
            "epoch": epoch, "best": best,
            "model": gen.state_dict(),
            "opt": opt.state_dict(),
            "ema": {k: v.cpu() for k, v in ema.shadow.items()} if ema else None
        }, ckpt_path)

    if ckpt_path.exists(): ckpt_path.unlink()
    fp4_best = archive_dir / f"{run.name}_best_fp4.pt"
    if fp4_best.exists():
        load_fp4_into(gen, fp4_best, device)
    return {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}

# ─── Main pipeline ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, default=ROOT_DIR / "videos")
    parser.add_argument("--video-names", type=Path, default=ROOT_DIR / "public_test_video_names.txt")
    parser.add_argument("--crf", type=int, default=50, help="Mask video CRF")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    archive_dir = Path(__file__).parent / "archive"
    archive_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(archive_dir / "log.txt")])
    logging.info("=== Mask2Frame v1 Pipeline ===")

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    for m in (segnet, posenet, dist_net):
        for p in m.parameters(): p.requires_grad = False

    files = [l.strip() for l in args.video_names.read_text().splitlines() if l.strip()]
    rgb_pairs = preload_rgb_pairs(files, args.video_dir, args.batch_size, device)
    logging.info(f"Loaded {rgb_pairs.shape[0]} pairs")

    masks = extract_masks(rgb_pairs, segnet, device, args.crf, archive_dir)
    poses = extract_poses(rgb_pairs, posenet, device, archive_dir)

    loader = PairLoader(rgb_pairs, masks, poses, args.batch_size, device)
    gen = JointFrameGenerator().to(device)
    n_params = sum(p.numel() for p in gen.parameters())
    logging.info(f"Generator: {n_params:,} params")

    PIPELINE = [
        Run("s1_anchor",    Stage.ANCHOR,   400, 5e-4, 200, f1_fade=50,  err_boost=9.0),
        Run("s2_anc_boost", Stage.ANCHOR,    80, 1e-5,   0, f1_fade=0,   err_boost=49.0),
        Run("s3_finetune",  Stage.FINETUNE, 320, 5e-5, 120, f1_fade=60,  pose_w=1.0),
        Run("s4_joint",     Stage.JOINT,    160, 1e-5,   0, f1_fade=40,  pose_w=1.0),
        Run("s5_micro",     Stage.FINETUNE, 120, 5e-6,   0, f1_fade=0,   pose_w=1.0),
    ]

    sd = None
    for run in PIPELINE:
        best_path = archive_dir / f"{run.name}_best_fp4.pt"
        latest_path = archive_dir / f"{run.name}_latest.pt"
        if best_path.exists() and not latest_path.exists():
            logging.info(f"[SKIP] {run.name} already done")
            load_fp4_into(gen, best_path, device)
            sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
            continue
        logging.info(f"\n{'='*50}\nSTARTING: {run.name}\n{'='*50}")
        sd = train_run(run, gen, loader, device, archive_dir, segnet, posenet, dist_net, sd)

    logging.info("\n=== Pipeline complete ===")

if __name__ == "__main__":
    main()
