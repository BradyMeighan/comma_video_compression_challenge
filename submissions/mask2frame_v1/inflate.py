#!/usr/bin/env python
"""
Mask2Frame inflate: decode mask video + pose vectors through trained generator -> .raw
"""
import io, os, sys, tempfile
from pathlib import Path

import av
import brotli
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ─── FP4 Dequantization ──────────────────────────────────────────────

class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)

    @staticmethod
    def dequantize_from_nibbles(nibbles, scales, orig_shape):
        flat_n = int(torch.tensor(orig_shape).prod().item())
        block_size = nibbles.numel() // scales.numel()
        nibbles = nibbles.view(-1, block_size)
        signs = (nibbles >> 3).to(torch.int64)
        mag_idx = (nibbles & 0x7).to(torch.int64)
        levels = FP4Codebook.pos_levels.to(scales.device, torch.float32)
        q = torch.where(signs.bool(), -levels[mag_idx], levels[mag_idx])
        return (q * scales[:, None].float()).view(-1)[:flat_n].reshape(orig_shape)

def unpack_nibbles(packed, count):
    flat = packed.reshape(-1)
    out = torch.empty(flat.numel() * 2, dtype=torch.uint8, device=packed.device)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
    return out[:count]

def load_model_weights(data_bytes, device):
    data = torch.load(io.BytesIO(data_bytes), map_location=device, weights_only=False)
    sd = {}
    for name, rec in data["quantized"].items():
        if rec["weight_kind"] == "fp4_packed":
            nib = unpack_nibbles(rec["packed_weight"].to(device), rec["packed_weight"].numel() * 2)
            w = FP4Codebook.dequantize_from_nibbles(nib, rec["scales_fp16"].to(device), rec["weight_shape"])
        else:
            w = rec["weight_fp16"].to(device).float()
        sd[f"{name}.weight"] = w.float()
        if rec.get("bias_fp16") is not None:
            sd[f"{name}.bias"] = rec["bias_fp16"].to(device).float()
    for name, tensor in data.get("dense_fp16", {}).items():
        sd[name] = tensor.to(device).float() if torch.is_floating_point(tensor) else tensor.to(device)
    return sd

# ─── Architecture (inference only) ───────────────────────────────────

class QConv2d(nn.Conv2d):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw)

class QEmbedding(nn.Embedding):
    def __init__(self, *a, block_size=32, quantize_weight=True, **kw):
        super().__init__(*a, **kw)

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
    def forward(self, mask2, pose6):
        coords = make_coord_grid(mask2.shape[0], 384, 512, mask2.device, torch.float32)
        feat = self.trunk(mask2, coords)
        return self.head1(feat, self.pose_mlp(pose6)), self.head2(feat)

# ─── Mask video loading ──────────────────────────────────────────────

def load_mask_video(path):
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="gray")
        cls_img = np.clip(np.round(img / 63.0).astype(np.uint8), 0, 4)
        frames.append(cls_img)
    container.close()
    return torch.from_numpy(np.stack(frames)).contiguous()

# ─── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: python inflate.py <data_dir> <out_dir> <file_list>")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    file_list = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = [l.strip() for l in file_list.read_text().splitlines() if l.strip()]

    model_br = data_dir / "model.pt.br"
    mask_br = data_dir / "mask.obu.br"
    pose_br = data_dir / "pose.npy.br"

    gen = JointFrameGenerator().to(device)
    with open(model_br, "rb") as f:
        sd = load_model_weights(brotli.decompress(f.read()), device)
    gen.load_state_dict(sd, strict=True)
    gen.eval()

    with tempfile.NamedTemporaryFile(suffix=".obu", delete=False) as tmp:
        with open(mask_br, "rb") as f:
            tmp.write(brotli.decompress(f.read()))
        tmp_path = tmp.name
    masks_all = load_mask_video(tmp_path)
    os.remove(tmp_path)

    with open(pose_br, "rb") as f:
        poses_all = torch.from_numpy(np.load(io.BytesIO(brotli.decompress(f.read())))).float()

    out_h, out_w = 874, 1164
    pairs_per_file = 600
    cursor = 0
    batch_size = 4

    with torch.inference_mode():
        for fname in files:
            base = os.path.splitext(fname)[0]
            raw_path = out_dir / f"{base}.raw"
            file_masks = masks_all[cursor:cursor+pairs_per_file]
            file_poses = poses_all[cursor:cursor+pairs_per_file]
            cursor += pairs_per_file

            with open(raw_path, "wb") as f_out:
                for i in tqdm(range(0, file_masks.shape[0], batch_size), desc=f"Inflating {fname}"):
                    m = file_masks[i:i+batch_size].to(device).long()
                    p = file_poses[i:i+batch_size].to(device).float()
                    f1, f2 = gen(m, p)
                    f1_up = F.interpolate(f1, (out_h, out_w), mode="bilinear", align_corners=False)
                    f2_up = F.interpolate(f2, (out_h, out_w), mode="bilinear", align_corners=False)
                    frames = torch.stack([f1_up, f2_up], dim=1)
                    frames = einops.rearrange(frames, "b t c h w -> (b t) h w c")
                    f_out.write(frames.clamp(0, 255).round().to(torch.uint8).cpu().numpy().tobytes())

if __name__ == "__main__":
    main()
