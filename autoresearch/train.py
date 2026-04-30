#!/usr/bin/env python
"""
train.py — Model + training loop. THIS IS THE FILE THE AGENT EDITS.

Everything is fair game: architecture, hyperparameters, optimizer, loss,
training stages, quantization strategy. The only constraint is that it
runs within the 5-minute time budget and prints the parseable output.

Run: python train.py
"""
import sys, math, time, gc
from pathlib import Path

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from prepare import (
    load_data, evaluate, load_segnet, load_posenet, gpu_cleanup,
    diff_round, diff_rgb_to_yuv6, pack_pair_yuv6, get_pose6, kl_on_logits,
    fake_quant_fp4_ste,
    MODEL_H, MODEL_W, OUT_H, OUT_W, TRAIN_BUDGET_SEC,
)

# ══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS — tune these
# ══════════════════════════════════════════════════════════════════════

BATCH_SIZE    = 4
LR            = 5e-4       # anchor stage learning rate
FT_LR         = 5e-4       # finetune stage learning rate
JT_LR         = 5e-5       # joint stage learning rate
ERR_BOOST     = 9.0        # error boosting multiplier (normal)
ERR_BOOST_HI  = 49.0       # error boosting multiplier (late anchor)
GRAD_CLIP     = 0.5
QAT_FRAC      = 0.7        # fraction of anchor stage before enabling QAT

# Time allocation (fraction of TRAIN_BUDGET_SEC)
T_ANCHOR      = 0.55       # 55% for anchor (frame2 seg)
T_FINETUNE    = 0.27       # 27% for finetune (frame1 pose)
T_JOINT       = 0.13       # 13% for joint (both)
# remaining 5% is eval overhead

# Architecture
C1            = 56         # stem / output width
C2            = 64         # bottleneck width
EMB_DIM       = 6          # mask class embedding dim
COND_DIM      = 64         # pose conditioning dim
HEAD_HIDDEN   = 52         # head pre-output hidden channels
DM            = 1          # depthwise expansion multiplier

# ══════════════════════════════════════════════════════════════════════
# QUANTIZABLE LAYERS
# ══════════════════════════════════════════════════════════════════════

class QConv2d(nn.Conv2d):
    def __init__(self, *a, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.quantize_weight = quantize_weight
        self.qat = False
    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight) if self.qat and self.quantize_weight else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

class QEmb(nn.Embedding):
    def __init__(self, *a, quantize_weight=True, **kw):
        super().__init__(*a, **kw)
        self.quantize_weight = quantize_weight
        self.qat = False
    def forward(self, x):
        w = fake_quant_fp4_ste(self.weight) if self.qat and self.quantize_weight else self.weight
        return F.embedding(x, w, self.padding_idx)

class QLinear(nn.Module):
    """Linear via internal 1x1 QConv2d → gets FP4 byte treatment instead of FP16."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.conv = QConv2d(in_features, out_features, 1, bias=bias)
    @property
    def weight(self):
        return self.conv.weight
    @property
    def bias(self):
        return self.conv.bias
    def forward(self, x):
        orig = x.shape
        x = x.reshape(-1, orig[-1], 1, 1)
        x = self.conv(x)
        return x.view(*orig[:-1], -1)

# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

class DSConv(nn.Module):
    """Depthwise-separable conv + GroupNorm + SiLU."""
    def __init__(self, ic, oc, k=3, s=1, act=True):
        super().__init__()
        mid = ic * DM
        self.dw = QConv2d(ic, mid, k, stride=s, padding=k//2, groups=ic, bias=False)
        self.pw = QConv2d(mid, oc, 1, bias=True)
        self.norm = nn.GroupNorm(min(2, oc), oc)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))

class Res(nn.Module):
    """Pre-act residual with depthwise-separable convs."""
    def __init__(self, ch):
        super().__init__()
        self.c1 = DSConv(ch, ch)
        mid = ch * DM
        self.dw2 = QConv2d(ch, mid, 3, padding=1, groups=ch, bias=False)
        self.pw2 = QConv2d(mid, ch, 1, bias=True)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.norm(self.pw2(self.dw2(self.c1(x)))))

class FiLMRes(nn.Module):
    """Residual block with FiLM conditioning. FiLM is FP4-quantized + zero-init."""
    def __init__(self, ch, cd):
        super().__init__()
        self.c1 = DSConv(ch, ch)
        mid = ch * DM
        self.dw2 = QConv2d(ch, mid, 3, padding=1, groups=ch, bias=False)
        self.pw2 = QConv2d(mid, ch, 1, bias=True)
        self.norm = nn.GroupNorm(min(2, ch), ch)
        self.film = QLinear(cd, ch * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x, cond):
        r = self.norm(self.pw2(self.dw2(self.c1(x))))
        g, b = self.film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        return self.act(x + r * (1 + g) + b)

def coords(B, H, W, dev):
    ys = (torch.arange(H, device=dev, dtype=torch.float32) + 0.5) / H
    xs = (torch.arange(W, device=dev, dtype=torch.float32) + 0.5) / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx*2-1, yy*2-1], 0).unsqueeze(0).expand(B, -1, -1, -1)

class Trunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = QEmb(5, EMB_DIM, quantize_weight=False)
        self.stem = DSConv(EMB_DIM + 2, C1)
        self.s1 = Res(C1)
        self.down = DSConv(C1, C2, s=2)
        self.d1 = Res(C2)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            DSConv(C2, C1),
        )
        self.fuse = DSConv(C1 * 2, C1)
        self.f1 = Res(C1)
    def forward(self, mask, co):
        e = F.interpolate(self.emb(mask.long()).permute(0,3,1,2), co.shape[-2:], mode="bilinear", align_corners=False)
        s = self.s1(self.stem(torch.cat([e, co], 1)))
        z = self.up(self.d1(self.down(s)))
        return self.f1(self.fuse(torch.cat([z, s], 1)))

class Head2(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = Res(C1)
        self.r2 = Res(C1)
        self.pre = DSConv(C1, HEAD_HIDDEN)
        self.out = QConv2d(HEAD_HIDDEN, 3, 1, quantize_weight=False)
    def forward(self, f):
        return torch.sigmoid(self.out(self.pre(self.r2(self.r1(f))))) * 255.0

class Head1(nn.Module):
    def __init__(self):
        super().__init__()
        self.merge = DSConv(C1 + COND_DIM, C1)
        self.r1 = FiLMRes(C1, COND_DIM)
        self.r2 = Res(C1)
        self.pre = DSConv(C1, HEAD_HIDDEN)
        self.out = QConv2d(HEAD_HIDDEN, 3, 1, quantize_weight=False)
    def forward(self, f, c):
        cb = c.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *f.shape[-2:])
        m = self.merge(torch.cat([f, cb], 1))
        return torch.sigmoid(self.out(self.pre(self.r2(self.r1(m, c))))) * 255.0

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.pose_mlp = nn.Sequential(
            nn.Linear(6, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM),
        )
        # FiLM that modulates trunk features for h1 only (FP4-quantized via QLinear; init zeros)
        self.trunk_film = QLinear(COND_DIM, C1 * 2)
        nn.init.zeros_(self.trunk_film.weight)
        nn.init.zeros_(self.trunk_film.bias)
        self.h1 = Head1()
        self.h2 = Head2()

    def set_qat(self, on):
        for m in self.modules():
            if isinstance(m, (QConv2d, QEmb)):
                m.qat = on

    def forward(self, mask, pose):
        co = coords(mask.shape[0], MODEL_H, MODEL_W, mask.device)
        feat = self.trunk(mask, co)
        cond = self.pose_mlp(pose)
        g, b = self.trunk_film(cond).unsqueeze(-1).unsqueeze(-1).chunk(2, 1)
        feat_h1 = feat * (1 + g) + b
        return self.h1(feat_h1, cond), self.h2(feat)

# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def make_batches(rgb, masks, poses, epoch, device):
    n = rgb.shape[0]
    g = torch.Generator()
    g.manual_seed(42 + epoch)
    perm = torch.randperm(n, generator=g)
    for s in range(0, n, BATCH_SIZE):
        idx = perm[s:s+BATCH_SIZE]
        yield (
            rgb.index_select(0, idx).to(device, non_blocking=True),
            masks.index_select(0, idx).to(device, non_blocking=True),
            poses.index_select(0, idx).to(device, non_blocking=True),
        )

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_cleanup()

    # Deterministic for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # ── Load data (cached, <1s after first run) ──
    data = load_data(device)
    rgb = data["train_rgb"]
    masks = data["train_masks"]
    poses = data["train_poses"]
    n = rgb.shape[0]
    print(f"train: {n} pairs, val: {data['val_rgb'].shape[0]} pairs")

    # ── Build model ──
    gen = Generator().to(device)
    n_params = sum(p.numel() for p in gen.parameters())
    print(f"params: {n_params}")

    # ── Load frozen SegNet + PoseNet for loss computation ──
    segnet = load_segnet(device)
    posenet = load_posenet(device)
    print(f"gpu_mem: {gpu_mem_mb():.0f}MB")

    t_start = time.time()
    epoch = 0
    t_anchor_end = TRAIN_BUDGET_SEC * T_ANCHOR
    t_ft_end = TRAIN_BUDGET_SEC * (T_ANCHOR + T_FINETUNE)
    t_jt_end = TRAIN_BUDGET_SEC * (T_ANCHOR + T_FINETUNE + T_JOINT)

    # ════════════════ Stage 1: Anchor (frame2 SegNet) ════════════════
    for p in gen.h1.parameters(): p.requires_grad = False
    for p in gen.pose_mlp.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()), lr=LR, betas=(0.9, 0.99))

    while time.time() - t_start < t_anchor_end:
        gen.train()
        elapsed = time.time() - t_start
        qat = elapsed > t_anchor_end * QAT_FRAC
        gen.set_qat(qat)
        if elapsed < t_anchor_end * 0.85:
            boost = ERR_BOOST
        elif elapsed < t_anchor_end * 0.95:
            boost = ERR_BOOST_HI
        else:
            boost = ERR_BOOST_HI * 2.0
        # KL→CE schedule
        alpha = min(1.0, elapsed / max(1, t_anchor_end * QAT_FRAC * 0.5))
        kl_w = 0.9 - 0.9 * alpha
        ce_w = 0.1 + 0.9 * alpha

        for b_rgb, b_mask, b_pose in make_batches(rgb, masks, poses, epoch, device):
            batch = einops.rearrange(b_rgb, "b t h w c -> b t c h w").float()
            with torch.no_grad():
                r2 = F.interpolate(batch[:, 1], (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
                gt_logits = segnet(r2).float()
                gt_cls = gt_logits.argmax(1)
            opt.zero_grad(set_to_none=True)
            _, p2 = gen(b_mask.long(), b_pose.float())
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            pred_logits = segnet(f2d).float()
            ce = F.cross_entropy(pred_logits, gt_cls, reduction='none')
            with torch.no_grad():
                w = 1.0 + (pred_logits.argmax(1) != gt_cls).float() * boost
            loss = 100.0 * (kl_w * kl_on_logits(pred_logits, gt_logits) / (MODEL_H*MODEL_W) + ce_w * 0.5 * (ce * w).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt.step()
        epoch += 1

    anchor_ep = epoch
    print(f"anchor_epochs: {anchor_ep}")

    # ════════════════ Stage 2: Finetune (frame1 PoseNet) ════════════════
    for p in gen.parameters(): p.requires_grad = True
    for p in gen.trunk.parameters(): p.requires_grad = False
    for p in gen.h2.parameters(): p.requires_grad = False
    gen.trunk.eval(); gen.h2.eval()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, gen.parameters()), lr=FT_LR, betas=(0.9, 0.99))
    gen.set_qat(True)

    while time.time() - t_start < t_ft_end:
        gen.h1.train(); gen.pose_mlp.train()
        for b_rgb, b_mask, b_pose in make_batches(rgb, masks, poses, 1000 + epoch, device):
            batch = einops.rearrange(b_rgb, "b t h w c -> b t c h w").float()
            with torch.no_grad():
                gt_p = get_pose6(posenet, posenet.preprocess_input(batch)).float()
            opt.zero_grad(set_to_none=True)
            p1, p2 = gen(b_mask.long(), b_pose.float())
            f1d = F.interpolate(diff_round(F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False).clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False).clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            loss = 10.0 * F.smooth_l1_loss(fp, gt_p, beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt.step()
        epoch += 1

    ft_ep = epoch - anchor_ep
    print(f"finetune_epochs: {ft_ep}")

    # ════════════════ Stage 3: Joint ════════════════
    for p in gen.parameters(): p.requires_grad = True
    opt = torch.optim.AdamW(gen.parameters(), lr=JT_LR, betas=(0.9, 0.99))
    gen.set_qat(True)
    # EMA of gen parameters during joint
    ema_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    ema_decay = 0.9

    while time.time() - t_start < t_jt_end:
        gen.train()
        for b_rgb, b_mask, b_pose in make_batches(rgb, masks, poses, 2000 + epoch, device):
            batch = einops.rearrange(b_rgb, "b t h w c -> b t c h w").float()
            with torch.no_grad():
                r2 = F.interpolate(batch[:, 1], (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
                gt_logits = segnet(r2).float()
                gt_cls = gt_logits.argmax(1)
                gt_p = get_pose6(posenet, posenet.preprocess_input(batch)).float()
            opt.zero_grad(set_to_none=True)
            p1, p2 = gen(b_mask.long(), b_pose.float())
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f1d = F.interpolate(diff_round(f1u.clamp(0,255)), (MODEL_H,MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0,255)), (MODEL_H,MODEL_W), mode="bilinear", align_corners=False)
            pred_logits = segnet(f2d).float()
            ce = F.cross_entropy(pred_logits, gt_cls, reduction='none')
            with torch.no_grad():
                w = 1.0 + (pred_logits.argmax(1) != gt_cls).float() * ERR_BOOST
            seg_loss = 100.0 * (ce * w).mean()
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            pose_loss = 30.0 * F.mse_loss(fp, gt_p)
            loss = seg_loss + pose_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema_state[k].copy_(v)
        epoch += 1

    jt_ep = epoch - anchor_ep - ft_ep
    train_time = time.time() - t_start
    print(f"joint_epochs: {jt_ep}")
    print(f"total_epochs: {epoch}")
    print(f"training_sec: {train_time:.1f}")

    # ── Free training-only nets before eval ──
    del segnet, posenet, opt
    gpu_cleanup()

    # ── Swap to EMA weights for eval ──
    gen.load_state_dict(ema_state)

    # ── Evaluate ──
    result = evaluate(gen, data, device)

    # ── Print parseable output ──
    print("---")
    for k in ["score", "seg_term", "pose_term", "rate_term", "model_bytes", "total_bytes", "n_params"]:
        v = result[k]
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # ── Clean exit ──
    del gen, data
    gpu_cleanup()

if __name__ == "__main__":
    from prepare import gpu_mem_mb
    train()
