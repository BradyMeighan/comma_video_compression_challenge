#!/usr/bin/env python
"""
train.py — Model + training loop. THIS IS THE FILE THE AGENT EDITS.

Everything is fair game: architecture, hyperparameters, optimizer, loss,
training stages, quantization strategy. The only constraint is that it
runs within the 5-minute time budget and prints the parseable output.

Run: python train.py

Validation env vars:
  CONFIG=A|B|C|D       : A=baseline, B=boundary+Lion, C=joint-from-start+boundary, D=per-dim pose MSE
  TRAIN_BUDGET_SEC=N   : override training budget (default = prepare.TRAIN_BUDGET_SEC)
  FULL_DATA=1          : use 500/100 split from full 600-pair dataset (vs proxy 80/20)
"""
import sys, os, math, time, gc
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
    CACHE_DIR, ROOT,
)

# ── Validation config (env-var driven) ──
CONFIG          = os.environ.get("CONFIG", "PROD").upper()
USE_BOUNDARY    = CONFIG in ("B", "C")
USE_LION        = CONFIG == "B"
JOINT_ONLY      = CONFIG == "C"
PERDIM_POSE     = CONFIG == "D"
USE_FULL_DATA   = bool(int(os.environ.get("FULL_DATA", "0")))

_budget_override = os.environ.get("TRAIN_BUDGET_SEC_OVERRIDE")
if _budget_override:
    TRAIN_BUDGET_SEC = int(_budget_override)

# HP overrides for tuning runs (env-var driven)
EMA_DECAY_OVERRIDE  = float(os.environ.get("EMA_DECAY", "0.9"))
COSINE_LR           = bool(int(os.environ.get("COSINE_LR", "0")))
GRAD_CLIP_OVERRIDE  = float(os.environ.get("GRAD_CLIP_OVERRIDE", "0"))  # 0 = use default
CHECKPOINT_INTERVAL_SEC = int(os.environ.get("CHECKPOINT_INTERVAL_SEC", "0"))  # 0 = disabled
print(f"[config] CONFIG={CONFIG} budget={TRAIN_BUDGET_SEC}s full_data={USE_FULL_DATA} "
      f"ema={EMA_DECAY_OVERRIDE} cosine_lr={COSINE_LR} grad_clip={GRAD_CLIP_OVERRIDE or 'default'}")

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
if GRAD_CLIP_OVERRIDE > 0:
    GRAD_CLIP = GRAD_CLIP_OVERRIDE
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
# OPTIMIZERS / DATA LOADERS
# ══════════════════════════════════════════════════════════════════════

class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023). Use lr ~3-10x smaller than AdamW."""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if group['weight_decay']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                grad = p.grad
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = (exp_avg.mul(beta1).add(grad, alpha=1 - beta1)).sign_()
                p.add_(update, alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss

def load_data_full(device):
    """Full-dataset variant: split all 600 pairs into 500 train / 100 val. Cached separately."""
    from tqdm import tqdm
    from frame_utils import AVVideoDataset
    from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
    from safetensors.torch import load_file

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / "full_split_500_100.pt"
    if cache_file.exists():
        print(f"[full] Loading cached full data ({cache_file.stat().st_size // 1024 // 1024}MB)")
        return torch.load(cache_file, map_location="cpu", weights_only=True)

    print("[full] Building full data cache (first run, ~3min)...")
    t0 = time.time()
    files = [l.strip() for l in (ROOT / "public_test_video_names.txt").read_text().splitlines() if l.strip()]
    ds = AVVideoDataset(files, data_dir=ROOT / "videos", batch_size=16,
                        device=torch.device("cpu"), num_threads=2, seed=1234, prefetch_queue_depth=2)
    ds.prepare_data()
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
    all_rgb = []
    for _, _, batch in tqdm(dl, desc="[full] Loading video", leave=False):
        all_rgb.append(batch)
    all_rgb = torch.cat(all_rgb, 0)
    print(f"[full] Loaded {all_rgb.shape[0]} pairs in {time.time()-t0:.1f}s")

    g = torch.Generator(); g.manual_seed(42)
    idx = torch.randperm(all_rgb.shape[0], generator=g)
    train_idx, val_idx = idx[:500], idx[500:600]
    train_rgb = all_rgb[train_idx].contiguous()
    val_rgb = all_rgb[val_idx].contiguous()
    del all_rgb; gc.collect()

    segnet = SegNet().eval().to(device); segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))
    posenet = PoseNet().eval().to(device); posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))

    def extract(rgb_subset, label):
        masks_l, poses_l = [], []
        BS = 8
        with torch.inference_mode():
            for i in tqdm(range(0, rgb_subset.shape[0], BS), desc=f"[full] {label}"):
                b = rgb_subset[i:i+BS].to(device).float()
                bc = einops.rearrange(b, 'b t h w c -> b t c h w')
                r2 = F.interpolate(bc[:, 1], (MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
                masks_l.append(segnet(r2).float().argmax(1).to(torch.uint8).cpu())
                poses_l.append(get_pose6(posenet, posenet.preprocess_input(bc)).float().cpu())
        return torch.cat(masks_l, 0).contiguous(), torch.cat(poses_l, 0).contiguous()

    train_masks, train_poses = extract(train_rgb, "train")
    val_masks, val_poses = extract(val_rgb, "val")
    del segnet, posenet; gpu_cleanup()

    data = {
        "train_rgb": train_rgb, "train_masks": train_masks, "train_poses": train_poses,
        "val_rgb": val_rgb,     "val_masks": val_masks,     "val_poses": val_poses,
    }
    torch.save(data, cache_file)
    print(f"[full] Cached {cache_file.stat().st_size // 1024 // 1024}MB in {time.time()-t0:.1f}s")
    return data

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
        self.pre = DSConv(C1, HEAD_HIDDEN)
        self.out = QConv2d(HEAD_HIDDEN, 3, 1, quantize_weight=False)
    def forward(self, f):
        return torch.sigmoid(self.out(self.pre(self.r1(f)))) * 255.0

class Head1(nn.Module):
    def __init__(self):
        super().__init__()
        self.r1 = FiLMRes(C1, COND_DIM)
        self.r2 = FiLMRes(C1, COND_DIM)
        self.out = QConv2d(C1, 3, 1, quantize_weight=False)
    def forward(self, f, c):
        return torch.sigmoid(self.out(self.r2(self.r1(f, c), c))) * 255.0

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = Trunk()
        self.pose_mlp = nn.Sequential(
            nn.Linear(6, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM), nn.SiLU(),
            nn.Linear(COND_DIM, COND_DIM),
        )
        # trunk_film removed: FiLMRes inside Head1 handles pose conditioning directly
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
        return self.h1(feat, cond), self.h2(feat)

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
    data = load_data_full(device) if USE_FULL_DATA else load_data(device)
    rgb = data["train_rgb"]
    masks = data["train_masks"]
    poses = data["train_poses"]
    n = rgb.shape[0]
    print(f"train: {n} pairs, val: {data['val_rgb'].shape[0]} pairs")
    # Per-dim pose std for normalization (CONFIG=D)
    pose_std = data["train_poses"].std(0).clamp_min(1e-3).to(device) if PERDIM_POSE else None

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

    # Optimizer factory: AdamW (default) or Lion (CONFIG=B). Lion uses lr/3.
    def make_opt(params, lr_val):
        if USE_LION:
            return Lion(list(params), lr=lr_val / 3.0, betas=(0.9, 0.99))
        return torch.optim.AdamW(list(params), lr=lr_val, betas=(0.9, 0.99))

    def cosine_lr_factor(progress):
        """Cosine decay factor in [0.1, 1.0] for progress in [0, 1]."""
        p = max(0.0, min(1.0, progress))
        return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * p)))

    def apply_cosine(opt, init_lr, progress):
        if COSINE_LR:
            f = cosine_lr_factor(progress)
            for g in opt.param_groups:
                g['lr'] = init_lr * f

    def boundary_mask(gt_cls):
        """Returns (B,H,W) float mask, 1.0 at class boundaries."""
        gt_pad = F.pad(gt_cls.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate')
        c = gt_pad[:, :, 1:-1, 1:-1]
        return ((c != gt_pad[:, :, :-2, 1:-1]) | (c != gt_pad[:, :, 2:, 1:-1]) |
                (c != gt_pad[:, :, 1:-1, :-2]) | (c != gt_pad[:, :, 1:-1, 2:])).squeeze(1).float()

    # ════════════════ JOINT-ONLY (CONFIG=C): single stage from epoch 0 ════════════════
    if JOINT_ONLY:
        for p in gen.parameters(): p.requires_grad = True
        opt = make_opt(gen.parameters(), LR)
        joint_only_init_lr = opt.param_groups[0]['lr']
        ema_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
        ema_decay = EMA_DECAY_OVERRIDE
        total_budget = TRAIN_BUDGET_SEC * 0.95
        last_ckpt_time = time.time()

        while time.time() - t_start < total_budget:
            gen.train()
            elapsed = time.time() - t_start
            progress = elapsed / total_budget
            apply_cosine(opt, joint_only_init_lr, progress)
            qat = progress > 0.4  # QAT after 40% of training
            gen.set_qat(qat)
            # KL→CE schedule (early)
            alpha = min(1.0, progress / 0.2)
            kl_w = 0.9 - 0.9 * alpha
            ce_w = 0.1 + 0.9 * alpha
            # Pose weight ramps up over training
            pose_w = min(1.0, progress / 0.3)

            for b_rgb, b_mask, b_pose in make_batches(rgb, masks, poses, epoch, device):
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
                    p_t = torch.exp(-ce.detach()).clamp_max(0.999)
                    focal_w = (1.0 - p_t).pow(2.0)
                    weight = focal_w * (1.0 + 4.0 * boundary_mask(gt_cls))  # boundary-weighted
                seg_loss = 100.0 * (kl_w * kl_on_logits(pred_logits, gt_logits) / (MODEL_H*MODEL_W) + ce_w * 25.0 * (weight * ce).mean())
                fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
                pose_loss = 30.0 * F.mse_loss(fp, gt_p)
                loss = seg_loss + pose_w * pose_loss
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
            # Periodic checkpoint to a sibling .ckpt file
            ckpt_path = os.environ.get("SAVE_MODEL_PATH", "")
            if ckpt_path and CHECKPOINT_INTERVAL_SEC > 0 and (time.time() - last_ckpt_time) > CHECKPOINT_INTERVAL_SEC:
                torch.save(ema_state, ckpt_path + ".ckpt")
                last_ckpt_time = time.time()
                print(f"[ckpt] saved EMA state to {ckpt_path}.ckpt at epoch {epoch}")

        train_time = time.time() - t_start
        print(f"joint_only_epochs: {epoch}")
        print(f"total_epochs: {epoch}")
        print(f"training_sec: {train_time:.1f}")
        del segnet, posenet, opt
        gpu_cleanup()
        gen.load_state_dict(ema_state)
        result = evaluate(gen, data, device)
        print("---")
        for k in ["score", "seg_term", "pose_term", "rate_term", "model_bytes", "total_bytes", "n_params"]:
            v = result[k]
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")
        save_path = os.environ.get("SAVE_MODEL_PATH", "")
        if save_path:
            torch.save(gen.state_dict(), save_path)
            print(f"saved_model: {save_path}")
        del gen, data; gpu_cleanup()
        return

    # ════════════════ Stage 1: Anchor (frame2 SegNet) ════════════════
    for p in gen.h1.parameters(): p.requires_grad = False
    for p in gen.pose_mlp.parameters(): p.requires_grad = False
    opt = make_opt(filter(lambda p: p.requires_grad, gen.parameters()), LR)
    anchor_init_lr = opt.param_groups[0]['lr']

    while time.time() - t_start < t_anchor_end:
        gen.train()
        elapsed = time.time() - t_start
        apply_cosine(opt, anchor_init_lr, elapsed / max(1e-3, t_anchor_end))
        qat = elapsed > t_anchor_end * QAT_FRAC
        gen.set_qat(qat)
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
                p_t = torch.exp(-ce.detach()).clamp_max(0.999)
                focal_w = (1.0 - p_t).pow(2.0)
                if USE_BOUNDARY:
                    weight = focal_w * (1.0 + 4.0 * boundary_mask(gt_cls))
                else:
                    weight = focal_w
            loss = 100.0 * (kl_w * kl_on_logits(pred_logits, gt_logits) / (MODEL_H*MODEL_W) + ce_w * 25.0 * (weight * ce).mean())
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
    opt = make_opt(filter(lambda p: p.requires_grad, gen.parameters()), FT_LR)
    ft_init_lr = opt.param_groups[0]['lr']
    gen.set_qat(True)

    while time.time() - t_start < t_ft_end:
        ft_progress = (time.time() - t_start - t_anchor_end) / max(1e-3, t_ft_end - t_anchor_end)
        apply_cosine(opt, ft_init_lr, ft_progress)
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
            if PERDIM_POSE:
                loss = 10.0 * (((fp - gt_p) / pose_std).pow(2)).mean()
            else:
                loss = 10.0 * F.smooth_l1_loss(fp, gt_p, beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt.step()
        epoch += 1

    ft_ep = epoch - anchor_ep
    print(f"finetune_epochs: {ft_ep}")

    # ════════════════ Stage 3: Joint ════════════════
    for p in gen.parameters(): p.requires_grad = True
    opt = make_opt(gen.parameters(), JT_LR)
    jt_init_lr = opt.param_groups[0]['lr']
    gen.set_qat(True)
    # EMA of gen parameters during joint
    ema_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    ema_decay = EMA_DECAY_OVERRIDE
    last_ckpt_time = time.time()

    while time.time() - t_start < t_jt_end:
        jt_progress = (time.time() - t_start - t_ft_end) / max(1e-3, t_jt_end - t_ft_end)
        apply_cosine(opt, jt_init_lr, jt_progress)
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
            if PERDIM_POSE:
                pose_loss = 30.0 * (((fp - gt_p) / pose_std).pow(2)).mean()
            else:
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
        # Periodic checkpoint
        ckpt_path = os.environ.get("SAVE_MODEL_PATH", "")
        if ckpt_path and CHECKPOINT_INTERVAL_SEC > 0 and (time.time() - last_ckpt_time) > CHECKPOINT_INTERVAL_SEC:
            torch.save(ema_state, ckpt_path + ".ckpt")
            last_ckpt_time = time.time()
            print(f"[ckpt] saved EMA state at joint epoch {epoch - anchor_ep - ft_ep}")

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

    # ── Save model state_dict if requested (set SAVE_MODEL_PATH env var) ──
    save_path = os.environ.get("SAVE_MODEL_PATH", "")
    if save_path:
        torch.save(gen.state_dict(), save_path)
        print(f"saved_model: {save_path}")

    # ── Clean exit ──
    del gen, data
    gpu_cleanup()

if __name__ == "__main__":
    from prepare import gpu_mem_mb
    train()
