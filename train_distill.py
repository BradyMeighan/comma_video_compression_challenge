#!/usr/bin/env python
"""
Distill teacher SegNet/PoseNet into tiny student models for adversarial decode.
v2: MobileUNet with Fourier features + spatial bias + base map + boundary-weighted loss
    PoseLUT with differentiable lookup table

Phase 0: Extract teacher data + compute base map
Phase 1: Train student models (1a=seg, 1b=pose)
Phase 2: Validate adversarial transfer (go/no-go gate)

Usage:
  python train_distill.py --phase 0 --video videos/0.mkv
  python train_distill.py --phase 1a --seg-ch 48 --seg-epochs 200
  python train_distill.py --phase 1b --pose-epochs 300
  python train_distill.py --phase 2
  python train_distill.py --phase all
"""
import sys, time, math, argparse, os
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, einops
from pathlib import Path
from safetensors.torch import load_file

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None

from frame_utils import (
    AVVideoDataset, camera_size, segnet_model_input_size, rgb_to_yuv6
)
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path

DISTILL_DIR = Path('distill_data')
MODEL_DIR = Path('tiny_models')


# ─────────────────────────────────────────────────────────────────────
#  Differentiable preprocessing (needed for adversarial decode)
# ─────────────────────────────────────────────────────────────────────

def rgb_to_yuv6_diff(rgb_chw):
    """Differentiable RGB->YUV6 conversion."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]
    R, G, B = rgb[..., 0, :, :], rgb[..., 1, :, :], rgb[..., 2, :, :]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    return torch.stack([Y[..., 0::2, 0::2], Y[..., 1::2, 0::2],
                        Y[..., 0::2, 1::2], Y[..., 1::2, 1::2],
                        U_sub, V_sub], dim=-3)


def posenet_preprocess_diff(x):
    """Differentiable (B, 2, 3, H, W) -> (B, 12, H/2, W/2) YUV6."""
    B, T = x.shape[0], x.shape[1]
    flat = x.reshape(B * T, *x.shape[2:])
    yuv = rgb_to_yuv6_diff(flat)
    return yuv.reshape(B, T * yuv.shape[1], *yuv.shape[2:])


def margin_loss(logits, target, margin=3.0):
    """Push target class logit above all competitors by margin."""
    target_logits = logits.gather(1, target.unsqueeze(1))
    competitor = logits.clone()
    competitor.scatter_(1, target.unsqueeze(1), float('-inf'))
    max_other = competitor.max(dim=1, keepdim=True).values
    return F.relu(max_other - target_logits + margin).mean()


# ─────────────────────────────────────────────────────────────────────
#  Building Blocks
# ─────────────────────────────────────────────────────────────────────

class InvertedResidual(nn.Module):
    """MobileNetV2-style: pointwise expand -> depthwise 3x3 -> pointwise project."""
    def __init__(self, in_ch, out_ch, expand_ratio=4):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = (in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU6(True)]
        layers += [nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(True),
                   nn.Conv2d(mid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return (x + out) if self.use_res else out


class DWSConv(nn.Module):
    """Depthwise-separable: depthwise 3x3 -> pointwise 1x1."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU6(True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU6(True))

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────
#  Student Architectures
# ─────────────────────────────────────────────────────────────────────

class MobileUNet(nn.Module):
    """
    Tiny UNet with inverted residual encoder, Fourier positional encoding,
    learnable low-rank spatial bias, and fixed base map prior.
    ~470K params at base_ch=48.
    """
    def __init__(self, in_ch=3, n_classes=5, base_ch=48, fourier_L=10,
                 H=384, W=512, bias_rank=16, base_scale=10.0):
        super().__init__()
        self.n_classes = n_classes
        self.base_scale = base_scale
        self.fourier_L = fourier_L

        fourier_ch = 2 + 4 * fourier_L   # 42 for L=10
        total_in = in_ch + fourier_ch     # 45
        b = base_ch

        # ---- Encoder ----
        self.stem = nn.Sequential(
            nn.Conv2d(total_in, b, 3, padding=1, bias=False),
            nn.BatchNorm2d(b), nn.ReLU6(True))
        self.enc1 = InvertedResidual(b, b, expand_ratio=2)       # full res
        self.enc2 = nn.Sequential(                                # half res
            InvertedResidual(b, 2*b, expand_ratio=4),
            InvertedResidual(2*b, 2*b, expand_ratio=4))
        self.bottleneck = nn.Sequential(                          # quarter res
            InvertedResidual(2*b, 4*b, expand_ratio=2),
            InvertedResidual(4*b, 4*b, expand_ratio=2))
        self.pool = nn.MaxPool2d(2)

        # ---- Decoder ----
        self.reduce2 = nn.Sequential(
            nn.Conv2d(4*b, 2*b, 1, bias=False), nn.BatchNorm2d(2*b))
        self.dec2 = DWSConv(4*b, 2*b)       # cat(reduce2, enc2) -> 4b -> 2b
        self.reduce1 = nn.Sequential(
            nn.Conv2d(2*b, b, 1, bias=False), nn.BatchNorm2d(b))
        self.dec1 = DWSConv(2*b, b)          # cat(reduce1, enc1) -> 2b -> b

        # ---- Output ----
        self.head = nn.Conv2d(b, n_classes, 1)

        # Learnable spatial bias (rank-R factorized)
        self.bias_left = nn.Parameter(torch.randn(n_classes, H, bias_rank) * 0.01)
        self.bias_right = nn.Parameter(torch.randn(n_classes, bias_rank, W) * 0.01)

        # Fixed base map prior (set via init_base_map)
        self.register_buffer('base_prior', torch.zeros(1, n_classes, H, W))

        # Fourier cache (recomputed on first forward, not saved)
        self._fourier = None

    def _make_fourier(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        ch = [xx.unsqueeze(0), yy.unsqueeze(0)]
        for l in range(self.fourier_L):
            freq = math.pi * (2 ** l)
            ch += [torch.sin(freq * xx).unsqueeze(0),
                   torch.cos(freq * xx).unsqueeze(0),
                   torch.sin(freq * yy).unsqueeze(0),
                   torch.cos(freq * yy).unsqueeze(0)]
        return torch.cat(ch, dim=0)

    def init_base_map(self, base_map_tensor):
        """Set base map prior. base_map_tensor: (H, W) LongTensor of class indices."""
        oh = F.one_hot(base_map_tensor.long(), self.n_classes).permute(2, 0, 1).float()
        self.base_prior.copy_((oh * self.base_scale).unsqueeze(0))

    def forward(self, x):
        B, C, H, W = x.shape

        # Fourier positional encoding (lazy-init, not saved in state_dict)
        if self._fourier is None or self._fourier.device != x.device \
                or self._fourier.shape[-2] != H or self._fourier.shape[-1] != W:
            self._fourier = self._make_fourier(H, W, x.device)
        x = torch.cat([x, self._fourier.unsqueeze(0).expand(B, -1, -1, -1)], dim=1)

        # Encoder
        e1 = self.enc1(self.stem(x))              # (B, b, H, W)
        e2 = self.enc2(self.pool(e1))              # (B, 2b, H/2, W/2)
        bn = self.bottleneck(self.pool(e2))        # (B, 4b, H/4, W/4)

        # Decoder
        d = F.interpolate(bn, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([self.reduce2(d), e2], dim=1))
        d = F.interpolate(d, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec1(torch.cat([self.reduce1(d), e1], dim=1))

        # Output = CNN logits + learned spatial bias + fixed base prior
        logits = self.head(d)
        logits = logits + torch.bmm(self.bias_left, self.bias_right).unsqueeze(0)
        logits = logits + self.base_prior
        return logits


class PoseLUT(nn.Module):
    """
    Differentiable pose lookup table.
    Stores teacher's exact pose outputs. Tiny CNN produces embedding,
    softmax over anchor similarities gives interpolation weights,
    output = weighted sum of stored poses. ~49K params at base_ch=16.
    """
    def __init__(self, n_frames=600, pose_dim=6, embed_dim=32, base_ch=16):
        super().__init__()
        b = base_ch

        # Fixed pose table (teacher's exact outputs)
        self.register_buffer('pose_table', torch.zeros(n_frames, pose_dim))

        # Tiny CNN -> embedding
        self.features = nn.Sequential(
            nn.Conv2d(12, b, 5, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(b), nn.ReLU(True),
            nn.Conv2d(b, 2*b, 3, stride=4, padding=1, bias=False),
            nn.BatchNorm2d(2*b), nn.ReLU(True),
            nn.Conv2d(2*b, 4*b, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4*b), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1))
        self.embed = nn.Linear(4*b, embed_dim)

        # Learned anchor embedding per frame
        self.anchors = nn.Parameter(torch.randn(n_frames, embed_dim) * 0.1)

        # Learned temperature for softmax sharpness
        self.log_temp = nn.Parameter(torch.tensor(0.0))

        # Normalization (same constants as teacher PoseNet)
        self.register_buffer('_mean', torch.tensor([127.5]*12).view(1, 12, 1, 1))
        self.register_buffer('_std', torch.tensor([63.75]*12).view(1, 12, 1, 1))

    def forward(self, x):
        x = (x - self._mean) / self._std
        feat = self.features(x).flatten(1)
        emb = self.embed(feat)                          # (B, D)
        emb_n = F.normalize(emb, dim=1)
        anc_n = F.normalize(self.anchors, dim=1)        # (N, D)
        sim = emb_n @ anc_n.t()                         # (B, N)
        temp = self.log_temp.exp().clamp(min=0.01)
        weights = F.softmax(sim / temp, dim=1)           # (B, N)
        pose = weights @ self.pose_table                 # (B, pose_dim)
        return {'pose': pose}


# ─────────────────────────────────────────────────────────────────────
#  Boundary weight computation
# ─────────────────────────────────────────────────────────────────────

def compute_boundary_weights(argmax_np, alpha=20.0, eps=1.0):
    """Per-pixel weight: higher near class boundaries."""
    H, W = argmax_np.shape
    bd = np.zeros((H, W), dtype=bool)
    bd[:-1] |= argmax_np[:-1] != argmax_np[1:]
    bd[1:]  |= argmax_np[1:]  != argmax_np[:-1]
    bd[:, :-1] |= argmax_np[:, :-1] != argmax_np[:, 1:]
    bd[:, 1:]  |= argmax_np[:, 1:]  != argmax_np[:, :-1]
    if distance_transform_edt is not None:
        dist = distance_transform_edt(~bd)
        return (1.0 + alpha / (dist + eps)).astype(np.float32)
    # Fallback: just flag boundary pixels
    return (1.0 + alpha * bd.astype(np.float32)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
#  Phase 0: Extract teacher data
# ─────────────────────────────────────────────────────────────────────

def extract_teacher_data(video_path, device):
    print("=" * 60)
    print("PHASE 0: Extracting teacher data")
    print("=" * 60)
    DISTILL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    ds = AVVideoDataset([video_path.name], data_dir=video_path.parent,
                        batch_size=16, device=torch.device('cpu'))
    ds.prepare_data()

    all_sl, all_si, all_po, all_pi = [], [], [], []
    print("Running teacher inference...")
    with torch.inference_mode():
        for path, idx, batch in ds:
            batch = batch.to(device)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
            seg_in = segnet.preprocess_input(x)
            all_sl.append(segnet(seg_in).cpu())
            all_si.append(seg_in.cpu())
            pn_in = posenet.preprocess_input(x)
            all_po.append(posenet(pn_in)['pose'][:, :6].cpu())
            all_pi.append(pn_in.cpu())
            print(f"  Batch {idx}: {batch.shape[0]} pairs")

    del segnet, posenet
    torch.cuda.empty_cache()

    seg_logits = torch.cat(all_sl, 0)
    seg_inputs = torch.cat(all_si, 0)
    pose_outputs = torch.cat(all_po, 0)
    pose_inputs = torch.cat(all_pi, 0)

    # Base map: mode class per pixel across all frames
    seg_argmax = seg_logits.argmax(1).numpy()
    N, H, W = seg_argmax.shape
    class_counts = np.zeros((H, W, 5), dtype=np.int32)
    for c in range(5):
        class_counts[..., c] = (seg_argmax == c).sum(axis=0)
    base_map = class_counts.argmax(axis=-1).astype(np.uint8)
    base_agree = (seg_argmax == base_map[None]).mean()

    print(f"\nBase map: {base_agree*100:.1f}% of pixels are static across 600 frames")
    print(f"  Class distribution in base map: {np.bincount(base_map.flatten(), minlength=5)}")

    torch.save(seg_logits, DISTILL_DIR / 'seg_logits.pt')
    torch.save(seg_inputs, DISTILL_DIR / 'seg_inputs.pt')
    torch.save(pose_outputs, DISTILL_DIR / 'pose_outputs.pt')
    torch.save(pose_inputs, DISTILL_DIR / 'pose_inputs.pt')
    np.save(DISTILL_DIR / 'base_map.npy', base_map)

    print(f"Saved to {DISTILL_DIR}/ in {time.time()-t0:.1f}s")
    print(f"  seg_logits: {seg_logits.shape}, seg_inputs: {seg_inputs.shape}")
    print(f"  pose_outputs: {pose_outputs.shape}, pose_inputs: {pose_inputs.shape}")
    print(f"  base_map: {base_map.shape}")


# ─────────────────────────────────────────────────────────────────────
#  Augmentation
# ─────────────────────────────────────────────────────────────────────

IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251],
    [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493],
    [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


def augment_seg_batch(x, seg_argmax, device):
    """Adversarial-style augmented seg inputs (flat colors, blends, noise)."""
    B, C, H, W = x.shape
    r = torch.rand(B, 1, 1, 1, device=device)
    colors = IDEAL_COLORS.to(device)
    flat = colors[seg_argmax].permute(0, 3, 1, 2)
    aug = x.to(device)

    # 40%: blend original with flat-colored (mid-optimization)
    m = (r < 0.4).float()
    a = torch.rand(B, 1, 1, 1, device=device) * 0.7 + 0.3
    aug = m * (a * flat + (1-a) * aug) + (1-m) * aug

    # 30%: gaussian noise
    m = ((r >= 0.4) & (r < 0.7)).float()
    aug = aug + m * torch.randn_like(aug) * (10 + torch.rand(B,1,1,1,device=device)*20)

    # 20%: pure flat colored (initial adversarial state)
    m = ((r >= 0.7) & (r < 0.9)).float()
    aug = m * flat + (1-m) * aug

    # 10%: brightness jitter
    m = (r >= 0.9).float()
    br = 1 + (torch.rand(B,1,1,1,device=device) - 0.5) * 0.6
    aug = m * aug * br + (1-m) * aug

    return aug.clamp(0, 255)


def augment_pose_batch(x, seg_argmax_for_pose, device):
    """Adversarial-style augmented pose inputs."""
    B, C, H, W = x.shape
    r = torch.rand(B, device=device)
    aug = x.to(device)
    colors = IDEAL_COLORS.to(device)
    results = []

    for i in range(B):
        if r[i] < 0.35 and seg_argmax_for_pose is not None:
            sm = seg_argmax_for_pose[i]
            f1 = colors[sm].permute(2, 0, 1).unsqueeze(0)
            f1 = F.interpolate(f1, size=(H*2, W*2), mode='bilinear', align_corners=False)
            f0 = f1.mean(dim=(-2,-1), keepdim=True).expand_as(f1)
            nl = torch.rand(1, device=device).item() * 30
            f1 = (f1 + torch.randn_like(f1) * nl).clamp(0, 255)
            f0 = (f0 + torch.randn_like(f0) * nl).clamp(0, 255)
            yuv = posenet_preprocess_diff(torch.stack([f0, f1], dim=1))
            results.append(yuv.squeeze(0))
        elif r[i] < 0.65:
            n = torch.randn(C, H, W, device=device) * (5 + torch.rand(1,device=device).item()*20)
            results.append((aug[i] + n).clamp(0, 255))
        elif r[i] < 0.85:
            s = (torch.rand(1, device=device).item() - 0.5) * 40
            sc = 0.8 + torch.rand(1, device=device).item() * 0.4
            results.append((aug[i] * sc + s).clamp(0, 255))
        else:
            results.append(aug[i])

    return torch.stack(results, dim=0)


# ─────────────────────────────────────────────────────────────────────
#  Phase 1a: Train SegNet student
# ─────────────────────────────────────────────────────────────────────

def train_segnet_student(device, seg_base_ch=48, epochs=200, lr=3e-3):
    print("\n" + "=" * 60)
    print(f"PHASE 1a: MobileUNet (base_ch={seg_base_ch}, epochs={epochs})")
    print("=" * 60)

    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)   # (N, H, W)
    N = seg_logits.shape[0]
    base_map = torch.from_numpy(np.load(DISTILL_DIR / 'base_map.npy'))

    # Teacher for on-the-fly augmented logits
    teacher = SegNet().eval().to(device)
    teacher.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student
    student = MobileUNet(in_ch=3, n_classes=5, base_ch=seg_base_ch).to(device)
    student.init_base_map(base_map.to(device))

    n_params = sum(p.numel() for p in student.parameters())
    print(f"Params: {n_params:,} ({n_params*4/1024:.1f} KB f32, {n_params*2/1024:.1f} KB f16)")

    # Check base map agreement
    base_agree = (teacher_argmax.numpy() == base_map.numpy()[None]).mean()
    print(f"Base map covers {base_agree*100:.1f}% of pixels (student only learns the rest)")

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    T_kd = 4.0
    batch_size = 4  # Small — student IR blocks + teacher forward eat VRAM
    best_acc, best_state = 0.0, None
    t0 = time.time()

    for epoch in range(epochs):
        student.train()
        perm = torch.randperm(N)
        sum_loss, sum_correct, sum_pixels, n_b = 0.0, 0, 0, 0
        use_aug = (epoch % 2 == 1)

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            Ba = len(idx)

            x_o = seg_inputs[idx].to(device)
            tl_o = seg_logits[idx].to(device)
            ta_o = teacher_argmax[idx].to(device)

            # Boundary weights (on-the-fly, ~4ms per batch)
            ta_np = teacher_argmax[idx].numpy()
            bw_o = torch.from_numpy(
                np.stack([compute_boundary_weights(ta_np[j]) for j in range(Ba)])
            ).to(device)

            # --- Forward on original frames ---
            sl_o = student(x_o)
            kd_o = F.kl_div(F.log_softmax(sl_o / T_kd, 1),
                            F.softmax(tl_o / T_kd, 1),
                            reduction='none').sum(1)
            ce_o = F.cross_entropy(sl_o, ta_o, reduction='none')
            loss_o = 0.7 * (kd_o * bw_o).mean() * (T_kd**2) + \
                     0.3 * (ce_o * bw_o).mean()

            if use_aug:
                # --- Teacher on augmented (separate pass to save VRAM) ---
                x_a = augment_seg_batch(seg_inputs[idx], teacher_argmax[idx], device)
                with torch.no_grad():
                    tl_a = teacher(x_a)
                    ta_a = tl_a.argmax(1)
                ta_a_np = ta_a.cpu().numpy()
                bw_a = torch.from_numpy(
                    np.stack([compute_boundary_weights(ta_a_np[j]) for j in range(Ba)])
                ).to(device)

                # --- Forward on augmented frames (separate pass) ---
                sl_a = student(x_a)
                kd_a = F.kl_div(F.log_softmax(sl_a / T_kd, 1),
                                F.softmax(tl_a / T_kd, 1),
                                reduction='none').sum(1)
                ce_a = F.cross_entropy(sl_a, ta_a, reduction='none')
                loss_a = 0.7 * (kd_a * bw_a).mean() * (T_kd**2) + \
                         0.3 * (ce_a * bw_a).mean()
                loss = 0.5 * loss_o + 0.5 * loss_a
            else:
                loss = loss_o

            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item() * Ba
            pred = sl_o.detach().argmax(1)
            sum_correct += (pred == ta_o).sum().item()
            sum_pixels += ta_o.numel()
            n_b += 1

        sched.step()
        acc = sum_correct / sum_pixels
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | loss={sum_loss/(n_b*batch_size):.4f} | "
              f"acc={acc:.6f} | best={best_acc:.6f} | "
              f"lr={sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s",
              flush=True)

    del teacher
    torch.cuda.empty_cache()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / 'mobile_segnet.pt')
    size_kb = os.path.getsize(MODEL_DIR / 'mobile_segnet.pt') / 1024
    print(f"\nBest accuracy: {best_acc:.6f}")
    print(f"Saved to {MODEL_DIR / 'mobile_segnet.pt'} ({size_kb:.1f} KB)")
    return student, best_acc


# ─────────────────────────────────────────────────────────────────────
#  Phase 1b: Train PoseNet student
# ─────────────────────────────────────────────────────────────────────

def train_posenet_student(device, pose_base_ch=16, epochs=300, lr=3e-3):
    print("\n" + "=" * 60)
    print(f"PHASE 1b: PoseLUT (base_ch={pose_base_ch}, epochs={epochs})")
    print("=" * 60)

    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True)
    pose_inputs = torch.load(DISTILL_DIR / 'pose_inputs.pt', weights_only=True)
    N = pose_outputs.shape[0]

    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    seg_argmax = seg_logits.argmax(1)
    del seg_logits

    teacher = PoseNet().eval().to(device)
    teacher.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = PoseLUT(n_frames=N, pose_dim=6, embed_dim=32,
                      base_ch=pose_base_ch).to(device)
    student.pose_table.copy_(pose_outputs)  # Store teacher's exact poses

    n_params = sum(p.numel() for p in student.parameters())
    print(f"Params: {n_params:,} ({n_params*4/1024:.1f} KB f32)")
    print(f"Pose table: {student.pose_table.shape} "
          f"({student.pose_table.numel()*4/1024:.1f} KB buffer)")

    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    batch_size = 16
    best_mse, best_state = float('inf'), None
    t0 = time.time()

    for epoch in range(epochs):
        student.train()
        perm = torch.randperm(N)
        sum_loss, n_tr = 0.0, 0
        use_aug = (epoch % 2 == 1)

        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            Ba = len(idx)

            x_o = pose_inputs[idx].to(device)
            tp_o = pose_outputs[idx].to(device)

            if use_aug:
                x_a = augment_pose_batch(pose_inputs[idx], seg_argmax[idx], device)
                with torch.no_grad():
                    tp_a = teacher(x_a)['pose'][:, :6]
                x = torch.cat([x_o, x_a])
                tp = torch.cat([tp_o, tp_a])
            else:
                x, tp = x_o, tp_o

            loss = F.mse_loss(student(x)['pose'], tp)
            opt.zero_grad()
            loss.backward()
            opt.step()

            sum_loss += loss.item() * Ba
            n_tr += Ba

        sched.step()
        mse = sum_loss / n_tr
        if mse < best_mse:
            best_mse = mse
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{epochs} | mse={mse:.8f} | best={best_mse:.8f} | "
              f"sqrt(10*mse)={math.sqrt(10*best_mse):.4f} | "
              f"lr={sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s",
              flush=True)

    del teacher
    torch.cuda.empty_cache()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, MODEL_DIR / 'pose_lut.pt')
    size_kb = os.path.getsize(MODEL_DIR / 'pose_lut.pt') / 1024
    print(f"\nBest MSE: {best_mse:.8f} (sqrt(10*mse)={math.sqrt(10*best_mse):.4f})")
    print(f"Saved to {MODEL_DIR / 'pose_lut.pt'} ({size_kb:.1f} KB)")
    return student, best_mse


# ─────────────────────────────────────────────────────────────────────
#  Phase 2: Transfer validation
# ─────────────────────────────────────────────────────────────────────

def validate_transfer(device, seg_base_ch=48, pose_base_ch=16,
                      num_batches=5, iters=200):
    print("\n" + "=" * 60)
    print("PHASE 2: Transfer validation (go/no-go)")
    print("=" * 60)

    # Teachers
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters():
        p.requires_grad_(False)
    for p in t_pose.parameters():
        p.requires_grad_(False)

    # Students
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True)
    N_frames = pose_outputs.shape[0]

    s_seg = MobileUNet(in_ch=3, n_classes=5, base_ch=seg_base_ch).to(device)
    s_seg.load_state_dict(torch.load(MODEL_DIR / 'mobile_segnet.pt', weights_only=True))
    s_seg.eval()
    for p in s_seg.parameters():
        p.requires_grad_(False)

    # No prior zeroing — use model as trained

    s_pose = PoseLUT(n_frames=N_frames, pose_dim=6, embed_dim=32,
                     base_ch=pose_base_ch).to(device)
    s_pose.load_state_dict(torch.load(MODEL_DIR / 'pose_lut.pt', weights_only=True))
    s_pose.eval()
    for p in s_pose.parameters():
        p.requires_grad_(False)

    seg_p = sum(p.numel() for p in s_seg.parameters())
    pose_p = sum(p.numel() for p in s_pose.parameters())
    print(f"SegNet student: {seg_p:,} params | PoseNet student: {pose_p:,} params")

    # Targets
    from submissions.phase2.inflate import load_targets
    seg_np, pose_np, colors_np = load_targets(Path('submissions/phase2/0'))
    seg_all = torch.from_numpy(seg_np).long().to(device)
    pose_all = torch.from_numpy(pose_np.copy()).float().to(device)
    colors = torch.from_numpy(colors_np.copy()).float().to(device)

    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4  # Small — 4 models loaded + grad tensors eat VRAM
    starts = [0, 64, 128, 256, 400][:num_batches]
    res = {'ss': [], 'sp': [], 'ts': [], 'tp': []}

    for bi, st in enumerate(starts):
        end = min(st + bs, len(seg_all))
        tgt_s = seg_all[st:end]
        tgt_p = pose_all[st:end]
        B = tgt_s.shape[0]

        # Initialize from ideal colors
        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1), keepdim=True).expand_as(init).clone()
        f0 = f0.requires_grad_(True)

        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lrs = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters-1, 1)))
               for i in range(iters)]

        print(f"  Batch {bi+1}/{num_batches}: optimizing {B} frames x {iters} iters...",
              end='', flush=True)
        for it in range(iters):
            for pg in opt.param_groups:
                pg['lr'] = lrs[it]
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
                # Input diversity: add noise (decays over iterations)
                noise_std = 15.0 * (1.0 - it / iters)  # 15 -> 0
                f1_noisy = f1 + torch.randn_like(f1) * noise_std
                f0_noisy = f0 + torch.randn_like(f0) * noise_std

                seg_l = margin_loss(s_seg(f1_noisy), tgt_s, 5.0)  # margin 0.1->5.0
                both = torch.stack([f0_noisy, f1_noisy], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(s_pose(pn_in)['pose'], tgt_p)
                total = 120.0 * seg_l + 2.0 * pose_l  # pose weight 0.5->2.0

            total.backward()
            opt.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)

            if (it + 1) % 50 == 0:
                print(f" {it+1}", end='', flush=True)
        print(" done", flush=True)

        # Evaluate
        with torch.no_grad():
            # Student assessment
            sp = s_seg(f1.data).argmax(1)
            res['ss'].extend((sp != tgt_s).float().mean((1,2)).cpu().tolist())
            sb = torch.stack([f0.data, f1.data], dim=1)
            spo = s_pose(posenet_preprocess_diff(sb))['pose']
            res['sp'].extend((spo - tgt_p).pow(2).mean(1).cpu().tolist())

            # Teacher assessment (with resolution round-trip)
            f0u = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                align_corners=False).clamp(0, 255).round().byte().float()
            f1u = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                align_corners=False).clamp(0, 255).round().byte().float()

            ts_in = F.interpolate(f1u, (mH, mW), mode='bilinear')
            res['ts'].extend(
                (t_seg(ts_in).argmax(1) != tgt_s).float().mean((1,2)).cpu().tolist())

            tp_pair = F.interpolate(
                torch.stack([f0u, f1u], 1).reshape(-1, 3, Hc, Wc),
                (mH, mW), mode='bilinear'
            ).reshape(B, 2, 3, mH, mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:, :6]
            res['tp'].extend((tpo - tgt_p).pow(2).mean(1).cpu().tolist())

            print(f"  Batch {bi+1}/{num_batches} (pairs {st}-{end-1}): "
                  f"Stu seg={np.mean(res['ss'][-B:]):.6f} pose={np.mean(res['sp'][-B:]):.6f} | "
                  f"Tea seg={np.mean(res['ts'][-B:]):.6f} pose={np.mean(res['tp'][-B:]):.6f}")

        del f0, f1, opt
        torch.cuda.empty_cache()

    # Summary
    ats, atp = np.mean(res['ts']), np.mean(res['tp'])
    ass_, asp = np.mean(res['ss']), np.mean(res['sp'])

    print(f"\n{'='*60}")
    print(f"TRANSFER RESULTS:")
    print(f"{'='*60}")
    print(f"  Student: seg_dist={ass_:.6f}  pose_mse={asp:.6f}")
    print(f"  Teacher: seg_dist={ats:.6f}  pose_mse={atp:.6f}")
    print(f"  Score components (teacher-evaluated):")
    print(f"    100 * seg_dist     = {100*ats:.4f}")
    print(f"    sqrt(10*pose_dist) = {math.sqrt(10*atp):.4f}")
    dist = 100 * ats + math.sqrt(10 * atp)
    print(f"    Distortion total   = {dist:.4f}")

    tot = seg_p + pose_p
    print(f"\n  Model sizes: seg={seg_p:,} + pose={pose_p:,} = {tot:,} params")
    for nb, lab in [(tot * 2, 'f16'), (tot, 'int8')]:
        arc = 313_747 + nb
        rate = arc / 37_545_489
        score = dist + 25 * rate
        print(f"    {lab}: archive={arc/1024:.1f} KB, 25*rate={25*rate:.4f}, "
              f"score={score:.4f}")

    go = ats < 0.01 and atp < 0.1
    print(f"\n  {'GO' if go else 'NO-GO'}")
    if not go:
        if ats >= 0.01:
            print(f"    SegNet transfer too poor ({ats:.4f} >= 0.01)")
        if atp >= 0.1:
            print(f"    PoseNet transfer too poor ({atp:.4f} >= 0.1)")
    return go, ats, atp


# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all',
                        choices=['0', '1', '1a', '1b', '2', 'all'])
    parser.add_argument('--video', type=Path, default=Path('videos/0.mkv'))
    parser.add_argument('--seg-ch', type=int, default=48)
    parser.add_argument('--pose-ch', type=int, default=16)
    parser.add_argument('--seg-epochs', type=int, default=200)
    parser.add_argument('--pose-epochs', type=int, default=300)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: seg_ch={args.seg_ch} pose_ch={args.pose_ch} "
          f"seg_epochs={args.seg_epochs} pose_epochs={args.pose_epochs}")

    if args.phase in ('0', 'all'):
        extract_teacher_data(args.video, device)

    if args.phase in ('1', '1a', 'all'):
        train_segnet_student(device, seg_base_ch=args.seg_ch,
                             epochs=args.seg_epochs)

    if args.phase in ('1', '1b', 'all'):
        train_posenet_student(device, pose_base_ch=args.pose_ch,
                              epochs=args.pose_epochs)

    if args.phase in ('2', 'all'):
        validate_transfer(device, seg_base_ch=args.seg_ch,
                          pose_base_ch=args.pose_ch)


if __name__ == '__main__':
    main()
