#!/usr/bin/env python
"""
SIREN v2: coordinate-based segmentation network for adversarial decode.

The key challenge: adversarial decode optimizes PIXEL VALUES via gradients.
SIREN maps (x, y, frame_id) -> logits, not (pixel_colors) -> logits.

Solution: SIREN acts as a DIFFERENTIABLE TARGET ORACLE during adversarial decode.
The adversarial decode loop becomes:

    for each iteration:
        # SIREN provides the target segmentation (replaces stored seg maps)
        target = siren(coordinates, frame_id).argmax(1)
        # Color-distance proxy provides gradients w.r.t. pixels
        proxy_logits = -(pixel - ideal_color[c])^2
        loss = margin_loss(proxy_logits, target)
        loss.backward()  # gradients flow to pixels via color proxy

Wait — this is just the color proxy again, which gave 3.4% error. SIREN doesn't help here.

ALTERNATIVE: SIREN takes pixel colors AS INPUT (like a tiny CNN) instead of coordinates.
This makes it a standard image-to-segmentation model that's just very small.

Let's test BOTH:
A) Coordinate SIREN (pure memorization, acts as compressed target storage)
B) Image SIREN (takes pixel colors, provides gradients like SegNet)
"""
import sys, time, math
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss

DISTILL_DIR = Path('distill_data')
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
# Image SIREN: takes (B, 3, H, W) pixel colors -> (B, 5, H, W) logits
# Like a tiny SegNet replacement that's natively 300KB
# Uses sin() activations for smooth Jacobians
# ═══════════════════════════════════════════════════════════════════════

class SirenConv(nn.Module):
    """Conv2d with sinusoidal activation."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        # SIREN initialization
        fan_in = in_ch * kernel_size * kernel_size
        with torch.no_grad():
            if is_first:
                self.conv.weight.uniform_(-1/fan_in, 1/fan_in)
            else:
                self.conv.weight.uniform_(-math.sqrt(6/fan_in)/omega,
                                           math.sqrt(6/fan_in)/omega)
            self.conv.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega * self.conv(x))


class ImageSIREN(nn.Module):
    """Image-to-segmentation SIREN. Takes pixel colors, outputs class logits.

    Uses sinusoidal activations throughout for C-infinity smooth Jacobians.
    Includes Fourier positional encoding for sharp spatial boundaries.
    """
    def __init__(self, in_ch=3, n_classes=5, hidden=64, n_layers=4,
                 omega=30.0, fourier_L=6):
        super().__init__()
        self.fourier_L = fourier_L
        fourier_ch = 2 + 4 * fourier_L  # 26 for L=6
        total_in = in_ch + fourier_ch  # 29

        layers = []
        layers.append(SirenConv(total_in, hidden, 3, 1, 1, omega, is_first=True))
        for _ in range(n_layers - 1):
            layers.append(SirenConv(hidden, hidden, 3, 1, 1, omega))
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(hidden, n_classes, 1)
        nn.init.zeros_(self.head.bias)

        self._fourier = None

    def _make_fourier(self, H, W, device):
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        ch = [xx.unsqueeze(0), yy.unsqueeze(0)]
        for l in range(self.fourier_L):
            freq = math.pi * (2 ** l)
            ch += [torch.sin(freq*xx).unsqueeze(0), torch.cos(freq*xx).unsqueeze(0),
                   torch.sin(freq*yy).unsqueeze(0), torch.cos(freq*yy).unsqueeze(0)]
        return torch.cat(ch, dim=0)

    def forward(self, x):
        B, C, H, W = x.shape
        if self._fourier is None or self._fourier.device != x.device or \
                self._fourier.shape[-2] != H or self._fourier.shape[-1] != W:
            self._fourier = self._make_fourier(H, W, x.device)
        # Normalize pixel values to [-1, 1] (CRITICAL for SIREN)
        x_norm = x / 127.5 - 1.0
        fourier = self._fourier.unsqueeze(0).expand(B, -1, -1, -1)
        x_norm = torch.cat([x_norm, fourier], dim=1)
        return self.head(self.backbone(x_norm))


def check_acc(model, seg_inputs, teacher_argmax, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(seg_inputs), 32):
            x = seg_inputs[i:i+32].to(device)
            ta = teacher_argmax[i:i+32].to(device)
            correct += (model(x).argmax(1) == ta).sum().item()
            total += ta.numel()
    return correct / total


def quick_adv_test(model, t_seg, t_pose, ta, pt, device, iters=100):
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    results = {'ts': [], 'tp': []}
    for st in [0, 200, 400]:
        end = min(st+4, 600); tgt_s = ta[st:end]; tgt_p = pt[st:end]; B = tgt_s.shape[0]
        init = colors[tgt_s].permute(0,3,1,2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1),keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0,f1], lr=1.2, weight_decay=0)
        lr_s = [0.06+0.57*(1+math.cos(math.pi*i/max(iters-1,1))) for i in range(iters)]
        for it in range(iters):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            p = it/max(iters-1,1)
            seg_l = margin_loss(model(f1), tgt_s, 0.1 if p<0.5 else 5.0)
            if p >= 0.3:
                both = torch.stack([f0,f1],dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:,:6], tgt_p)
                total = 120*seg_l + 0.2*pose_l
            else: total = 120*seg_l
            total.backward(); opt.step()
            with torch.no_grad(): f0.data.clamp_(0,255); f1.data.clamp_(0,255)
        with torch.no_grad():
            f1u = F.interpolate(f1.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
            f0u = F.interpolate(f0.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
            ts_in = F.interpolate(f1u,(mH,mW),mode='bilinear')
            results['ts'].extend((t_seg(ts_in).argmax(1)!=tgt_s).float().mean((1,2)).cpu().tolist())
            tp_pair = F.interpolate(torch.stack([f0u,f1u],1).reshape(-1,3,Hc,Wc),(mH,mW),mode='bilinear').reshape(B,2,3,mH,mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:,:6]
            results['tp'].extend((tpo-tgt_p).pow(2).mean(1).cpu().tolist())
        del f0,f1,opt; torch.cuda.empty_cache()
    seg_d, pose_d = np.mean(results['ts']), np.mean(results['tp'])
    return seg_d, pose_d, 100*seg_d + math.sqrt(10*pose_d)


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")

    # Load teachers + data
    print("Loading data...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)

    # Load trajectory data
    traj_frames = torch.load(DISTILL_DIR / 'trajectory/traj_frames.pt', weights_only=True)
    traj_logits = torch.load(DISTILL_DIR / 'trajectory/traj_logits.pt', weights_only=True)
    traj_argmax = traj_logits.float().argmax(1)
    print(f"  {seg_inputs.shape[0]} original + {traj_frames.shape[0]} trajectory frames")

    # ═══ Test: Image SIREN ═══
    for hidden in [64, 96, 128]:
        n_layers = 4
        model = ImageSIREN(in_ch=3, n_classes=5, hidden=hidden,
                           n_layers=n_layers, omega=10.0, fourier_L=6).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n{'='*70}")
        print(f"ImageSIREN hidden={hidden}, {n_layers} layers: {n_params:,} params ({n_params*2/1024:.0f}KB FP16)")
        print(f"{'='*70}")

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60, eta_min=1e-5)
        T_kd = 6.0
        N = seg_inputs.shape[0]
        N_traj = traj_frames.shape[0]
        best_acc = 0
        t0 = time.time()

        for ep in range(60):
            model.train()
            ep_loss = n_b = 0

            # Original frames
            perm = torch.randperm(N)
            for i in range(0, N, 8):
                idx = perm[i:i+8]
                x = seg_inputs[idx].to(device)
                tl = seg_logits[idx].to(device)
                ta = teacher_argmax[idx].to(device)
                out = model(x)
                kd = F.kl_div(F.log_softmax(out/T_kd,1), F.softmax(tl/T_kd,1),
                              reduction='batchmean') * T_kd**2
                ce = F.cross_entropy(out, ta)
                mse = F.mse_loss(out, tl)
                loss = 0.4*kd + 0.3*ce + 0.3*mse
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); n_b += 1

            # Trajectory frames (subsample)
            tp = torch.randperm(N_traj)[:N]
            for i in range(0, len(tp), 8):
                idx = tp[i:i+8]
                x = traj_frames[idx].float().to(device)
                tl = traj_logits[idx].float().to(device)
                ta = traj_argmax[idx].to(device)
                out = model(x)
                kd = F.kl_div(F.log_softmax(out/T_kd,1), F.softmax(tl/T_kd,1),
                              reduction='batchmean') * T_kd**2
                ce = F.cross_entropy(out, ta)
                loss = 0.5*kd + 0.5*ce
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); n_b += 1

            sched.step()
            model.eval()
            acc = check_acc(model, seg_inputs, teacher_argmax, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if ep % 5 == 0 or ep == 59:
                print(f"  ep {ep:2d}: loss={ep_loss/n_b:.1f} acc={acc*100:.2f}% "
                      f"best={best_acc*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)

        # Adversarial decode test
        print(f"\n  Running adversarial decode test (hidden={hidden})...")
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        for p in model.parameters(): p.requires_grad_(False)
        seg_d, pose_d, dist = quick_adv_test(
            model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
        print(f"  => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")
        print(f"  Size: {n_params*2/1024:.0f}KB FP16 | Leader: 1.95")

        del model; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
