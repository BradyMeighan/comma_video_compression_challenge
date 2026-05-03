#!/usr/bin/env python
"""
GaborNet Neural Video Codec: store GaborNet coefficients instead of compressed video.

The idea:
- GaborNet maps (frame_id, x, y) → RGB pixel values
- Train OFFLINE using real SegNet/PoseNet to optimize: generate frames that fool the teachers
- Store just the GaborNet weights (~200KB) + targets (~300KB) = 500KB archive
- At inflate time: run GaborNet forward to generate all frames. NO adversarial decode needed.

This completely sidesteps the model compression problem because:
- The teacher models are used OFFLINE (not in archive)
- The GaborNet IS the compressed video
- No gradient transfer issue because we don't do adversarial decode at inflate time

GaborNet (Multiplicative Filter Network) implemented from scratch based on:
Fathony et al., "Multiplicative Filter Networks" (ICLR 2021)
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


# ═══════════════════════════════════════════════════════════════════════
# GaborNet / Multiplicative Filter Network
# ═══════════════════════════════════════════════════════════════════════

class GaborLayer(nn.Module):
    """Gabor wavelet filter applied to input coordinates.

    Output = linear(x) * sin(freq @ x + phase)
    The multiplicative structure creates exponentially many basis functions.
    """
    def __init__(self, in_features, out_features, alpha=6.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.freq = nn.Linear(in_features, out_features)
        self.phase = nn.Parameter(torch.randn(out_features) * 0.1)
        self.alpha = alpha

        # Initialize frequencies log-uniformly for multi-scale coverage
        with torch.no_grad():
            nn.init.uniform_(self.freq.weight, -alpha, alpha)
            nn.init.zeros_(self.freq.bias)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        # Gabor: envelope * oscillation
        envelope = self.linear(x)  # learned spatial envelope
        oscillation = torch.sin(self.freq(x) + self.phase)  # sinusoidal carrier
        return envelope * oscillation


class GaborNet(nn.Module):
    """Multiplicative Filter Network for video representation.

    Maps (frame_id_embedding, x, y) → RGB pixel values.
    Each layer multiplies Gabor features rather than composing them,
    creating an exponential number of basis functions from linear parameters.
    """
    def __init__(self, n_frames=600, coord_dim=2, latent_dim=32,
                 hidden=128, n_layers=3, out_dim=3):
        super().__init__()
        self.latent = nn.Embedding(n_frames, latent_dim)
        nn.init.normal_(self.latent.weight, 0, 0.1)

        in_dim = coord_dim + latent_dim

        # Gabor filter layers (multiplicative)
        self.gabor_layers = nn.ModuleList()
        self.gabor_layers.append(GaborLayer(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.gabor_layers.append(GaborLayer(in_dim, hidden))

        # Linear mixing layers between Gabor filters
        self.linears = nn.ModuleList()
        for _ in range(n_layers - 1):
            self.linears.append(nn.Linear(hidden, hidden))

        # Output head
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid()  # output in [0, 1], scale to [0, 255]
        )

    def forward_pixels(self, coords, frame_ids):
        """Generate pixels for given coordinates and frame IDs.

        coords: (N, 2) normalized [0,1] coordinates
        frame_ids: (N,) integer frame indices
        Returns: (N, 3) RGB values in [0, 255]
        """
        z = self.latent(frame_ids)  # (N, latent_dim)
        x = torch.cat([coords, z], dim=-1)  # (N, in_dim)

        # Multiplicative Gabor layers
        h = self.gabor_layers[0](x)
        for i in range(len(self.linears)):
            gabor_out = self.gabor_layers[i + 1](x)
            h = self.linears[i](h) * gabor_out  # multiplicative!

        rgb = self.output(h) * 255.0
        return rgb

    def render_frame(self, frame_id, H, W, device, batch_size=65536):
        """Render a complete frame."""
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)
        fids = torch.full((H * W,), frame_id, dtype=torch.long, device=device)

        # Process in chunks to avoid OOM
        pixels = []
        for i in range(0, len(coords), batch_size):
            c = coords[i:i+batch_size]
            f = fids[i:i+batch_size]
            pixels.append(self.forward_pixels(c, f))

        return torch.cat(pixels, dim=0).reshape(H, W, 3).permute(2, 0, 1)  # (3, H, W)


# ═══════════════════════════════════════════════════════════════════════
# Training: optimize GaborNet to fool SegNet + PoseNet
# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")

    # Load teachers (used OFFLINE for training only, NOT in archive)
    print("Loading teachers (offline training only)...")
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in segnet.parameters(): p.requires_grad_(False)
    for p in posenet.parameters(): p.requires_grad_(False)

    # Load targets
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)  # (600, H, W)
    pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    del seg_logits

    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512
    Wc, Hc = camera_size  # 1164, 874

    # Create GaborNet
    for hidden, n_layers, latent_dim in [(128, 3, 32), (192, 4, 48)]:
        gnet = GaborNet(n_frames=600, coord_dim=2, latent_dim=latent_dim,
                        hidden=hidden, n_layers=n_layers, out_dim=3).to(device)
        n_params = sum(p.numel() for p in gnet.parameters())
        print(f"\n{'='*70}")
        print(f"GaborNet h={hidden} L={n_layers} z={latent_dim}: {n_params:,} params ({n_params*2/1024:.0f}KB FP16)")
        print(f"{'='*70}\n")

        opt = torch.optim.Adam(gnet.parameters(), lr=3e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 30, eta_min=1e-5)

        # Pre-build coordinate grid at model resolution
        ys = torch.linspace(0, 1, mH, device=device)
        xs = torch.linspace(0, 1, mW, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords_flat = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)

        t0 = time.time()
        frames_per_batch = 4  # render 4 frames at a time

        for epoch in range(30):
            gnet.train()
            perm = torch.randperm(600)
            ep_seg_loss = ep_pose_loss = 0
            n_b = 0

            for bi in range(0, 600, frames_per_batch):
                frame_ids = perm[bi:bi+frames_per_batch]
                B = len(frame_ids)

                # Render frames at model resolution (384x512)
                rendered = []
                for fid in frame_ids:
                    fid_expanded = torch.full((mH * mW,), fid.item(), dtype=torch.long, device=device)
                    pixels = gnet.forward_pixels(coords_flat, fid_expanded)
                    rendered.append(pixels.reshape(mH, mW, 3).permute(2, 0, 1))
                frames = torch.stack(rendered)  # (B, 3, mH, mW)

                # SegNet loss: margin loss against target segmentation
                tgt_s = teacher_argmax[frame_ids]
                seg_out = segnet(frames)
                seg_loss = margin_loss(seg_out, tgt_s, margin=3.0)

                # PoseNet loss: need frame pairs
                # Use consecutive even/odd pairs from the batch
                if B >= 2:
                    f0_frames = frames[0::2] if B > 1 else frames[:1]
                    f1_frames = frames[1::2] if B > 1 else frames[:1]
                    n_pairs = min(f0_frames.shape[0], f1_frames.shape[0])
                    if n_pairs > 0:
                        # Upscale for PoseNet preprocessing
                        both = torch.stack([f0_frames[:n_pairs], f1_frames[:n_pairs]], dim=1)
                        from train_distill import posenet_preprocess_diff
                        pn_in = posenet_preprocess_diff(both)
                        pose_pred = posenet(pn_in)['pose'][:, :6]
                        pair_ids = frame_ids[0::2][:n_pairs]
                        pose_loss = F.smooth_l1_loss(pose_pred, pose_targets[pair_ids])
                    else:
                        pose_loss = seg_loss.new_zeros(())
                else:
                    pose_loss = seg_loss.new_zeros(())

                total = 100.0 * seg_loss + 1.0 * pose_loss
                opt.zero_grad()
                total.backward()
                opt.step()

                ep_seg_loss += seg_loss.item()
                ep_pose_loss += pose_loss.item()
                n_b += 1

            sched.step()

            # Evaluate: render a few frames and check with teacher
            gnet.eval()
            if epoch % 3 == 0 or epoch == 29:
                correct = total_px = 0
                with torch.no_grad():
                    for fid in range(0, 600, 30):  # sample 20 frames
                        frame = gnet.render_frame(fid, mH, mW, device)
                        pred = segnet(frame.unsqueeze(0)).argmax(1)
                        tgt = teacher_argmax[fid:fid+1]
                        correct += (pred == tgt).sum().item()
                        total_px += tgt.numel()
                acc = correct / total_px

                elapsed = time.time() - t0
                print(f"  ep {epoch:2d}: seg_loss={ep_seg_loss/n_b:.4f} "
                      f"pose_loss={ep_pose_loss/n_b:.6f} "
                      f"teacher_acc={acc*100:.2f}% ({elapsed:.0f}s)", flush=True)

        # Final evaluation: full teacher check on all 600 frames
        print("\n  Final evaluation on all 600 frames...")
        gnet.eval()
        all_seg_dist = []
        with torch.no_grad():
            for fid in range(600):
                frame = gnet.render_frame(fid, mH, mW, device)
                pred = segnet(frame.unsqueeze(0)).argmax(1)
                tgt = teacher_argmax[fid:fid+1]
                dist = (pred != tgt).float().mean().item()
                all_seg_dist.append(dist)

        avg_seg = np.mean(all_seg_dist)
        s_seg = 100 * avg_seg

        # Quick pose check on a few pairs
        pose_mses = []
        with torch.no_grad():
            for pair_id in range(0, 600, 30):
                f0 = gnet.render_frame(pair_id, mH, mW, device)
                f1 = gnet.render_frame(min(pair_id+1, 599), mH, mW, device)
                # Upscale to camera res
                f0u = F.interpolate(f0.unsqueeze(0), (Hc, Wc), mode='bicubic',
                                    align_corners=False).clamp(0, 255)
                f1u = F.interpolate(f1.unsqueeze(0), (Hc, Wc), mode='bicubic',
                                    align_corners=False).clamp(0, 255)
                both = torch.stack([f0u, f1u], dim=1)
                from train_distill import posenet_preprocess_diff
                pn_in = posenet_preprocess_diff(both)
                pose_pred = posenet(pn_in)['pose'][:, :6]
                pose_mse = (pose_pred - pose_targets[pair_id:pair_id+1]).pow(2).mean().item()
                pose_mses.append(pose_mse)

        avg_pose = np.mean(pose_mses)
        s_pose = math.sqrt(10 * avg_pose)
        dist = s_seg + s_pose

        print(f"\n  RESULT: seg_dist={avg_seg:.6f} ({(1-avg_seg)*100:.2f}% agreement)")
        print(f"  pose_mse={avg_pose:.6f}")
        print(f"  100*seg={s_seg:.4f}  sqrt(10*p)={s_pose:.4f}  distortion={dist:.4f}")

        archive_kb = n_params * 2 / 1024 + 300  # FP16 + targets
        rate = (archive_kb * 1024) / 37_545_489
        total_score = dist + 25 * rate
        print(f"  Archive: {archive_kb:.0f}KB  25*rate={25*rate:.3f}  TOTAL={total_score:.3f}")
        print(f"  Leader: 1.95")

        del gnet; torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
