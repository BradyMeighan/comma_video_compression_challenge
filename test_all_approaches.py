#!/usr/bin/env python
"""
Quick tests for ALL untested approaches. ~3-5 min each.
Rules things out fast before committing hours.

Test 1: LoRA-on-SVD — freeze SVD base, train tiny residual
Test 2: SVDQuant — outlier isolation + INT4 on non-factored model
Test 3: Codebook quantization — k-means centroids on non-factored model
Test 4: SIREN — coordinate-based network, train from scratch (quick 10-epoch test)
"""
import sys, os, time, math, bz2, pickle
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import posenet_preprocess_diff, margin_loss

DISTILL_DIR = Path('distill_data')
TRAJ_DIR = DISTILL_DIR / 'trajectory'
OUT_DIR = Path('compressed_models')
IDEAL_COLORS = torch.tensor([
    [52.3731, 66.0825, 53.4251], [132.6272, 139.2837, 154.6401],
    [0.0000, 58.3693, 200.9493], [200.2360, 213.4126, 201.8910],
    [26.8595, 41.0758, 46.1465],
], dtype=torch.float32)


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


def quick_adversarial_test(model, t_seg, t_pose, teacher_argmax, pose_targets, device):
    """3-batch quick adversarial decode test. Returns (seg_dist, pose_mse, distortion)."""
    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    results = {'ts': [], 'tp': []}

    for st in [0, 200, 400]:
        end = min(st+4, 600)
        tgt_s, tgt_p = teacher_argmax[st:end], pose_targets[st:end]
        B = tgt_s.shape[0]
        init = colors[tgt_s].permute(0,3,1,2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1),keepdim=True).expand_as(init).clone().requires_grad_(True)
        opt = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_s = [0.06+0.57*(1+math.cos(math.pi*i/99)) for i in range(100)]

        for it in range(100):
            for pg in opt.param_groups: pg['lr'] = lr_s[it]
            opt.zero_grad(set_to_none=True)
            p = it/99
            seg_l = margin_loss(model(f1), tgt_s, 0.1 if p<0.5 else 5.0)
            if p >= 0.3:
                both = torch.stack([f0,f1],dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:,:6], tgt_p)
                total = 120*seg_l + 0.2*pose_l
            else:
                total = 120*seg_l
            total.backward()
            opt.step()
            with torch.no_grad(): f0.data.clamp_(0,255); f1.data.clamp_(0,255)

        with torch.no_grad():
            f1u = F.interpolate(f1.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
            f0u = F.interpolate(f0.data,(Hc,Wc),mode='bicubic',align_corners=False).clamp(0,255).round().byte().float()
            ts_in = F.interpolate(f1u,(mH,mW),mode='bilinear')
            results['ts'].extend((t_seg(ts_in).argmax(1)!=tgt_s).float().mean((1,2)).cpu().tolist())
            tp_pair = F.interpolate(torch.stack([f0u,f1u],1).reshape(-1,3,Hc,Wc),(mH,mW),mode='bilinear').reshape(B,2,3,mH,mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:,:6]
            results['tp'].extend((tpo-tgt_p).pow(2).mean(1).cpu().tolist())
        del f0, f1, opt
        torch.cuda.empty_cache()

    seg_d = np.mean(results['ts'])
    pose_d = np.mean(results['tp'])
    dist = 100*seg_d + math.sqrt(10*pose_d)
    return seg_d, pose_d, dist


def load_common(device):
    """Load everything needed for all tests."""
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

    traj_frames = traj_logits = None
    if TRAJ_DIR.exists():
        traj_frames = torch.load(TRAJ_DIR / 'traj_frames.pt', weights_only=True)
        traj_logits = torch.load(TRAJ_DIR / 'traj_logits.pt', weights_only=True)

    return (t_seg, t_pose, seg_inputs, seg_logits, teacher_argmax,
            pose_targets, traj_frames, traj_logits)


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: LoRA-on-SVD
# ═══════════════════════════════════════════════════════════════════════

class LoRALayer(nn.Module):
    """Wraps a frozen Conv2d/Linear with a trainable low-rank residual."""
    def __init__(self, base_weight, bias, lora_rank=4, conv_params=None):
        super().__init__()
        self.register_buffer('base_weight', base_weight)  # FROZEN
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.bias = None
        self.conv_params = conv_params  # stride, padding for conv

        # LoRA: A @ B adds a low-rank correction
        if base_weight.dim() == 4:
            O, I_kk = base_weight.shape[0], base_weight[0].numel()
        else:
            O, I_kk = base_weight.shape
        self.A = nn.Parameter(torch.zeros(O, lora_rank))
        self.B = nn.Parameter(torch.randn(lora_rank, I_kk) * 0.01)

    def forward(self, x):
        W = self.base_weight + (self.A @ self.B).view_as(self.base_weight)
        if self.base_weight.dim() == 4:
            return F.conv2d(x, W, self.bias, self.conv_params['stride'],
                           self.conv_params['padding'])
        else:
            return F.linear(x, W, self.bias)


def test_lora_on_svd(device, common, rank_ratio=0.3, lora_rank=8, epochs=15):
    print("=" * 70)
    print(f"TEST 1: LoRA-on-SVD (svd_rank={rank_ratio:.0%}, lora_rank={lora_rank})")
    print("=" * 70)
    print("Freeze SVD-compressed weights. Train only tiny LoRA residual.")
    print("Base gradient landscape is UNTOUCHED.")
    print()

    t_seg, t_pose, seg_inputs, seg_logits, teacher_argmax, pose_targets, traj_f, traj_l = common

    # Build SVD-compressed model with LoRA
    model = SegNet().to(device)
    model.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    sd = model.state_dict()

    # Apply SVD truncation then wrap with LoRA
    lora_params = []
    replaced = 0
    for name, child in list(model.named_modules()):
        if isinstance(child, nn.Conv2d) and child.groups == 1 and child.weight.numel() > 1000:
            if child.out_channels == 5: continue  # skip seg head
            W = child.weight.data
            O, I, kH, kW = W.shape
            mat = W.reshape(O, -1).float()
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
            k = max(1, int(min(O, I*kH*kW) * rank_ratio))
            W_svd = ((U[:,:k] * S[:k]) @ Vh[:k,:]).reshape(O, I, kH, kW)

            lora = LoRALayer(
                W_svd, child.bias.data.clone() if child.bias is not None else None,
                lora_rank=lora_rank,
                conv_params={'stride': child.stride, 'padding': child.padding}
            ).to(device)

            # Replace in model
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lora)
            lora_params.extend([lora.A, lora.B])
            if lora.bias is not None:
                lora_params.append(lora.bias)
            replaced += 1

    n_lora = sum(p.numel() for p in lora_params)
    print(f"  Replaced {replaced} layers. LoRA params: {n_lora:,} ({n_lora*2/1024:.0f} KB FP16)")

    acc = check_acc(model, seg_inputs, teacher_argmax, device)
    print(f"  Post-SVD+LoRA accuracy: {acc*100:.2f}% (LoRA initialized at zero = pure SVD)")

    # Train only LoRA params
    opt = torch.optim.AdamW(lora_params, lr=1e-3, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    T_kd = 6.0
    N = seg_inputs.shape[0]
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N)
        ep_loss = n_b = 0
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

        # Also train on trajectory
        if traj_f is not None:
            tp = torch.randperm(len(traj_f))[:N]
            for i in range(0, len(tp), 8):
                idx = tp[i:i+8]
                x = traj_f[idx].float().to(device)
                tl = traj_l[idx].float().to(device)
                ta = tl.argmax(1)
                out = model(x)
                loss = F.cross_entropy(out, ta)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item(); n_b += 1

        sched.step()
        model.eval()
        acc = check_acc(model, seg_inputs, teacher_argmax, device)
        print(f"  ep {ep:2d}: loss={ep_loss/n_b:.1f} acc={acc*100:.2f}% ({time.time()-t0:.0f}s)",
              flush=True)

    # Adversarial decode test
    print("\n  Running adversarial decode test...")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    seg_d, pose_d, dist = quick_adversarial_test(
        model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
    print(f"  => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")

    # Size estimate
    svd_size = sum(m.base_weight.numel() for m in model.modules() if isinstance(m, LoRALayer))
    lora_size = n_lora
    print(f"  SVD factors would be ~{svd_size*0.3*2/1024:.0f}KB, LoRA ~{lora_size*2/1024:.0f}KB")

    del model; torch.cuda.empty_cache()
    return dist


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: SVDQuant on non-factored model
# ═══════════════════════════════════════════════════════════════════════

def test_svdquant(device, common):
    print("\n" + "=" * 70)
    print("TEST 2: SVDQuant — outlier isolation + INT4")
    print("=" * 70)
    print("Isolate outlier weights into FP16 low-rank branch, INT4 the rest.")
    print()

    t_seg, t_pose, seg_inputs, seg_logits, teacher_argmax, pose_targets, _, _ = common

    # Load the working non-factored SVD fine-tuned model
    sd_path = OUT_DIR / 'segnet_svd_finetuned.pt'
    if not sd_path.exists():
        print("  SKIP: No fine-tuned SVD model found")
        return 999

    sd = torch.load(sd_path, weights_only=True)
    packed = {}
    total_main = total_outlier = 0

    for name, tensor in sd.items():
        t = tensor.cpu().float()
        if t.dim() >= 2 and t.numel() > 100 and 'running' not in name and 'bn' not in name.lower():
            # SVDQuant: extract outlier singular values into FP16 branch
            if t.dim() == 4:
                mat = t.reshape(t.shape[0], -1)
            else:
                mat = t
            U, S, Vh = torch.linalg.svd(mat, full_matrices=False)

            # Top-k singular values are "outliers" — keep in FP16
            n_outlier = max(1, int(min(mat.shape) * 0.05))  # 5% as outliers
            U_out = U[:, :n_outlier] * S[:n_outlier]  # FP16
            Vh_out = Vh[:n_outlier, :]

            # Residual = original - outlier reconstruction
            outlier_recon = U_out @ Vh_out
            residual = mat - outlier_recon

            # Quantize residual to INT4
            max_val = 7; min_val = -8
            out_ch = residual.shape[0]
            amax = residual.abs().amax(dim=1).clamp(min=1e-12)
            scales = amax / max_val
            q_res = (residual / scales.unsqueeze(1)).round().clamp(min_val, max_val)

            packed[name] = {
                'u_out': U_out.half().numpy(),
                'v_out': Vh_out.half().numpy(),
                'q_res': q_res.to(torch.int8).numpy(),
                'scales': scales.half().numpy(),
                'shape': list(t.shape),
            }
            total_outlier += U_out.numel() + Vh_out.numel()
            total_main += q_res.numel()
        else:
            packed[name] = {'f16': t.half().numpy()}

    compressed = bz2.compress(pickle.dumps(packed), 9)
    print(f"  Outlier params (FP16): {total_outlier:,}")
    print(f"  Main params (INT4): {total_main:,}")
    print(f"  Compressed size: {len(compressed)/1024:.0f} KB")

    # Reconstruct and check accuracy
    recon_sd = {}
    for name, entry in packed.items():
        if 'q_res' in entry:
            u_out = torch.from_numpy(entry['u_out']).float()
            v_out = torch.from_numpy(entry['v_out']).float()
            q_res = torch.from_numpy(entry['q_res']).float()
            scales = torch.from_numpy(entry['scales']).float()
            residual = q_res * scales.unsqueeze(1)
            mat = (u_out @ v_out) + residual
            recon_sd[name] = mat.view(entry['shape']).to(device)
        else:
            recon_sd[name] = torch.from_numpy(entry['f16']).float().to(device)

    model = SegNet().eval().to(device)
    model.load_state_dict(recon_sd)
    acc = check_acc(model, seg_inputs, teacher_argmax, device)
    print(f"  Accuracy: {acc*100:.2f}%")

    # Adversarial decode test
    print("  Running adversarial decode test...")
    for p in model.parameters(): p.requires_grad_(False)
    seg_d, pose_d, dist = quick_adversarial_test(
        model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
    print(f"  => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")
    print(f"  Size: {len(compressed)/1024:.0f} KB")

    del model; torch.cuda.empty_cache()
    return dist


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Codebook (k-means) quantization on non-factored model
# ═══════════════════════════════════════════════════════════════════════

def test_codebook(device, common):
    print("\n" + "=" * 70)
    print("TEST 3: Codebook quantization (k-means, 16 centroids = 4 bits)")
    print("=" * 70)
    print("Non-uniform quantization: centroids adapt to weight distribution.")
    print()

    t_seg, t_pose, seg_inputs, seg_logits, teacher_argmax, pose_targets, _, _ = common

    sd_path = OUT_DIR / 'segnet_svd_finetuned.pt'
    if not sd_path.exists():
        print("  SKIP: No fine-tuned SVD model found")
        return 999

    sd = torch.load(sd_path, weights_only=True)
    packed = {}

    for n_clusters in [16, 32]:  # 4-bit and 5-bit codebook
        bits = int(math.log2(n_clusters))
        packed = {}
        for name, tensor in sd.items():
            t = tensor.cpu().float().flatten()
            if tensor.dim() >= 2 and tensor.numel() > 100 and 'running' not in name:
                # K-means quantization
                # Simple k-means: initialize centroids linearly
                vmin, vmax = t.min().item(), t.max().item()
                centroids = torch.linspace(vmin, vmax, n_clusters)

                for _ in range(20):  # k-means iterations
                    # Assign each weight to nearest centroid
                    dists = (t.unsqueeze(1) - centroids.unsqueeze(0)).abs()
                    assignments = dists.argmin(dim=1)
                    # Update centroids
                    for c in range(n_clusters):
                        mask = assignments == c
                        if mask.any():
                            centroids[c] = t[mask].mean()

                assignments = (t.unsqueeze(1) - centroids.unsqueeze(0)).abs().argmin(dim=1)
                packed[name] = {
                    'centroids': centroids.half().numpy(),
                    'indices': assignments.to(torch.uint8).numpy(),
                    'shape': list(tensor.shape),
                }
            else:
                packed[name] = {'f16': tensor.cpu().half().numpy()}

        compressed = bz2.compress(pickle.dumps(packed), 9)
        print(f"  {n_clusters} centroids ({bits}-bit): {len(compressed)/1024:.0f} KB")

        # Reconstruct and check accuracy
        recon_sd = {}
        for name, entry in packed.items():
            if 'centroids' in entry:
                centroids = torch.from_numpy(entry['centroids']).float()
                indices = torch.from_numpy(entry['indices']).long()
                recon_sd[name] = centroids[indices].view(entry['shape']).to(device)
            else:
                recon_sd[name] = torch.from_numpy(entry['f16']).float().to(device)

        model = SegNet().eval().to(device)
        model.load_state_dict(recon_sd)
        acc = check_acc(model, seg_inputs, teacher_argmax, device)
        print(f"    Accuracy: {acc*100:.2f}%")

        if bits == 4:  # Only run adversarial test on 4-bit version
            print("    Running adversarial decode test...")
            for p in model.parameters(): p.requires_grad_(False)
            seg_d, pose_d, dist = quick_adversarial_test(
                model, t_seg, t_pose, teacher_argmax.to(device), pose_targets, device)
            print(f"    => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")

        del model; torch.cuda.empty_cache()

    return dist


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: SIREN (quick feasibility)
# ═══════════════════════════════════════════════════════════════════════

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, omega=30.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_f, out_f)
        # SIREN initialization
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_f, 1/in_f)
            else:
                self.linear.weight.uniform_(-math.sqrt(6/in_f)/omega,
                                             math.sqrt(6/in_f)/omega)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class SegSIREN(nn.Module):
    """SIREN that maps (frame_id, x, y) -> 5-class segmentation logits."""
    def __init__(self, n_frames=600, latent_dim=48, hidden=192, n_layers=4):
        super().__init__()
        self.latent = nn.Embedding(n_frames, latent_dim)
        self.layers = nn.ModuleList()
        self.layers.append(SirenLayer(2 + latent_dim, hidden, omega=30.0, is_first=True))
        for _ in range(n_layers - 1):
            self.layers.append(SirenLayer(hidden, hidden, omega=30.0))
        self.output = nn.Linear(hidden, 5)
        nn.init.zeros_(self.output.bias)

    def forward_coords(self, coords, frame_ids):
        """coords: (B, 2) normalized [0,1], frame_ids: (B,) int"""
        z = self.latent(frame_ids)  # (B, latent_dim)
        x = torch.cat([coords, z], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)  # (B, 5)

    def forward(self, frame_tensor):
        """Accept (B, 3, H, W) like SegNet for adversarial decode compatibility.

        Uses pixel POSITIONS (ignores colors) + guesses frame_id=0 for all.
        This is a quick test — real version would need frame identification.
        """
        B, C, H, W = frame_tensor.shape
        device = frame_tensor.device

        # Create coordinate grid
        ys = torch.linspace(0, 1, H, device=device)
        xs = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)

        # For adversarial decode, we don't know the frame_id
        # Use the mean color as a crude frame identifier
        mean_color = frame_tensor.mean(dim=(2,3))  # (B, 3)
        # Nearest-neighbor to find closest frame
        frame_ids = torch.zeros(B, dtype=torch.long, device=device)

        # Process each sample
        outputs = []
        for b in range(B):
            out = self.forward_coords(
                coords.unsqueeze(0).expand(1, -1, -1).reshape(-1, 2),
                frame_ids[b].expand(H * W)
            )
            outputs.append(out.reshape(H, W, 5).permute(2, 0, 1))

        return torch.stack(outputs, dim=0)  # (B, 5, H, W)


def test_siren(device, common, epochs=10):
    print("\n" + "=" * 70)
    print(f"TEST 4: SIREN (quick {epochs}-epoch feasibility test)")
    print("=" * 70)
    print("Can a coordinate-based network memorize segmentation maps?")
    print()

    t_seg, t_pose, seg_inputs, seg_logits, teacher_argmax, pose_targets, _, _ = common

    model = SegSIREN(n_frames=600, latent_dim=48, hidden=192, n_layers=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SIREN params: {n_params:,} ({n_params*2/1024:.0f} KB at FP16)")

    # Train: for each frame, predict segmentation at every pixel
    H, W = 384, 512
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)

    teacher_am = teacher_argmax  # (600, H, W)
    teacher_logits_all = seg_logits  # (600, 5, H, W)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(600)
        ep_loss = correct = total = 0

        # Process frames one at a time (SIREN is per-pixel)
        for fi in range(0, 600, 4):
            batch_ids = perm[fi:fi+4]
            batch_loss = 0
            for frame_id in batch_ids:
                fid = frame_id.item()
                target = teacher_am[fid].to(device).flatten()  # (H*W,)
                fids = torch.full((H*W,), fid, dtype=torch.long, device=device)

                pred = model.forward_coords(coords, fids)  # (H*W, 5)
                loss = F.cross_entropy(pred, target)
                batch_loss += loss

                correct += (pred.argmax(1) == target).sum().item()
                total += target.numel()

            opt.zero_grad()
            (batch_loss / len(batch_ids)).backward()
            opt.step()
            ep_loss += batch_loss.item()

        acc = correct / total
        elapsed = time.time() - t0
        print(f"  ep {ep:2d}: loss={ep_loss/600:.3f} acc={acc*100:.2f}% ({elapsed:.0f}s)",
              flush=True)

    # Quick adversarial test
    print("\n  Running adversarial decode test...")
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    seg_d, pose_d, dist = quick_adversarial_test(
        model, t_seg, t_pose, teacher_am.to(device), pose_targets, device)
    print(f"  => seg_dist={seg_d:.6f} pose_mse={pose_d:.6f} distortion={dist:.4f}")
    print(f"  Size: {n_params*2/1024:.0f} KB at FP16")

    del model; torch.cuda.empty_cache()
    return dist


# ═══════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading common data...")
    common = load_common(device)
    print("Ready.\n")

    results = {}

    results['lora'] = test_lora_on_svd(device, common, rank_ratio=0.3, lora_rank=8, epochs=15)
    torch.cuda.empty_cache()

    results['svdquant'] = test_svdquant(device, common)
    torch.cuda.empty_cache()

    results['codebook'] = test_codebook(device, common)
    torch.cuda.empty_cache()

    results['siren'] = test_siren(device, common, epochs=10)
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("SUMMARY — All approaches")
    print("=" * 70)
    print(f"  Non-factored SVD (proven):  1.31 distortion, 5.5 MB")
    print(f"  LoRA-on-SVD:               {results['lora']:.2f} distortion")
    print(f"  SVDQuant:                  {results['svdquant']:.2f} distortion")
    print(f"  Codebook quantization:     {results['codebook']:.2f} distortion")
    print(f"  SIREN:                     {results['siren']:.2f} distortion")
    print(f"\n  Leader: 1.95")


if __name__ == '__main__':
    main()
