#!/usr/bin/env python
"""
Self-compression: continue training existing model with learnable per-channel bit depth.

CAUTIOUS DESIGN:
- Loads existing trained model (gen_continued.pt)
- Replaces QConv2d/QEmb forward with differentiable quantization (b, e learnable)
- Initializes b=4.0 per channel (matches current FP4) so model starts at SAME SCORE
- Initializes e to match current weight scaling (no quantization shock)
- Adds γ * average_bit_depth to loss
- Trains slowly, evaluates every N seconds
- Checkpoints frequently
- Reports per-channel bit depths so we can see which channels are getting compressed
- Does NOT physically remove channels (just observes b → 0 behavior)

Env vars:
  MODEL_PATH       : input model
  SAVE_MODEL_PATH  : output model
  TRAIN_BUDGET_SEC : how long to train (default 3600 = 1h cautious)
  GAMMA            : compression weight (default 1e-6 — very small to start)
  LR_QUANT         : LR for bit depth params (default 0.05 — much smaller than paper's 0.5)
  LR_WEIGHTS       : LR for weights (default 5e-5 — already converged, low rate)
  EVAL_INTERVAL_SEC: eval every N seconds (default 600 = 10min)
"""
import sys, os, time, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1")
os.environ.setdefault("CONFIG", "B")

from prepare import (load_data, evaluate, gpu_cleanup, MODEL_H, MODEL_W, OUT_H, OUT_W,
                     diff_round, pack_pair_yuv6, get_pose6, kl_on_logits,
                     load_segnet, load_posenet)
from train import Generator, load_data_full, QConv2d, QEmb

MODEL_PATH = os.environ.get("MODEL_PATH", "")
SAVE_MODEL_PATH = os.environ.get("SAVE_MODEL_PATH", MODEL_PATH + ".selfcomp.pt" if MODEL_PATH else "")
BUDGET_SEC = int(os.environ.get("TRAIN_BUDGET_SEC", "3600"))
GAMMA = float(os.environ.get("GAMMA", "1e-6"))
LR_QUANT = float(os.environ.get("LR_QUANT", "0.05"))
LR_WEIGHTS = float(os.environ.get("LR_WEIGHTS", "5e-5"))
EVAL_INTERVAL_SEC = int(os.environ.get("EVAL_INTERVAL_SEC", "600"))

if not MODEL_PATH or not Path(MODEL_PATH).exists():
    print(f"ERROR: MODEL_PATH not found: '{MODEL_PATH}'")
    sys.exit(1)


# ── Differentiable quantization with learnable bit depth ──
def diff_quant(weight, b, e):
    """Differentiable per-channel quantization.
    weight: (out_ch, ...) tensor
    b: (out_ch,) scalar bit depths (>=0)
    e: (out_ch,) exponents
    """
    out_ch = weight.shape[0]
    extra_dims = (1,) * (weight.dim() - 1)
    b_r = b.clamp_min(0).view(out_ch, *extra_dims)
    e_r = e.view(out_ch, *extra_dims)
    scale = 2.0 ** e_r
    max_val = 2.0 ** (b_r - 1) - 1.0
    min_val = -(2.0 ** (b_r - 1))
    # Clamp + round with STE
    normalized = weight / scale
    clipped = torch.clamp(normalized, min_val, max_val)
    rounded = clipped + (clipped.round() - clipped).detach()
    return rounded * scale


def attach_self_compress(module):
    """Attach learnable q_b, q_e to QConv2d / QEmb modules. Patch forward."""
    for m in module.modules():
        if isinstance(m, (QConv2d, QEmb)):
            if not getattr(m, 'quantize_weight', True):
                continue  # skip layers marked as FP16-only
            with torch.no_grad():
                w = m.weight
                if isinstance(m, QConv2d):
                    out_ch = w.shape[0]
                    flat = w.reshape(out_ch, -1)
                else:  # QEmb: weight is (num_emb, dim) — treat each row as a channel
                    out_ch = w.shape[0]
                    flat = w
                # Init e to match current per-channel scale (so initial quantization matches FP4)
                ch_max = flat.abs().max(dim=1).values.clamp_min(1e-8)
                # In FP4, max codebook value is 6, so scale = max/6
                e_init = torch.log2(ch_max / 6.0)
                b_init = torch.full((out_ch,), 4.0, device=w.device)
            m.q_b = nn.Parameter(b_init)
            m.q_e = nn.Parameter(e_init)
            m._original_forward = m.forward

            if isinstance(m, QConv2d):
                def make_forward(layer):
                    def fwd(x):
                        w_q = diff_quant(layer.weight, layer.q_b, layer.q_e)
                        return F.conv2d(x, w_q, layer.bias, layer.stride, layer.padding, layer.dilation, layer.groups)
                    return fwd
            else:  # QEmb
                def make_forward(layer):
                    def fwd(x):
                        w_q = diff_quant(layer.weight, layer.q_b, layer.q_e)
                        return F.embedding(x, w_q, layer.padding_idx)
                    return fwd
            m.forward = make_forward(m)


def compute_compression_loss(module):
    """Q = total bits / total weights. Returns scalar tensor."""
    total_bits = 0.0
    total_weights = 0
    for m in module.modules():
        if hasattr(m, 'q_b'):
            if isinstance(m, QConv2d):
                in_size = m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            else:  # QEmb
                in_size = m.embedding_dim
            b = m.q_b.clamp_min(0)
            total_bits = total_bits + (b * in_size).sum()
            total_weights += b.numel() * in_size
    return total_bits / max(total_weights, 1)


def report_bit_depths(module):
    """Print per-layer bit depth statistics."""
    lines = []
    for name, m in module.named_modules():
        if hasattr(m, 'q_b'):
            b = m.q_b.clamp_min(0).detach()
            n_zero = (b < 0.5).sum().item()
            lines.append(f"  {name}: shape={tuple(b.shape)} mean={b.mean().item():.2f} min={b.min().item():.2f} max={b.max().item():.2f} n_zero={n_zero}")
    return '\n'.join(lines)


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"[selfcomp] config: budget={BUDGET_SEC}s gamma={GAMMA} lr_quant={LR_QUANT} lr_weights={LR_WEIGHTS}", flush=True)

    # Load
    print(f"[selfcomp] Loading model from {MODEL_PATH}", flush=True)
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)
    print(f"[selfcomp] Loaded", flush=True)

    print(f"[selfcomp] Loading data...", flush=True)
    data = load_data_full(device)

    # Baseline eval (before attaching self-compression)
    print(f"[selfcomp] Eval baseline (no self-compression):", flush=True)
    base = evaluate(gen, data, device)
    print(f"  score={base['score']:.4f} seg={base['seg_term']:.4f} pose={base['pose_term']:.4f} rate={base['rate_term']:.4f} model_bytes={base['model_bytes']}", flush=True)

    # Re-load (evaluate calls apply_fp4_to_model which modifies in-place)
    gen.load_state_dict(sd, strict=False)

    # Attach self-compression machinery
    print(f"[selfcomp] Attaching learnable per-channel (b, e) parameters...", flush=True)
    attach_self_compress(gen)
    n_quant_params = sum(p.numel() for n, p in gen.named_parameters() if 'q_b' in n or 'q_e' in n)
    print(f"[selfcomp] Added {n_quant_params} quantization parameters", flush=True)
    print("Initial bit depths:")
    print(report_bit_depths(gen), flush=True)

    # Verify model still gives same score with diff quantization (should match FP4)
    print(f"[selfcomp] Eval with diff_quant (sanity check, should match baseline):", flush=True)
    sanity = evaluate(gen, data, device)
    print(f"  score={sanity['score']:.4f} (was {base['score']:.4f})", flush=True)
    if abs(sanity['score'] - base['score']) > 0.01:
        print(f"  WARNING: sanity score differs by {abs(sanity['score']-base['score']):.4f}. Investigate before training.", flush=True)
    gen.load_state_dict(sd, strict=False)
    attach_self_compress(gen)

    # Optimizer with separate param groups
    quant_params = [p for n, p in gen.named_parameters() if 'q_b' in n or 'q_e' in n]
    weight_params = [p for n, p in gen.named_parameters() if 'q_b' not in n and 'q_e' not in n]
    opt = torch.optim.Adam([
        {'params': weight_params, 'lr': LR_WEIGHTS, 'eps': 1e-5},
        {'params': quant_params,  'lr': LR_QUANT,   'eps': 1e-3},  # high eps for stability
    ])

    segnet = load_segnet(device)
    posenet = load_posenet(device)

    rgb = data["train_rgb"]; masks_data = data["train_masks"]; poses = data["train_poses"]
    BATCH_SIZE = 4

    t_start = time.time()
    last_eval = time.time()
    last_log = time.time()
    epoch = 0
    last_loss_seg = float('nan'); last_loss_pose = float('nan'); last_loss_comp = float('nan')

    while time.time() - t_start < BUDGET_SEC:
        gen.train()
        g = torch.Generator(); g.manual_seed(42 + epoch)
        perm = torch.randperm(rgb.shape[0], generator=g)
        for s in range(0, rgb.shape[0], BATCH_SIZE):
            idx = perm[s:s+BATCH_SIZE]
            b_rgb = rgb.index_select(0, idx).to(device, non_blocking=True)
            b_mask = masks_data.index_select(0, idx).to(device, non_blocking=True)
            b_pose = poses.index_select(0, idx).to(device, non_blocking=True)
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
            seg_loss = 100.0 * 25.0 * (focal_w * ce).mean()
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            pose_loss = 30.0 * F.mse_loss(fp, gt_p)
            comp_loss = GAMMA * compute_compression_loss(gen)
            loss = seg_loss + pose_loss + comp_loss
            last_loss_seg = seg_loss.item(); last_loss_pose = pose_loss.item(); last_loss_comp = comp_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_params, 0.5)
            opt.step()
        epoch += 1
        if time.time() - last_log > 60:
            avg_b = compute_compression_loss(gen).item() / GAMMA if GAMMA > 0 else 0
            print(f"[selfcomp] elapsed={int(time.time()-t_start)}s/{BUDGET_SEC}s epoch={epoch} avg_bits={avg_b:.3f} seg={last_loss_seg:.3f} pose={last_loss_pose:.3f} comp={last_loss_comp:.6f}", flush=True)
            last_log = time.time()

        # Periodic eval + checkpoint
        if time.time() - last_eval > EVAL_INTERVAL_SEC:
            print(f"\n[selfcomp] === Periodic eval at epoch {epoch} ===", flush=True)
            print(report_bit_depths(gen), flush=True)
            cur_sd = {k: v.detach().clone() for k, v in gen.state_dict().items()}
            gen.eval()
            res = evaluate(gen, data, device)
            print(f"  score={res['score']:.4f} seg={res['seg_term']:.4f} pose={res['pose_term']:.4f} rate={res['rate_term']:.4f}", flush=True)
            # Restore (evaluate may have modified weights via apply_fp4_to_model)
            gen.load_state_dict(cur_sd, strict=False)
            torch.save(gen.state_dict(), SAVE_MODEL_PATH + ".ckpt")
            print(f"  saved checkpoint to {SAVE_MODEL_PATH}.ckpt", flush=True)
            last_eval = time.time()

    # Final eval + save
    print(f"\n[selfcomp] === Training done after {epoch} epochs in {time.time()-t_start:.1f}s ===", flush=True)
    print("Final bit depths:")
    print(report_bit_depths(gen), flush=True)
    print(f"\n[selfcomp] Final eval:", flush=True)
    res = evaluate(gen, data, device)
    print(f"  score={res['score']:.4f} seg={res['seg_term']:.4f} pose={res['pose_term']:.4f} rate={res['rate_term']:.4f}", flush=True)
    torch.save(gen.state_dict(), SAVE_MODEL_PATH)
    print(f"saved_model: {SAVE_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
