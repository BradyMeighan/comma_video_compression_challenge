#!/usr/bin/env python
"""
FP4-preserving learnable channel gates for self-compression.

For each QConv2d output channel, add a learnable gate sigmoid(g) ∈ (0,1).
- Init g=10 → sigmoid ≈ 0.99995, model behavior unchanged at start
- Add L1 penalty on sum(sigmoid(g)) → push unused channel gates → 0
- Train jointly with weights (Adam for gates, Lion for weights to match config)
- Channels with sigmoid(g) → 0 can be removed architecturally after training
- FP4 quantization stays UNTOUCHED (gates applied AFTER conv via forward hook)

Env vars:
  MODEL_PATH       : input model
  SAVE_MODEL_PATH  : output (default: MODEL_PATH+.gated.pt)
  TRAIN_BUDGET_SEC : training time (default 1800 = 30 min smoke)
  GATE_LAMBDA      : gate L1 strength (default 1e-4 — very gentle)
  GATE_LR          : LR for gate logits (default 0.01)
  WEIGHT_LR        : LR for weights (default 5e-5 continued from converged)
  EVAL_INTERVAL_SEC: eval every N seconds (default 600)
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
from train import Generator, load_data_full, QConv2d, Lion

MODEL_PATH = os.environ.get("MODEL_PATH", "")
SAVE_MODEL_PATH = os.environ.get("SAVE_MODEL_PATH", MODEL_PATH + ".gated.pt" if MODEL_PATH else "")
BUDGET_SEC = int(os.environ.get("TRAIN_BUDGET_SEC", "1800"))
GATE_LAMBDA = float(os.environ.get("GATE_LAMBDA", "1e-4"))
GATE_LR = float(os.environ.get("GATE_LR", "0.01"))
WEIGHT_LR = float(os.environ.get("WEIGHT_LR", "5e-5"))
EVAL_INTERVAL_SEC = int(os.environ.get("EVAL_INTERVAL_SEC", "600"))

if not MODEL_PATH or not Path(MODEL_PATH).exists():
    print(f"ERROR: MODEL_PATH not found: '{MODEL_PATH}'")
    sys.exit(1)


def attach_gates(model, init_value=1.0):
    """Add learnable per-output-channel LINEAR gates clamped to [0,1].
    Init at 1.0 → no signal loss (no compounding). L1 push from gate_l1 loss.
    Important channels: task loss pulls back toward 1. Unimportant: drift to 0."""
    n_gated = 0
    for name, m in model.named_modules():
        if isinstance(m, QConv2d) and getattr(m, 'quantize_weight', True):
            out_ch = m.out_channels
            gate = nn.Parameter(torch.full((out_ch,), init_value, device=m.weight.device))
            m.register_parameter('_gate_logit', gate)  # name kept for compat

            def make_hook(layer):
                def hook(module, inp, output):
                    g = layer._gate_logit.clamp(0.0, 1.0)
                    return output * g.view(1, -1, 1, 1)
                return hook
            handle = m.register_forward_hook(make_hook(m))
            n_gated += 1
    return n_gated


def gate_l1(model):
    """L1 penalty on linear-clamped gates (encourages → 0)."""
    losses = []
    for m in model.modules():
        if hasattr(m, '_gate_logit'):
            losses.append(m._gate_logit.clamp(0.0, 1.0).sum())
    return sum(losses) if losses else torch.tensor(0.0, device=next(model.parameters()).device)


def gate_stats(model, threshold=0.05):
    """Per-layer gate statistics."""
    lines = []
    n_close_total = 0; n_total = 0
    for name, m in model.named_modules():
        if hasattr(m, '_gate_logit'):
            g = m._gate_logit.clamp(0.0, 1.0).detach()
            n_close = (g < threshold).sum().item()
            n_close_total += n_close
            n_total += g.numel()
            lines.append(f"  {name}: n={g.numel()} mean={g.mean():.3f} min={g.min():.3f} closed(<{threshold})={n_close}")
    lines.append(f"  TOTAL: {n_close_total}/{n_total} channels closed (<{threshold})")
    return '\n'.join(lines)


def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    print(f"[gate] config: budget={BUDGET_SEC}s lambda={GATE_LAMBDA} gate_lr={GATE_LR} weight_lr={WEIGHT_LR}", flush=True)

    print(f"[gate] Loading model {MODEL_PATH}", flush=True)
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)

    print(f"[gate] Loading data...", flush=True)
    data = load_data_full(device)

    # Baseline (no gates)
    print(f"[gate] Eval baseline (no gates):", flush=True)
    base = evaluate(gen, data, device)
    print(f"  score={base['score']:.4f} seg={base['seg_term']:.4f} pose={base['pose_term']:.4f}", flush=True)
    gen.load_state_dict(sd, strict=False)  # restore after FP4 in-place

    # Attach gates
    n_gated = attach_gates(gen, init_value=1.0)
    print(f"[gate] Attached gates to {n_gated} QConv2d layers", flush=True)

    # Sanity: with gate=sigmoid(10)≈0.99995, eval should be very close to baseline
    print(f"[gate] Sanity eval (gates ≈ 1):", flush=True)
    sanity = evaluate(gen, data, device)
    print(f"  score={sanity['score']:.4f} (was {base['score']:.4f}, diff={sanity['score']-base['score']:+.4f})", flush=True)
    gen.load_state_dict(sd, strict=False)
    attach_gates(gen, init_value=1.0)  # re-attach (load_state_dict may have lost gates)

    # Optimizer: Lion for weights (matches CONFIG=B + faster + less memory),
    # AdamW for gate logits (need adaptive LR for tiny scalar params)
    gate_params = [p for n, p in gen.named_parameters() if '_gate_logit' in n]
    weight_params = [p for n, p in gen.named_parameters() if '_gate_logit' not in n]
    weight_opt = Lion(weight_params, lr=WEIGHT_LR / 3.0, betas=(0.9, 0.99))
    gate_opt = torch.optim.AdamW(gate_params, lr=GATE_LR, eps=1e-3)
    class CombinedOpt:
        def zero_grad(self, set_to_none=True):
            weight_opt.zero_grad(set_to_none=set_to_none)
            gate_opt.zero_grad(set_to_none=set_to_none)
        def step(self):
            weight_opt.step()
            gate_opt.step()
    opt = CombinedOpt()

    segnet = load_segnet(device); posenet = load_posenet(device)
    rgb = data["train_rgb"]; masks_data = data["train_masks"]; poses = data["train_poses"]
    BATCH_SIZE = 4

    t_start = time.time()
    last_eval = time.time(); last_log = time.time()
    epoch = 0
    last_seg = float('nan'); last_pose = float('nan'); last_glos = float('nan')

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
            g_loss = GATE_LAMBDA * gate_l1(gen)
            loss = seg_loss + pose_loss + g_loss
            last_seg = seg_loss.item(); last_pose = pose_loss.item(); last_glos = g_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(weight_params, 0.5)
            opt.step()
        epoch += 1
        if time.time() - last_log > 60:
            n_close = sum((torch.sigmoid(p).detach() < 0.05).sum().item() for n, p in gen.named_parameters() if '_gate_logit' in n)
            n_total = sum(p.numel() for n, p in gen.named_parameters() if '_gate_logit' in n)
            print(f"[gate] elapsed={int(time.time()-t_start)}s/{BUDGET_SEC}s epoch={epoch} closed={n_close}/{n_total} seg={last_seg:.3f} pose={last_pose:.3f} g_loss={last_glos:.6f}", flush=True)
            last_log = time.time()

        if time.time() - last_eval > EVAL_INTERVAL_SEC:
            print(f"\n[gate] === Periodic eval at epoch {epoch} ===", flush=True)
            print(gate_stats(gen), flush=True)
            cur_sd = {k: v.detach().clone() for k, v in gen.state_dict().items()}
            res = evaluate(gen, data, device)
            print(f"  score={res['score']:.4f} seg={res['seg_term']:.4f} pose={res['pose_term']:.4f}", flush=True)
            gen.load_state_dict(cur_sd, strict=False)
            torch.save(gen.state_dict(), SAVE_MODEL_PATH + ".ckpt")
            last_eval = time.time()

    # Final
    print(f"\n[gate] === DONE: {epoch} epochs in {time.time()-t_start:.1f}s ===", flush=True)
    print(gate_stats(gen), flush=True)
    cur_sd = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    res = evaluate(gen, data, device)
    print(f"\nFinal eval (gates active): score={res['score']:.4f} seg={res['seg_term']:.4f} pose={res['pose_term']:.4f}", flush=True)
    torch.save(cur_sd, SAVE_MODEL_PATH)
    print(f"saved_model: {SAVE_MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
