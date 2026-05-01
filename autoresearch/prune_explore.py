#!/usr/bin/env python
"""
Pruning exploration: take a trained model, test multiple pruning methods.

Outputs a CSV mapping (method, sparsity) → (score, real brotli size).
Use this to find the best pruning approach to apply to the final Colab model.

Env vars:
  MODEL_PATH   : path to gen.pt to prune (REQUIRED)
  OUTPUT_DIR   : where to save results (default: autoresearch/prune_results)
  FULL_DATA    : 1 = use all 600 pairs (default), 0 = proxy 80/20
  ITER_RETRAIN_SEC : seconds per retrain cycle in iterative method (default: 1200 = 20min)
"""
import sys, os, time, csv, io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from prepare import (load_data, evaluate, gpu_cleanup, MODEL_H, MODEL_W, OUT_H, OUT_W,
                     diff_round, pack_pair_yuv6, get_pose6, kl_on_logits, fake_quant_fp4_ste,
                     load_segnet, load_posenet)
import einops

# ── Config ──
MODEL_PATH = os.environ.get("MODEL_PATH", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "autoresearch/prune_results")
USE_FULL_DATA = bool(int(os.environ.get("FULL_DATA", "1")))
ITER_RETRAIN_SEC = int(os.environ.get("ITER_RETRAIN_SEC", "1200"))

if not MODEL_PATH or not Path(MODEL_PATH).exists():
    print(f"ERROR: MODEL_PATH not set or not found: '{MODEL_PATH}'")
    print("Set: MODEL_PATH=path/to/gen.pt python autoresearch/prune_explore.py")
    sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = Path(OUTPUT_DIR) / "prune_results.csv"

# ── Brotli for actual file size measurement ──
try:
    import brotli
except ImportError:
    print("ERROR: install brotli (pip install brotli)")
    sys.exit(1)


def actual_brotli_size(state_dict):
    """Real brotli size of a state_dict (vs our score's fixed 0.78 ratio assumption)."""
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return len(brotli.compress(buf.getvalue(), quality=11))


def is_prunable_weight(name, tensor):
    """Skip biases, GroupNorm affine, embeddings, output convs (already FP16)."""
    if not tensor.dtype.is_floating_point or tensor.dim() < 2:
        return False
    # Skip patterns:
    if 'bias' in name: return False
    if '.norm.' in name: return False
    if name.endswith('.bias'): return False
    return True


def static_magnitude_prune(state_dict, sparsity):
    """Zero out the smallest |w| weights globally up to sparsity fraction."""
    pruned = {k: v.clone() for k, v in state_dict.items()}
    # Pool all prunable weights, find global threshold
    all_abs = []
    for k, v in pruned.items():
        if is_prunable_weight(k, v):
            all_abs.append(v.abs().flatten())
    pool = torch.cat(all_abs)
    threshold = torch.quantile(pool.float(), sparsity).item()
    for k, v in pruned.items():
        if is_prunable_weight(k, v):
            mask = v.abs() >= threshold
            pruned[k] = v * mask.to(v.dtype)
    return pruned


def per_layer_magnitude_prune(state_dict, sparsity):
    """Zero out the smallest |w| per-layer (each layer gets same sparsity %)."""
    pruned = {k: v.clone() for k, v in state_dict.items()}
    for k, v in pruned.items():
        if is_prunable_weight(k, v):
            n = v.numel()
            k_keep = max(1, int(n * (1 - sparsity)))
            threshold = v.abs().flatten().kthvalue(n - k_keep + 1).values.item() if k_keep < n else 0.0
            mask = v.abs() >= threshold
            pruned[k] = v * mask.to(v.dtype)
    return pruned


def channel_prune(state_dict, sparsity):
    """Zero entire output channels with smallest L1 norm (per conv layer)."""
    pruned = {k: v.clone() for k, v in state_dict.items()}
    for k, v in pruned.items():
        if is_prunable_weight(k, v) and v.dim() == 4:  # conv weights
            # v shape: (out_ch, in_ch, kH, kW)
            ch_l1 = v.abs().flatten(1).sum(dim=1)  # (out_ch,)
            n_ch = ch_l1.shape[0]
            k_keep = max(1, int(n_ch * (1 - sparsity)))
            if k_keep < n_ch:
                _, top_idx = torch.topk(ch_l1, k_keep)
                mask = torch.zeros(n_ch, dtype=torch.bool, device=v.device)
                mask[top_idx] = True
                pruned[k] = v * mask.view(-1, 1, 1, 1).to(v.dtype)
    return pruned


def nm_sparsity_prune(state_dict, n=2, m=4):
    """N:M structured sparsity: keep N largest in every group of M."""
    pruned = {k: v.clone() for k, v in state_dict.items()}
    for k, v in pruned.items():
        if is_prunable_weight(k, v):
            flat = v.flatten()
            pad = (m - flat.numel() % m) % m
            if pad: flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])
            grouped = flat.view(-1, m)
            _, top_idx = grouped.abs().topk(n, dim=1)
            mask = torch.zeros_like(grouped)
            mask.scatter_(1, top_idx, 1.0)
            new_flat = (grouped * mask).flatten()[:v.numel()]
            pruned[k] = new_flat.view_as(v)
    return pruned


def random_prune(state_dict, sparsity, seed=0):
    """Random pruning baseline (control)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    pruned = {k: v.clone() for k, v in state_dict.items()}
    for k, v in pruned.items():
        if is_prunable_weight(k, v):
            mask = (torch.rand(v.shape, generator=g) >= sparsity).to(v.dtype).to(v.device)
            pruned[k] = v * mask
    return pruned


def main():
    from train import Generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[prune] Device: {device}")
    print(f"[prune] Loading state_dict from {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    print(f"[prune] Loaded {len(state_dict)} tensors")

    print(f"[prune] Loading data (full_data={USE_FULL_DATA})...")
    if USE_FULL_DATA:
        from train import load_data_full
        data = load_data_full(device)
    else:
        data = load_data(device)
    print(f"[prune] Data ready: train={data['train_rgb'].shape[0]} val={data['val_rgb'].shape[0]} pairs")

    # Eval helper
    def eval_state(sd, label=""):
        gen = Generator().to(device)
        try:
            gen.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"  load_state_dict ERROR: {e}")
            return None
        try:
            result = evaluate(gen, data, device)
        except Exception as e:
            print(f"  evaluate ERROR: {e}")
            result = None
        del gen
        gpu_cleanup()
        return result

    # CSV header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "sparsity_param", "score", "seg_term", "pose_term", "rate_term",
                         "model_bytes_estimate", "actual_brotli_bytes", "elapsed_sec"])

    def write_row(method, sparsity, result, actual_sz, elapsed):
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if result is None:
                writer.writerow([method, sparsity, "FAIL", "", "", "", "", actual_sz, elapsed])
            else:
                writer.writerow([method, sparsity, result["score"], result["seg_term"], result["pose_term"],
                                 result["rate_term"], result["model_bytes"], actual_sz, elapsed])

    # ════════════════ BASELINE ════════════════
    print("\n[baseline] No pruning")
    t0 = time.time()
    baseline = eval_state(state_dict, "baseline")
    baseline_sz = actual_brotli_size(state_dict)
    elapsed = time.time() - t0
    if baseline:
        print(f"  score={baseline['score']:.4f} seg={baseline['seg_term']:.4f} pose={baseline['pose_term']:.4f} "
              f"rate={baseline['rate_term']:.4f} brotli={baseline_sz/1024:.1f}KB ({elapsed:.1f}s)")
    write_row("baseline", 0.0, baseline, baseline_sz, elapsed)

    # ════════════════ METHOD 1: Static magnitude pruning ════════════════
    print("\n[method 1] Static (global) magnitude pruning")
    for sp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        t0 = time.time()
        sd = static_magnitude_prune(state_dict, sp)
        result = eval_state(sd)
        sz = actual_brotli_size(sd)
        elapsed = time.time() - t0
        score = result['score'] if result else float('nan')
        print(f"  sparsity={sp:.1f}: score={score:.4f} brotli={sz/1024:.1f}KB ({elapsed:.1f}s)")
        write_row("static_magnitude", sp, result, sz, elapsed)

    # ════════════════ METHOD 2: Per-layer magnitude pruning ════════════════
    print("\n[method 2] Per-layer magnitude pruning")
    for sp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        t0 = time.time()
        sd = per_layer_magnitude_prune(state_dict, sp)
        result = eval_state(sd)
        sz = actual_brotli_size(sd)
        elapsed = time.time() - t0
        score = result['score'] if result else float('nan')
        print(f"  sparsity={sp:.1f}: score={score:.4f} brotli={sz/1024:.1f}KB ({elapsed:.1f}s)")
        write_row("per_layer_magnitude", sp, result, sz, elapsed)

    # ════════════════ METHOD 3: Channel pruning ════════════════
    print("\n[method 3] Channel pruning (zero entire output channels)")
    for sp in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
        t0 = time.time()
        sd = channel_prune(state_dict, sp)
        result = eval_state(sd)
        sz = actual_brotli_size(sd)
        elapsed = time.time() - t0
        score = result['score'] if result else float('nan')
        print(f"  sparsity={sp:.2f}: score={score:.4f} brotli={sz/1024:.1f}KB ({elapsed:.1f}s)")
        write_row("channel", sp, result, sz, elapsed)

    # ════════════════ METHOD 4: N:M sparsity ════════════════
    print("\n[method 4] N:M structured sparsity")
    for n, m in [(1, 4), (2, 4), (3, 4), (1, 2)]:
        t0 = time.time()
        sd = nm_sparsity_prune(state_dict, n, m)
        result = eval_state(sd)
        sz = actual_brotli_size(sd)
        elapsed = time.time() - t0
        score = result['score'] if result else float('nan')
        sp = 1 - n/m
        print(f"  {n}:{m} (sparsity={sp:.2f}): score={score:.4f} brotli={sz/1024:.1f}KB ({elapsed:.1f}s)")
        write_row(f"nm_{n}of{m}", sp, result, sz, elapsed)

    # ════════════════ METHOD 5: Random pruning (control) ════════════════
    print("\n[method 5] Random pruning (control baseline)")
    for sp in [0.2, 0.4, 0.6, 0.8]:
        t0 = time.time()
        sd = random_prune(state_dict, sp)
        result = eval_state(sd)
        sz = actual_brotli_size(sd)
        elapsed = time.time() - t0
        score = result['score'] if result else float('nan')
        print(f"  sparsity={sp:.1f}: score={score:.4f} brotli={sz/1024:.1f}KB ({elapsed:.1f}s)")
        write_row("random", sp, result, sz, elapsed)

    # ════════════════ METHOD 6: Iterative magnitude pruning + retrain ════════════════
    # Slow but most powerful — incrementally increases sparsity, retrains in between
    print(f"\n[method 6] Iterative magnitude prune + retrain ({ITER_RETRAIN_SEC}s per cycle)")
    cur_sd = {k: v.clone() for k, v in state_dict.items()}
    for cycle, target_sp in enumerate([0.2, 0.35, 0.5, 0.6, 0.7, 0.8]):
        t0 = time.time()
        # Apply pruning to current weights
        cur_sd = per_layer_magnitude_prune(cur_sd, target_sp)
        # Retrain briefly to recover accuracy (with QAT enabled, focal loss)
        cur_sd = retrain_with_mask(cur_sd, data, device, ITER_RETRAIN_SEC, target_sp)
        # Evaluate
        result = eval_state(cur_sd)
        sz = actual_brotli_size(cur_sd)
        elapsed = time.time() - t0
        score = result['score'] if result else float('nan')
        print(f"  cycle {cycle+1} sparsity={target_sp:.2f}: score={score:.4f} brotli={sz/1024:.1f}KB ({elapsed:.1f}s)")
        write_row("iterative_magnitude_retrain", target_sp, result, sz, elapsed)

    print(f"\n[done] Results saved to {csv_path}")
    print(f"[done] Baseline brotli: {baseline_sz/1024:.1f}KB; estimate_model_bytes: {baseline['model_bytes']/1024:.1f}KB")


def retrain_with_mask(state_dict, data, device, budget_sec, target_sparsity):
    """Brief retraining with sparse weights kept zero. Returns updated state_dict."""
    from train import Generator, GRAD_CLIP, BATCH_SIZE, make_batches
    gen = Generator().to(device)
    gen.load_state_dict(state_dict, strict=False)
    gen.train()

    # Compute mask: where state_dict has zeros, gradient should be zero
    masks = {}
    for k, v in state_dict.items():
        if is_prunable_weight(k, v):
            masks[k] = (v != 0).to(v.dtype).to(device)

    # Get refs to actual model parameters by name
    name_to_param = dict(gen.named_parameters())

    # Apply mask to corresponding parameters at start
    with torch.no_grad():
        for k, m in masks.items():
            if k in name_to_param:
                name_to_param[k].data.mul_(m)

    segnet = load_segnet(device); posenet = load_posenet(device)
    opt = torch.optim.AdamW(gen.parameters(), lr=1e-4, betas=(0.9, 0.99))
    gen.set_qat(True)  # QAT on so retrain stays FP4-compatible

    rgb = data["train_rgb"]
    masks_data = data["train_masks"]
    poses = data["train_poses"]

    t0 = time.time()
    epoch = 0
    last_log = time.time()
    while time.time() - t0 < budget_sec:
        for b_rgb, b_mask, b_pose in make_batches(rgb, masks_data, poses, epoch, device):
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
            f1d = F.interpolate(diff_round(f1u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            pred_logits = segnet(f2d).float()
            ce = F.cross_entropy(pred_logits, gt_cls, reduction='none')
            with torch.no_grad():
                p_t = torch.exp(-ce.detach()).clamp_max(0.999)
                focal_w = (1.0 - p_t).pow(2.0)
            seg_loss = 100.0 * 25.0 * (focal_w * ce).mean()
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            pose_loss = 30.0 * F.mse_loss(fp, gt_p)
            loss = seg_loss + pose_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt.step()
            # Re-apply mask after step (keeps pruned weights at zero)
            with torch.no_grad():
                for k, m in masks.items():
                    if k in name_to_param:
                        name_to_param[k].data.mul_(m)
            if time.time() - t0 > budget_sec: break
        epoch += 1
        if time.time() - last_log > 60:
            print(f"    [retrain sp={target_sparsity:.2f}] epoch={epoch} elapsed={int(time.time()-t0)}s/{budget_sec}s loss={loss.item():.4f}", flush=True)
            last_log = time.time()

    out_sd = {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()}
    del gen, segnet, posenet, opt
    gpu_cleanup()
    return out_sd


if __name__ == "__main__":
    main()
