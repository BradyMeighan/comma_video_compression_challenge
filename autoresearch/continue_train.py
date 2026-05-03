#!/usr/bin/env python
"""
Continue training a saved gen.pt from where it left off.
Skips anchor + finetune, only runs joint stage with combined seg+pose loss.

Useful when you have a model that converged in seg/finetune but pose was still
dropping in joint when the budget ran out (which is what happened to our Colab run).

Env vars:
  MODEL_PATH       : path to gen.pt to load (REQUIRED)
  SAVE_MODEL_PATH  : path to save updated model (default: {MODEL_PATH}.continued)
  TRAIN_BUDGET_SEC_OVERRIDE : seconds to train (default: 14400 = 4h)
  POSE_WEIGHT      : pose loss multiplier in combined loss (default: 60, original was 30)
  JT_LR_OVERRIDE   : starting LR for joint stage (default: 5e-5, same as original JT_LR)
  EMA_DECAY        : EMA decay rate (default: 0.999)
  COSINE_LR        : 1 = cosine decay LR over budget (default: 1)
  GRAD_CLIP_OVERRIDE : (default: 0.5)
  CHECKPOINT_INTERVAL_SEC : periodic time-based checkpoint frequency (default: 600 = 10min)
                            saves to {SAVE_MODEL_PATH}.ckpt (overwrites)
  CHECKPOINT_EPOCH_INTERVAL : periodic epoch-based checkpoint frequency (default: 0 = disabled)
                              saves to {SAVE_MODEL_PATH}.e{epoch}.ckpt (numbered, keeps history)
  FULL_DATA        : 1 = use 600 pairs (default: 1)
  CONFIG           : architecture config (default: B for boundary+Lion)
  BATCH_SIZE       : batch size (default 4, 3090-safe; bump to 8 for A100, 8-16 for H100)
"""
import sys, os, time, math, gc, einops
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

# Set defaults BEFORE importing train.py (which reads env vars at import time)
os.environ.setdefault("CONFIG", "B")
os.environ.setdefault("FULL_DATA", "1")
os.environ.setdefault("EMA_DECAY", "0.999")
os.environ.setdefault("COSINE_LR", "1")
os.environ.setdefault("GRAD_CLIP_OVERRIDE", "0.5")
os.environ.setdefault("CHECKPOINT_INTERVAL_SEC", "600")

from prepare import (load_data, evaluate, gpu_cleanup, MODEL_H, MODEL_W, OUT_H, OUT_W,
                     diff_round, pack_pair_yuv6, get_pose6, kl_on_logits, fake_quant_fp4_ste,
                     load_segnet, load_posenet)
from train import Generator, load_data_full, make_batches, Lion

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "")
SAVE_MODEL_PATH = os.environ.get("SAVE_MODEL_PATH", MODEL_PATH + ".continued.pt" if MODEL_PATH else "")
BUDGET_SEC = int(os.environ.get("TRAIN_BUDGET_SEC_OVERRIDE", "14400"))
POSE_WEIGHT = float(os.environ.get("POSE_WEIGHT", "60.0"))
JT_LR = float(os.environ.get("JT_LR_OVERRIDE", "5e-5"))
USE_LION = os.environ.get("CONFIG", "B").upper() == "B"
EMA_DECAY = float(os.environ["EMA_DECAY"])
COSINE_LR = bool(int(os.environ["COSINE_LR"]))
GRAD_CLIP = float(os.environ["GRAD_CLIP_OVERRIDE"])
CHECKPOINT_INTERVAL_SEC = int(os.environ["CHECKPOINT_INTERVAL_SEC"])
CHECKPOINT_EPOCH_INTERVAL = int(os.environ.get("CHECKPOINT_EPOCH_INTERVAL", "0"))
USE_BOUNDARY = os.environ.get("CONFIG", "B").upper() in ("B", "C")

if not MODEL_PATH or not Path(MODEL_PATH).exists():
    print(f"ERROR: MODEL_PATH not set or not found: '{MODEL_PATH}'")
    sys.exit(1)

print(f"[continue] Loading {MODEL_PATH}")
print(f"[continue] Will save to {SAVE_MODEL_PATH}")
print(f"[continue] Budget: {BUDGET_SEC}s ({BUDGET_SEC/3600:.1f}h) | pose_weight={POSE_WEIGHT} | jt_lr={JT_LR} | lion={USE_LION}")


def boundary_mask(gt_cls):
    gt_pad = F.pad(gt_cls.unsqueeze(1).float(), (1, 1, 1, 1), mode='replicate')
    c = gt_pad[:, :, 1:-1, 1:-1]
    return ((c != gt_pad[:, :, :-2, 1:-1]) | (c != gt_pad[:, :, 2:, 1:-1]) |
            (c != gt_pad[:, :, 1:-1, :-2]) | (c != gt_pad[:, :, 1:-1, 2:])).squeeze(1).float()


def cosine_lr_factor(progress):
    p = max(0.0, min(1.0, progress))
    return 0.1 + 0.9 * (0.5 * (1.0 + math.cos(math.pi * p)))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Load model + data
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)
    print(f"[continue] Loaded gen.state_dict ({sum(p.numel() for p in gen.parameters())} params)")

    data = load_data_full(device)
    rgb = data["train_rgb"]; masks = data["train_masks"]; poses = data["train_poses"]
    print(f"[continue] Data ready: {rgb.shape[0]} train pairs")

    # Eval baseline (current model)
    print("[continue] Eval baseline:")
    result = evaluate(gen, data, device)
    print(f"  baseline score: {result['score']:.6f}  seg: {result['seg_term']:.4f}  pose: {result['pose_term']:.4f}  rate: {result['rate_term']:.4f}")

    # Re-load gen weights from sd (evaluate calls apply_fp4_to_model which modifies in-place)
    gen.load_state_dict(sd, strict=False)

    segnet = load_segnet(device); posenet = load_posenet(device)

    # Optimizer
    for p in gen.parameters(): p.requires_grad = True
    opt = Lion(gen.parameters(), lr=JT_LR / 3.0, betas=(0.9, 0.99)) if USE_LION else \
          torch.optim.AdamW(gen.parameters(), lr=JT_LR, betas=(0.9, 0.99))
    init_lr = opt.param_groups[0]['lr']
    gen.set_qat(True)

    # EMA init from current weights
    ema_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}

    t_start = time.time()
    epoch = 0
    last_log_time = time.time()
    last_ckpt_time = time.time()
    last_seg = float('nan'); last_pose = float('nan')

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))  # 3090=4, A100=8, H100=8-16
    print(f"[continue] BATCH_SIZE={BATCH_SIZE}")

    print(f"[continue] Starting joint stage with warm restart (LR={init_lr:.2e})")
    while time.time() - t_start < BUDGET_SEC:
        progress = (time.time() - t_start) / BUDGET_SEC
        if COSINE_LR:
            f = cosine_lr_factor(progress)
            for g in opt.param_groups:
                g['lr'] = init_lr * f
        gen.train()
        if time.time() - last_log_time > 60:
            print(f"[joint+] elapsed={int(time.time()-t_start)}s/{BUDGET_SEC}s ({100*progress:.0f}%) epoch={epoch} lr={opt.param_groups[0]['lr']:.2e} last_seg={last_seg:.4f} last_pose={last_pose:.4f}", flush=True)
            last_log_time = time.time()

        for s in range(0, rgb.shape[0], BATCH_SIZE):
            g_perm = torch.Generator().manual_seed(42 + epoch)
            perm = torch.randperm(rgb.shape[0], generator=g_perm)
            idx = perm[s:s+BATCH_SIZE]
            b_rgb = rgb.index_select(0, idx).to(device, non_blocking=True)
            b_mask = masks.index_select(0, idx).to(device, non_blocking=True)
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
            f1d = F.interpolate(diff_round(f1u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            f2d = F.interpolate(diff_round(f2u.clamp(0, 255)), (MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            pred_logits = segnet(f2d).float()
            ce = F.cross_entropy(pred_logits, gt_cls, reduction='none')
            with torch.no_grad():
                p_t = torch.exp(-ce.detach()).clamp_max(0.999)
                focal_w = (1.0 - p_t).pow(2.0)
                weight = focal_w * (1.0 + 4.0 * boundary_mask(gt_cls)) if USE_BOUNDARY else focal_w
            seg_loss = 100.0 * 25.0 * (weight * ce).mean()
            fp = get_pose6(posenet, pack_pair_yuv6(f1d, f2d).float()).float()
            pose_loss = POSE_WEIGHT * F.mse_loss(fp, gt_p)
            loss = seg_loss + pose_loss
            last_seg = seg_loss.item(); last_pose = pose_loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gen.parameters(), GRAD_CLIP)
            opt.step()
            with torch.no_grad():
                for k, v in gen.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema_state[k].mul_(EMA_DECAY).add_(v.detach(), alpha=1 - EMA_DECAY)
                    else:
                        ema_state[k].copy_(v)
        epoch += 1
        # Periodic time-based checkpoint (overwrites)
        if CHECKPOINT_INTERVAL_SEC > 0 and (time.time() - last_ckpt_time) > CHECKPOINT_INTERVAL_SEC:
            torch.save(ema_state, SAVE_MODEL_PATH + ".ckpt")
            last_ckpt_time = time.time()
            print(f"[ckpt] saved EMA to {SAVE_MODEL_PATH}.ckpt at epoch {epoch}", flush=True)
        # Periodic epoch-based checkpoint (numbered, keeps history)
        if CHECKPOINT_EPOCH_INTERVAL > 0 and epoch % CHECKPOINT_EPOCH_INTERVAL == 0:
            ep_ckpt = f"{SAVE_MODEL_PATH}.e{epoch}.ckpt"
            torch.save(ema_state, ep_ckpt)
            print(f"[ckpt-ep] saved EMA to {ep_ckpt}", flush=True)
        # Inline eval — blocks training so no GPU contention
        EVAL_EPOCH_INTERVAL = int(os.environ.get("EVAL_EPOCH_INTERVAL", "0"))
        if EVAL_EPOCH_INTERVAL > 0 and epoch % EVAL_EPOCH_INTERVAL == 0:
            # Snapshot current weights, swap in EMA, eval, restore
            cur_state = {k: v.detach().clone() for k, v in gen.state_dict().items()}
            gen.load_state_dict(ema_state)
            t_eval = time.time()
            eval_result = evaluate(gen, data, device)
            eval_time = time.time() - t_eval
            print(f"[eval] epoch={epoch} score={eval_result['score']:.6f} "
                  f"seg={eval_result['seg_term']:.4f} pose={eval_result['pose_term']:.4f} "
                  f"rate={eval_result['rate_term']:.4f} ({eval_time:.0f}s)", flush=True)
            # write to a CSV next to ckpts
            eval_csv = f"{SAVE_MODEL_PATH}.eval_log.csv"
            new_file = not Path(eval_csv).exists()
            with open(eval_csv, 'a') as f:
                if new_file:
                    f.write("epoch,score,seg_term,pose_term,rate_term,eval_time\n")
                f.write(f"{epoch},{eval_result['score']},{eval_result['seg_term']},"
                        f"{eval_result['pose_term']},{eval_result['rate_term']},{eval_time}\n")
            # restore live weights for continued training
            gen.load_state_dict(cur_state)
            del cur_state, eval_result
            gpu_cleanup()

    train_time = time.time() - t_start
    print(f"[continue] Done. {epoch} epochs in {train_time:.1f}s")

    # Eval with EMA weights
    del segnet, posenet, opt
    gpu_cleanup()
    gen.load_state_dict(ema_state)
    print("[continue] Final eval:")
    result = evaluate(gen, data, device)
    print("---")
    for k in ["score", "seg_term", "pose_term", "rate_term", "model_bytes", "total_bytes", "n_params"]:
        v = result[k]
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Save final
    if SAVE_MODEL_PATH:
        torch.save(gen.state_dict(), SAVE_MODEL_PATH)
        print(f"saved_model: {SAVE_MODEL_PATH}")


if __name__ == "__main__":
    main()
