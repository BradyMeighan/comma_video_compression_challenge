"""Quick test: FP4-quantize pose_mlp weights in-place, eval, see how badly it hurts."""
import sys, os, copy
from pathlib import Path
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from train import Generator, load_data_full
from prepare import (apply_fp4_to_model, fp4_round_trip, evaluate, gpu_cleanup,
                     estimate_model_bytes)

MODEL_PATH = str(HERE / "colab_run" / "3090_run" / "gen_3090.pt.e80.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[test] device={device}")

print("[test] loading data (full 600)...")
data = load_data_full(device)
print(f"[test] data: train={data['train_rgb'].shape[0]} val={data['val_rgb'].shape[0]}")

def eval_with_pose_quant(quantize_pose: bool, label: str):
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)
    apply_fp4_to_model(gen)  # standard FP4 on Conv2d/Embedding
    if quantize_pose:
        with torch.no_grad():
            for m in gen.pose_mlp:
                if isinstance(m, nn.Linear):
                    m.weight.data = fp4_round_trip(m.weight.data)
                    # leave bias FP16 (matches QConv2d treatment)
    res = evaluate(gen, data, device)
    bytes_est = estimate_model_bytes(gen)
    # Manual: report what bytes WOULD be if pose_mlp counted as FP4 (real Q-Linear conversion)
    pose_w = sum(p.numel() for n, p in gen.named_parameters() if 'pose_mlp' in n and 'weight' in n)
    pose_b = sum(p.numel() for n, p in gen.named_parameters() if 'pose_mlp' in n and 'bias' in n)
    extra_bits_now = pose_w * 16  # current accounting (Linear)
    extra_bits_q   = pose_w * 4 + (pose_w // 32 + 4) * 16  # if treated as QConv2d
    delta_bytes = (extra_bits_now - extra_bits_q) // 8
    delta_brotli = int(delta_bytes * 0.78)
    score_if_arch = res["score"] - 25.0 * delta_brotli / 37545489
    print(f"[{label}] score={res['score']:.4f} seg={res['seg_term']:.4f} "
          f"pose={res['pose_term']:.4f} rate={res['rate_term']:.4f} bytes={bytes_est}")
    print(f"   if pose_mlp were QLinear: model_bytes - {delta_brotli} = {bytes_est - delta_brotli}, "
          f"score -> {score_if_arch:.4f}")
    del gen; gpu_cleanup()
    return res["score"]

print("\n=== A: baseline (pose_mlp FP16) ===")
sA = eval_with_pose_quant(False, "baseline")

print("\n=== B: pose_mlp weights round-tripped through FP4 (no retrain) ===")
sB = eval_with_pose_quant(True, "pose_fp4_naive")

print(f"\nDelta from naive FP4: {sB - sA:+.4f}")
print(f"Hypothetical net score after byte savings: {sB - 25*9610/37545489:.4f}")
