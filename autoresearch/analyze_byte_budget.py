"""Break down model_bytes per module — find where 3KB can come from."""
import sys, os
from pathlib import Path
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from train import Generator
from prepare import estimate_model_bytes, apply_fp4_to_model

MODEL_PATH = str(HERE / "colab_run" / "3090_run" / "gen_3090.pt.e80.ckpt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
gen.load_state_dict(sd, strict=False)
apply_fp4_to_model(gen)

ratio = 0.78

def module_bits(m):
    """Bits per the score formula for a single module."""
    if isinstance(m, nn.Conv2d):
        w = m.weight
        if getattr(m, 'quantize_weight', True):
            b = w.numel() * 4 + (w.numel() // 32 + 1) * 16
        else:
            b = w.numel() * 16
        if m.bias is not None:
            b += m.bias.numel() * 16
        return b
    if isinstance(m, nn.Embedding):
        if getattr(m, 'quantize_weight', True):
            return m.weight.numel() * 4 + (m.weight.numel() // 32 + 1) * 16
        return m.weight.numel() * 16
    if isinstance(m, nn.Linear):
        b = m.weight.numel() * 16
        if m.bias is not None: b += m.bias.numel() * 16
        return b
    if isinstance(m, nn.GroupNorm):
        b = 0
        if m.weight is not None: b += m.weight.numel() * 16
        if m.bias is not None:   b += m.bias.numel() * 16
        return b
    return 0

# Group by top-level submodule.
groups = {}
for name, m in gen.named_modules():
    bits = module_bits(m)
    if bits == 0: continue
    top = name.split('.')[0] if '.' in name else name
    groups.setdefault(top, 0)
    groups[top] += bits

print(f"Total estimate: {estimate_model_bytes(gen)} bytes (ratio={ratio})")
print(f"{'group':<14} {'bits':>10} {'raw bytes':>12} {'brotli est':>12} {'%':>6}")
total_brotli = 0
for k, v in sorted(groups.items(), key=lambda x: -x[1]):
    raw = v // 8
    br = int(raw * ratio)
    total_brotli += br
    print(f"{k:<14} {v:>10} {raw:>12} {br:>12} {br*100/(estimate_model_bytes(gen)):>5.1f}")

print()
print("Per-named-module breakdown (top 15):")
mod_bits = []
for name, m in gen.named_modules():
    bits = module_bits(m)
    if bits == 0: continue
    mod_bits.append((name, type(m).__name__, bits))
for name, t, bits in sorted(mod_bits, key=lambda x: -x[2])[:15]:
    raw = bits // 8
    br = int(raw * ratio)
    print(f"  {name:<40} {t:<14} {bits:>8} bits = {br:>5} brotli")

# What if we converted pose_mlp Linear -> QLinear (FP4 conv)?
pose_bits_now = sum(module_bits(m) for n, m in gen.named_modules() if 'pose_mlp' in n)
pose_total_w = 6*64 + 64*64 + 64*64
pose_total_b = 64 + 64 + 64
pose_bits_qlinear = pose_total_w * 4 + (pose_total_w // 32 + 4) * 16 + pose_total_b * 16
print()
print(f"pose_mlp now (FP16):   {pose_bits_now} bits = {int(pose_bits_now / 8 * ratio)} brotli")
print(f"pose_mlp as QLinear:   {pose_bits_qlinear} bits = {int(pose_bits_qlinear / 8 * ratio)} brotli")
print(f"  → SAVINGS: {(pose_bits_now - pose_bits_qlinear) // 8} raw, "
      f"{int((pose_bits_now - pose_bits_qlinear) / 8 * ratio)} brotli bytes")
