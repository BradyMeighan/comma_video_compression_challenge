#!/usr/bin/env python
"""
Train a tiny SegNet-focused REN on ODD frames only, then evaluate impact.

This is a controlled experiment for the evenframe_meta_v1 submission:
- input: current inflated output (with even-frame sidecars already applied)
- target: cached GT SegNet class maps (odd frames)
- objective: reduce SegNet disagreement while keeping edits small
- output:
    * quantized model sidecar in archive/: frame1_ren_seg.int8.bz2
    * corrected raw: inflated/0_ren.raw
    * fast_eval score with model size penalty included
"""
import argparse
import bz2
import io
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
SUB = ROOT / "submissions" / "evenframe_meta_v1"
RAW_IN = SUB / "inflated" / "0.raw"
RAW_OUT = SUB / "inflated" / "0_ren.raw"
GT_CACHE = SUB / "_cache" / "gt.pt"
ARCHIVE_ZIP = SUB / "archive.zip"
MODEL_OUT = SUB / "archive" / "frame1_ren_seg.int8.bz2"
UNCOMPRESSED_BYTES = 37_545_489

W_CAM, H_CAM = 1164, 874
MODEL_W, MODEL_H = 512, 384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OddSegREN(nn.Module):
    """
    Tiny residual model operating at model-space resolution (512x384).
    PixelUnshuffle/Shuffle keeps compute manageable.
    """

    def __init__(self, features: int = 24, max_delta: float = 10.0):
        super().__init__()
        self.max_delta = float(max_delta)
        self.down = nn.PixelUnshuffle(2)  # 3 -> 12 channels at H/2, W/2
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        # x in [0,255], shape (B,3,H,W)
        x_norm = x / 255.0
        res_norm = self.up(self.body(self.down(x_norm)))
        # Bounded residual so training doesn't explode pose.
        res = torch.tanh(res_norm) * self.max_delta
        y = (x + res).clamp(0.0, 255.0)
        return y, res


def save_int8_bz2(model: nn.Module, path: Path) -> int:
    """
    Serialize model state_dict as int8 + per-tensor scale and bz2-compress.
    Compatible with a lightweight custom loader at inflate time.
    """
    sd = model.state_dict()
    buf = io.BytesIO()
    buf.write(struct.pack("<I", len(sd)))
    for name, tensor in sd.items():
        name_b = name.encode("utf-8")
        buf.write(struct.pack("<I", len(name_b)))
        buf.write(name_b)
        buf.write(struct.pack("<I", tensor.ndim))
        for s in tensor.shape:
            buf.write(struct.pack("<I", int(s)))

        t = tensor.detach().float().flatten()
        max_abs = float(t.abs().max().item())
        scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
        q = (t / scale).round().clamp(-127, 127).to(torch.int8).cpu().numpy().tobytes()
        buf.write(struct.pack("<f", float(scale)))
        buf.write(struct.pack("<I", len(q)))
        buf.write(q)

    packed = bz2.compress(buf.getvalue(), compresslevel=9)
    path.write_bytes(packed)
    return len(packed)


def compute_boundary_weights(seg: torch.Tensor, boundary_boost: float) -> torch.Tensor:
    """
    seg: (N,H,W), class labels.
    Returns per-pixel weights emphasizing class boundaries.
    """
    center = seg
    up = torch.roll(seg, shifts=1, dims=1)
    down = torch.roll(seg, shifts=-1, dims=1)
    left = torch.roll(seg, shifts=1, dims=2)
    right = torch.roll(seg, shifts=-1, dims=2)
    edge = ((center != up) | (center != down) | (center != left) | (center != right)).float()
    return 1.0 + boundary_boost * edge


def load_training_tensors():
    if not RAW_IN.exists():
        raise FileNotFoundError(f"Missing raw input: {RAW_IN}")
    if not GT_CACHE.exists():
        raise FileNotFoundError(f"Missing GT cache: {GT_CACHE}")

    gt = torch.load(GT_CACHE, weights_only=True)
    seg = gt["seg"].long()  # (600, 384, 512), odd-frame target by evaluator design
    n_pairs = int(seg.shape[0])

    raw = np.fromfile(RAW_IN, dtype=np.uint8).reshape(n_pairs * 2, H_CAM, W_CAM, 3)
    odd = raw[1::2]  # (600, H, W, 3)

    odd_model = np.zeros((n_pairs, MODEL_H, MODEL_W, 3), dtype=np.uint8)
    for i in range(n_pairs):
        odd_model[i] = cv2.resize(odd[i], (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)

    x = torch.from_numpy(odd_model).permute(0, 3, 1, 2).float()  # (N,3,H,W)
    y = seg
    return x, y, raw


def train_model(args, x_train: torch.Tensor, y_train: torch.Tensor):
    sys.path.insert(0, str(ROOT))
    from modules import SegNet, segnet_sd_path

    segnet = SegNet().eval().to(DEVICE)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(DEVICE)))
    for p in segnet.parameters():
        p.requires_grad_(False)

    # Optional hard-example mining: train on worst-K odd frames by current seg mismatch.
    if args.train_k > 0 and args.train_k < len(x_train):
        print(f"Selecting worst-{args.train_k} odd frames by baseline seg disagreement...", flush=True)
        with torch.inference_mode():
            errs = []
            bs_sel = 32
            for i in range(0, len(x_train), bs_sel):
                xb = x_train[i : i + bs_sel].to(DEVICE)
                yb = y_train[i : i + bs_sel].to(DEVICE)
                pred = segnet(xb).argmax(1)
                e = (pred != yb).float().mean(dim=(1, 2)).cpu()
                errs.append(e)
            errs = torch.cat(errs, dim=0)
            top_idx = torch.topk(errs, k=args.train_k, largest=True).indices
        x_work = x_train[top_idx]
        y_work = y_train[top_idx]
        print(
            f"Selected subset mean seg disagreement: {errs[top_idx].mean().item():.6f} "
            f"(full-set mean={errs.mean().item():.6f})",
            flush=True,
        )
    else:
        x_work = x_train
        y_work = y_train

    model = OddSegREN(features=args.features, max_delta=args.max_delta).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    weights = compute_boundary_weights(y_work, args.boundary_boost)
    ds = torch.utils.data.TensorDataset(x_work, y_work, weights)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0
    )

    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    use_amp = DEVICE.type == "cuda"

    print(f"Training REN on {len(ds)} odd frames | device={DEVICE} | bs={args.batch_size} | epochs={args.epochs}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        sum_loss = 0.0
        sum_seg = 0.0
        sum_l1 = 0.0
        n_batches = 0

        for batch_idx, (xb, yb, wb) in enumerate(dl, start=1):
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            wb = wb.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out, res = model(xb)
                    logits = segnet(out)
                    ce = F.cross_entropy(logits, yb, reduction="none")
                    seg_loss = (ce * wb).mean()
                    l1_loss = res.abs().mean()
                    loss = seg_loss + args.l1_lambda * l1_loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out, res = model(xb)
                logits = segnet(out)
                ce = F.cross_entropy(logits, yb, reduction="none")
                seg_loss = (ce * wb).mean()
                l1_loss = res.abs().mean()
                loss = seg_loss + args.l1_lambda * l1_loss
                loss.backward()
                optimizer.step()

            sum_loss += float(loss.detach().item())
            sum_seg += float(seg_loss.detach().item())
            sum_l1 += float(l1_loss.detach().item())
            n_batches += 1
            if batch_idx % max(1, args.log_every) == 0:
                print(
                    f"  epoch {epoch:02d} batch {batch_idx:03d}/{len(dl):03d} "
                    f"loss={float(loss.detach().item()):.5f} seg={float(seg_loss.detach().item()):.5f}",
                    flush=True,
                )
        print(
            f"epoch {epoch:02d}: loss={sum_loss/max(n_batches,1):.5f} "
            f"seg={sum_seg/max(n_batches,1):.5f} l1={sum_l1/max(n_batches,1):.5f}",
            flush=True,
        )

    return model


def apply_model_to_odd_frames(model: nn.Module, raw_all: np.ndarray, batch_size: int = 16, mode: str = "delta"):
    """
    Apply low-res learned residual to full-res odd frames.
    We upsample residual to full res and add to original odd frames.
    """
    model.eval()
    odd = raw_all[1::2]  # (N,H,W,3)
    n = odd.shape[0]

    out_odd = np.empty_like(odd)
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            full = torch.from_numpy(odd[i:j]).permute(0, 3, 1, 2).to(DEVICE).float()
            low = F.interpolate(full, size=(MODEL_H, MODEL_W), mode="bilinear", align_corners=False)
            corr_low, _ = model(low)
            if mode == "replace":
                corr_full_f = F.interpolate(corr_low, size=(H_CAM, W_CAM), mode="bicubic", align_corners=False)
            else:
                delta_low = corr_low - low
                delta_full = F.interpolate(delta_low, size=(H_CAM, W_CAM), mode="bicubic", align_corners=False)
                corr_full_f = full + delta_full
            corr_full = corr_full_f.clamp(0, 255).round().to(torch.uint8)
            out_odd[i:j] = corr_full.permute(0, 2, 3, 1).cpu().numpy()

    raw_new = raw_all.copy()
    raw_new[1::2] = out_odd
    return raw_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--features", type=int, default=24)
    ap.add_argument("--max-delta", type=float, default=8.0)
    ap.add_argument("--boundary-boost", type=float, default=2.0)
    ap.add_argument("--l1-lambda", type=float, default=0.02)
    ap.add_argument("--train-k", type=int, default=220, help="Train only top-K worst seg frames (0=all)")
    ap.add_argument("--log-every", type=int, default=3, help="Mini-batch log interval")
    ap.add_argument("--apply-mode", type=str, default="delta", choices=["delta", "replace"])
    args = ap.parse_args()

    print(f"Device: {DEVICE}", flush=True)
    x_train, y_train, raw_all = load_training_tensors()
    model = train_model(args, x_train, y_train)

    model_bytes = save_int8_bz2(model, MODEL_OUT)
    print(f"Saved model sidecar: {MODEL_OUT} ({model_bytes} bytes)", flush=True)

    raw_new = apply_model_to_odd_frames(model, raw_all, batch_size=args.batch_size, mode=args.apply_mode)
    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    raw_new.tofile(RAW_OUT)
    print(f"Wrote corrected raw: {RAW_OUT}", flush=True)

    archive_bytes = int(ARCHIVE_ZIP.stat().st_size)
    total_bytes = archive_bytes + model_bytes
    rate = total_bytes / UNCOMPRESSED_BYTES
    print(
        f"Size accounting: archive={archive_bytes} + ren={model_bytes} -> total={total_bytes} "
        f"(25*rate={25*rate:.4f})",
        flush=True,
    )

    cmd = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "submissions.evenframe_meta_v1.fast_eval",
        str(RAW_OUT),
        str(total_bytes),
    ]
    print("Running fast_eval with size penalty included...", flush=True)
    subprocess.run(cmd, check=False, cwd=str(ROOT))


if __name__ == "__main__":
    main()

