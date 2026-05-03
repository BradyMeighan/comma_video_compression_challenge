"""Seg-targeted RGB patches on FRAME 2.

Why frame 2: SegNet's preprocess_input does `x = x[:, -1, ...]` (uses ONLY the
last frame). So all our existing pose-targeted patches on f1 are invisible to
SegNet. Patches on f2 affect SegNet's input directly.

Method: cross-entropy proxy gradient through SegNet → pick top-K pixels by
max-channel |grad|, optimize int8 delta on chosen channel via Adam.

Format identical to channel-only patches: u16 x, u16 y, u8 channel, i8 delta = 6B.
"""
import sys, os, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import OUT_H, OUT_W, MODEL_H, MODEL_W, load_segnet


def _segnet_input(f2_bchw, segnet):
    """Apply SegNet's preprocess_input to a batch of f2 frames (already CHW float).
    The input arg is the f2 only; we wrap it as (b, 1, c, h, w) since SegNet
    indexes [:, -1, ...]."""
    x = f2_bchw.unsqueeze(1)   # (b, 1, c, h, w)
    x = x[:, -1, ...]          # → (b, c, h, w)
    return F.interpolate(x, size=(MODEL_H, MODEL_W), mode='bilinear')


def _segnet_logits(f2_bchw, segnet):
    """Run SegNet forward on f2 frames (b, c, h, w), return logits (b, 5, h_seg, w_seg)."""
    return segnet(_segnet_input(f2_bchw, segnet))


def find_seg_patches_f2(f2_pred, f2_target, segnet, pair_indices,
                          K, n_iter, device, lr=2.0):
    """Find K patches per pair on f2 to reduce SegNet argmax mismatch vs target.

    f2_pred, f2_target: (n_pairs, OUT_H, OUT_W, 3) uint8 — predicted & target frame 2
    segnet: frozen SegNet
    pair_indices: list of pair_i to optimize
    K: patches per pair
    n_iter: Adam iters per batch

    Returns dict pair_i → [(x, y, channel, delta), ...]
    """
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f2p = f2_pred[sel].to(device).float().permute(0, 3, 1, 2)   # (b, 3, OUT_H, OUT_W)
        f2t = f2_target[sel].to(device).float().permute(0, 3, 1, 2)

        # Compute target seg labels (argmax of segnet on target frame)
        with torch.no_grad():
            tgt_logits = _segnet_logits(f2t, segnet)
            tgt_labels = tgt_logits.argmax(dim=1)  # (b, h_seg, w_seg)

        # Initial gradient: cross-entropy of segnet(f2_pred) vs target labels
        f2_param = f2p.clone().requires_grad_(True)
        pred_logits = _segnet_logits(f2_param, segnet)
        ce_loss = F.cross_entropy(pred_logits, tgt_labels, reduction='sum')
        grad = torch.autograd.grad(ce_loss, f2_param)[0]  # (b, 3, OUT_H, OUT_W)
        grad_abs = grad.abs()

        # Pick top-K pixel positions by max-channel grad
        max_chan_grad, best_chan = grad_abs.max(dim=1)   # (b, OUT_H, OUT_W)
        flat = max_chan_grad.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        chan_t = best_chan[batch_idx, ys_t, xs_t]  # (b, K)

        # Optimize delta per (pos, channel)
        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=lr)
        for _ in range(n_iter):
            opt.zero_grad()
            f2_p = f2p.clone()
            for c in range(3):
                mask_c = (chan_t == c)
                if mask_c.any():
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    yy = ys_t[rows_b, cols_k]
                    xx = xs_t[rows_b, cols_k]
                    dd = cur_d[rows_b, cols_k]
                    f2_p[rows_b, c, yy, xx] = f2_p[rows_b, c, yy, xx] + dd
            f2_p = f2_p.clamp(0, 255)
            pred_logits = _segnet_logits(f2_p, segnet)
            loss = F.cross_entropy(pred_logits, tgt_labels, reduction='sum')
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)

        xs_np = xs_t.cpu().numpy().astype(np.uint16)
        ys_np = ys_t.cpu().numpy().astype(np.uint16)
        chan_np = chan_t.cpu().numpy().astype(np.uint8)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
            # filter out zero-delta patches (no improvement)
            patches = [p for p in patches if p[3] != 0]
            if patches:
                out[int(pair_i)] = patches
    return out


def seg_patches_sidecar_size(patches):
    """6 bytes per patch, bz2 compressed."""
    if not patches:
        return 0, b''
    parts = [struct.pack("<H", len(patches))]
    for pair_i, ps in sorted(patches.items()):
        parts.append(struct.pack("<HH", pair_i, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    raw = b''.join(parts)
    comp = bz2.compress(raw, compresslevel=9)
    return len(comp), comp


def apply_seg_patches_f2(f2_all, patches):
    """Apply patches to a copy of f2_all (uint8 tensor), return new f2."""
    out = f2_all.clone()
    for pair_i, ps in patches.items():
        for (x, y, c, d) in ps:
            v = int(out[pair_i, y, x, c]) + int(d)
            out[pair_i, y, x, c] = max(0, min(255, v))
    return out
