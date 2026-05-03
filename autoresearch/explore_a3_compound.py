#!/usr/bin/env python
"""
A3: Compound channel-only patches (shared coordinate, two channels).

Currently each patch = (x, y, channel, delta) = 6 bytes.
Compound: when 2 channels at SAME (x, y) both have signal, store as
(x, y, channel_pair, delta1, delta2) — saves 1 coordinate, costs 1 extra
delta. Net savings per compound patch: 4B vs 12B for two independent.

Storage: u16 x, u16 y, u8 chan_pair_id (0..6 for valid combos), i8 d1, i8 d2 = 7B
  vs 12B for 2 independent patches → 42% saving.

Combos: chan_pair encodes which 2 channels (RG=0, RB=1, GB=2, R only=3, G=4, B=5, RGB=6 with 3 deltas)
For simplicity start with just pair vs single distinction.
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, get_pose6, load_posenet, estimate_model_bytes)
from train import Generator, load_data_full
import sidecar_explore as se
from sidecar_stack import (get_dist_net, fast_eval, fast_compose)
from sidecar_mask_verified import mask_sidecar_size

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_compound_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                            K, n_iter, device):
    """Find K positions, optimize ALL 3 channel deltas at each, but at storage time,
    only keep channels with significant magnitude. Compound storage when 2 channels are kept.

    Returns dict pair_i -> list of (x, y, type, deltas)
      type=0: single channel (x, y, chan, d1)
      type=1: two-channel  (x, y, chan_pair_id, d1, d2)
      type=2: three-channel (x, y, _, d1, d2, d3)
    """
    out = {}
    bs = 4
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Initial gradient: pick top-K by sum-channel |grad|
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()
        flat = grad_abs.sum(dim=1).contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)

        # Optimize ALL 3 channels per position
        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            for c in range(3):
                f1_p[batch_idx, c, ys_t, xs_t] = f1_p[batch_idx, c, ys_t, xs_t] + cur_d[..., c]
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)

        xs_np = xs_t.cpu().numpy().astype(np.uint16)
        ys_np = ys_t.cpu().numpy().astype(np.uint16)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        # For each position, decide: 1, 2, or 3 channels (keep those with |d| >= threshold=2)
        for bi, pair_i in enumerate(idx_list):
            patches = []
            for k in range(K):
                ds = d_np[bi, k]
                kept_chans = [(c, int(ds[c])) for c in range(3) if abs(ds[c]) >= 2]
                if len(kept_chans) == 0:
                    continue
                elif len(kept_chans) == 1:
                    c, d = kept_chans[0]
                    patches.append((int(xs_np[bi, k]), int(ys_np[bi, k]), 0, c, d, 0, 0))
                elif len(kept_chans) == 2:
                    c1, d1 = kept_chans[0]; c2, d2 = kept_chans[1]
                    pair_code = c1 * 3 + c2  # encodes (c1, c2)
                    patches.append((int(xs_np[bi, k]), int(ys_np[bi, k]), 1, pair_code, d1, d2, 0))
                else:
                    patches.append((int(xs_np[bi, k]), int(ys_np[bi, k]), 2, 0,
                                     int(ds[0]), int(ds[1]), int(ds[2])))
            out[pair_i] = patches
    return out


def compound_sidecar_size(patches):
    """Variable-length per patch:
      type 0: u16 x, u16 y, u8 (type<<6 | channel) -> 5B
      type 1: u16 x, u16 y, u8 (type<<6 | pair_code), i8 d1, i8 d2 -> 7B
      type 2: u16 x, u16 y, u8 (type<<6), i8 d1, i8 d2, i8 d3 -> 8B
    """
    if not patches:
        return 0
    parts = [struct.pack("<H", len(patches))]
    for pi in sorted(patches.keys()):
        ps = patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for tup in ps:
            x, y, typ, code, d1, d2, d3 = tup
            if typ == 0:
                parts.append(struct.pack("<HHBb", x, y, (typ << 6) | (code & 0x3F), d1))
            elif typ == 1:
                parts.append(struct.pack("<HHBbb", x, y, (typ << 6) | (code & 0x3F), d1, d2))
            else:
                parts.append(struct.pack("<HHBbbb", x, y, (typ << 6), d1, d2, d3))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def apply_compound_patches(f_all, patches):
    out = f_all.clone()
    H, W = out.shape[1], out.shape[2]
    for pi in patches:
        arr = out[pi].float().numpy()
        for tup in patches[pi]:
            x, y, typ, code, d1, d2, d3 = tup
            if typ == 0:
                arr[y, x, code] += d1
            elif typ == 1:
                c1 = code // 3; c2 = code % 3
                arr[y, x, c1] += d1; arr[y, x, c2] += d2
            else:
                arr[y, x, 0] += d1; arr[y, x, 1] += d2; arr[y, x, 2] += d3
        out[pi] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def main():
    device = torch.device("cuda")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
    f1_new, f2_new = bf['f1_new'], bf['f2_new']
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    sb_mask = bp['sb_mask_bz2']
    score_bl = bp['score']

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    poses = data["val_poses"]

    print(f"Baseline: {score_bl:.4f}, sb_mask={sb_mask}B")
    print("\n=== A3: compound channel patches 250_K5+250_K2 ===")
    t0 = time.time()
    p_top = find_compound_patches(f1_new, f2_new, poses, posenet,
                                    [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_compound_patches(f1_new, f2_new, poses, posenet,
                                     [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches = {**p_top, **p_tail}
    elapsed = time.time() - t0

    n_total = sum(len(v) for v in rgb_patches.values())
    n_t0 = sum(1 for v in rgb_patches.values() for t in v if t[2] == 0)
    n_t1 = sum(1 for v in rgb_patches.values() for t in v if t[2] == 1)
    n_t2 = sum(1 for v in rgb_patches.values() for t in v if t[2] == 2)
    print(f"Patch types: 1ch={n_t0} 2ch={n_t1} 3ch={n_t2} total={n_total}")
    sb_rgb = compound_sidecar_size(rgb_patches)
    f1_combined = apply_compound_patches(f1_new, rgb_patches)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb)
    print(f"A3: sb_mask={sb_mask}B sb_rgb={sb_rgb}B sb_total={sb_mask+sb_rgb}B "
          f"score={full['score']:.4f} delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "a3_compound_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "n_patches", "n_1ch", "n_2ch", "n_3ch", "sb_total", "score", "delta"])
        w.writerow(["a3_compound_250_K5+250_K2", n_total, n_t0, n_t1, n_t2,
                    sb_mask+sb_rgb, full['score'], full['score']-score_bl])


if __name__ == "__main__":
    main()
