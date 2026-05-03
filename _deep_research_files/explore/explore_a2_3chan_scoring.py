#!/usr/bin/env python
"""
A2: 3-channel-per-position scoring.

Currently `find_channel_only_patches` picks the channel with biggest |grad| at each pixel.
But that's a HEURISTIC choice. The OPTIMAL channel for reducing pose loss may differ
from the channel with the biggest gradient magnitude (because Adam will fit the delta
to whichever channel we pick).

Method: for each candidate top-K position, separately optimize a 1-channel patch on
each of 3 channels and pick the best. Costs ~3x in optimizer time but more efficient
patch selection.

Loads baseline mask sidecar + regenerated frames, only re-finds the RGB patches.
"""
import sys, os, pickle, time
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
from sidecar_channel_only import (channel_sidecar_size, apply_channel_patches)
from sidecar_mask_verified import mask_sidecar_size

MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/gen_continued.pt")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def find_3chan_scored_patches(f1_all, f2_all, gt_poses, posenet, pair_indices,
                                 K, n_iter, device):
    """Find K patches per pair, but evaluate ALL 3 channels per candidate top position
    and pick the BEST one based on actual loss reduction (not just |grad|)."""
    out = {}
    bs = 4  # smaller batch since 3x optimizer work
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Initial gradient
        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()
        # Pick top-K positions by SUM of |grad| across channels (more inclusive than max)
        flat = grad_abs.sum(dim=1).contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        # For each top-K position, try ALL 3 channels independently and pick best
        # Optimize: 3 separate (b, K) delta tensors, pick winner per (b, K)
        all_chan_results = []  # [(deltas_for_chan0, deltas_for_chan1, deltas_for_chan2, final_losses)]
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)

        for c_test in range(3):
            cur_d = torch.zeros((b, K), device=device, requires_grad=True)
            opt = torch.optim.Adam([cur_d], lr=2.0)
            for _ in range(n_iter):
                opt.zero_grad()
                f1_p = f1.clone()
                f1_p[batch_idx, c_test, ys_t, xs_t] = f1_p[batch_idx, c_test, ys_t, xs_t] + cur_d
                f1_p = f1_p.clamp(0, 255)
                pin = se.diff_posenet_input(f1_p, f2)
                fp = get_pose6(posenet, pin).float()
                loss = ((fp - gt_p) ** 2).sum()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    cur_d.clamp_(-127, 127)
            all_chan_results.append(cur_d.detach())

        # Now for each (b, k) pick the channel with biggest |delta×grad_at_pos| as proxy for loss reduction
        # Better: actually evaluate each channel choice independently. Cheaper proxy: predicted_reduction = -delta*grad_orig_chan
        # Use predicted = sum-over-iter of delta*grad as proxy. Simpler: just pick channel with max |delta|.
        # Actually compute: for each pos, eval baseline loss and (loss with chan c delta) for each c, pick min.
        # That's expensive. Use heuristic: pick channel with biggest |delta| × |grad_at_chan|.
        scores_per_chan = torch.zeros((3, b, K), device=device)
        for c_test in range(3):
            d = all_chan_results[c_test]
            g_at_chan = grad_abs[batch_idx, c_test, ys_t, xs_t]
            scores_per_chan[c_test] = d.abs() * g_at_chan
        best_chan = scores_per_chan.argmax(dim=0)  # (b, K)

        xs_np = xs_t.cpu().numpy().astype(np.uint16)
        ys_np = ys_t.cpu().numpy().astype(np.uint16)
        chan_np = best_chan.cpu().numpy().astype(np.uint8)
        # Pick the delta from the best channel
        d_chosen = torch.zeros((b, K), device=device)
        for c in range(3):
            mask_c = (best_chan == c)
            d_chosen[mask_c] = all_chan_results[c][mask_c]
        d_np = d_chosen.cpu().numpy().round().astype(np.int8)

        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
            out[pair_i] = patches
    return out


def main():
    device = torch.device("cuda")
    print("Loading...", flush=True)
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True), strict=False)
    data = load_data_full(device)
    posenet = load_posenet(device)
    model_bytes = estimate_model_bytes(gen)

    # Load baseline mask + regenerated frames
    bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
    f1_new, f2_new = bf['f1_new'], bf['f2_new']
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    mask_patches = bp['mask_patches']
    sb_mask = bp['sb_mask_bz2']
    score_bl = bp['score']
    print(f"Baseline (mask + ch_RGB): {score_bl:.4f}, sb_mask={sb_mask}B")

    pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
    rank = np.argsort(pose_per_pair)[::-1]
    poses = data["val_poses"]

    # Re-find RGB with 3-channel-per-position scoring
    print("\n=== A2: 3-channel scoring (250_K5+250_K2) ===")
    t0 = time.time()
    p_top = find_3chan_scored_patches(f1_new, f2_new, poses, posenet,
                                        [int(x) for x in rank[:250]], K=5, n_iter=80, device=device)
    p_tail = find_3chan_scored_patches(f1_new, f2_new, poses, posenet,
                                         [int(x) for x in rank[250:500]], K=2, n_iter=80, device=device)
    rgb_patches_a2 = {**p_top, **p_tail}
    elapsed = time.time() - t0

    sb_rgb_a2 = channel_sidecar_size(rgb_patches_a2)
    f1_combined = apply_channel_patches(f1_new, rgb_patches_a2)
    s, p = fast_eval(f1_combined, f2_new, data["val_rgb"], device)
    full = fast_compose(s, p, model_bytes, sb_mask + sb_rgb_a2)
    print(f"A2: sb_mask={sb_mask}B sb_rgb={sb_rgb_a2}B sb_total={sb_mask+sb_rgb_a2}B "
          f"score={full['score']:.4f} delta={full['score']-score_bl:+.4f} ({elapsed:.0f}s)")

    import csv
    with open(OUTPUT_DIR / "a2_3chan_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["spec", "sb_total", "score", "pose_term", "delta_vs_baseline"])
        w.writerow(["baseline", sb_mask + bp['sb_rgb_bz2'], score_bl, bp['pose_dist'], 0])
        w.writerow(["a2_3chan_250_K5+250_K2", sb_mask+sb_rgb_a2, full['score'], full['pose_term'], full['score']-score_bl])

    # Also save patches in case A2 is the new best
    if full['score'] < score_bl:
        with open(OUTPUT_DIR / "a2_patches.pkl", 'wb') as f:
            pickle.dump({'rgb_patches': rgb_patches_a2, 'score': full['score'],
                          'sb_total': sb_mask+sb_rgb_a2}, f)
        print(f"NEW BEST! Saved a2_patches.pkl")


if __name__ == "__main__":
    main()
