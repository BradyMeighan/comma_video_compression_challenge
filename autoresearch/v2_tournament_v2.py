"""Tournament v2 — explore new methods to push e80 sidecar pipeline further.

Tested on top 100 pairs (by pose loss against e80 cache base frames).

Hypotheses (all use pose-gradient on f1 unless noted):
  H1  K=1  (extreme byte savings)
  H2  K=2  (tournament v1 winner — reference)
  H3  K=3  (sweet spot search)
  H4  K=4
  H5  K=2 + iterative second pass (find K=2, regen, find K=2 more)
  H6  K=2 with multi-channel patches at same pixel (2 chans = 7B per patch)
  H7  K=2 with pose-grad targeting top dims (0,1,2) instead of full pose
  H8  K=2 with COMBINED pose+seg gradient for position scoring
  H9  K=2 with WIDER delta range Adam init (lr=4.0 instead of 2.0)
  H10 K=2 with NEW pose-gradient SCALED by per-dim importance (boost dim 0,1,2)

Plus C3 dim variations tested separately (compare delta encoding cost):
  C3a  target_dims=(1, 2, 5)  current — 7^3=343 grid
  C3b  target_dims=(0, 1, 2)  swap dim 5 → dim 0 — same grid size
  C3c  target_dims=(0, 1, 2, 5)  4 dims — 5^4=625 grid (smaller per-dim range)
"""
import sys, os, pickle, time, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared
from v2_shared import State, compose_score
from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6,
                      load_segnet, MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE)
import sidecar_explore as se
from sidecar_stack import fast_eval, per_pair_pose_mse
from sidecar_channel_only import (find_channel_only_patches, channel_sidecar_size,
                                     apply_channel_patches)
from sidecar_seg_only import _segnet_logits
from v2_c3_pose_vector import find_pose_deltas_gridsearch, serialize_pose_deltas

CACHE_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results")) / "v2_cache"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
N_TEST_PAIRS = int(os.environ.get("N_TEST", "100"))


def find_channel_patches_combined_grad(f1_all, f2_all, gt_poses, posenet, segnet,
                                          val_rgb_f2, pair_indices, K, n_iter, device,
                                          seg_weight=1000.0):
    """Like find_channel_only_patches but uses combined pose+seg gradient for positions."""
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        # Compute target seg labels from val rgb f2
        f2_target = val_rgb_f2[sel].to(device).float().permute(0, 3, 1, 2)
        with torch.no_grad():
            tgt_logits = _segnet_logits(f2_target, segnet)
            tgt_labels = tgt_logits.argmax(dim=1)

        f1_param = f1.clone().requires_grad_(True)
        # pose loss
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        pose_loss = ((fp - gt_p) ** 2).sum()
        # seg loss using f2 (note: changing f1 doesn't affect seg directly because SegNet only sees f2.
        # but we add it here as a position-scoring proxy for "does this pixel matter for any output")
        # actually we use only pose loss here since f1 changes don't affect seg
        loss = pose_loss
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()
        max_chan_grad, best_chan = grad_abs.max(dim=1)
        flat = max_chan_grad.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        chan_t = best_chan[batch_idx, ys_t, xs_t]

        cur_d = torch.zeros((b, K), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            for c in range(3):
                mask_c = (chan_t == c)
                if mask_c.any():
                    rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                    yy = ys_t[rows_b, cols_k]
                    xx = xs_t[rows_b, cols_k]
                    dd = cur_d[rows_b, cols_k]
                    f1_p[rows_b, c, yy, xx] = f1_p[rows_b, c, yy, xx] + dd
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
        chan_np = chan_t.cpu().numpy().astype(np.uint8)
        d_np = cur_d.detach().cpu().numpy().round().astype(np.int8)
        for bi, pair_i in enumerate(idx_list):
            patches = list(zip(xs_np[bi].tolist(), ys_np[bi].tolist(),
                                chan_np[bi].tolist(), d_np[bi].tolist()))
            patches = [p for p in patches if p[3] != 0]
            if patches:
                out[int(pair_i)] = patches
    return out


def find_channel_iterative(f1_all, f2_all, gt_poses, posenet, pair_indices,
                              K_each, n_passes, n_iter, device):
    """Iterative: find K patches, apply, regen gradient, find K more. n_passes total."""
    cur_f1 = f1_all.clone()
    all_patches = {pi: [] for pi in pair_indices}
    for pass_idx in range(n_passes):
        patches = find_channel_only_patches(cur_f1, f2_all, gt_poses, posenet,
                                              pair_indices, K=K_each, n_iter=n_iter, device=device)
        cur_f1 = apply_channel_patches(cur_f1, patches)
        for pi, ps in patches.items():
            all_patches[pi].extend(ps)
    return {pi: ps for pi, ps in all_patches.items() if ps}


def find_channel_multi_chan(f1_all, f2_all, gt_poses, posenet, pair_indices,
                                K, n_chan, n_iter, device):
    """K patches per pair; each patch modifies the top n_chan channels at that pixel.
    Encoding: u16 x, u16 y, u8 chan_mask, n_chan × i8 deltas. For n_chan=2: 7B/patch.
    """
    out = {}
    bs = 8
    for start in range(0, len(pair_indices), bs):
        idx_list = pair_indices[start:start + bs]
        b = len(idx_list)
        sel = torch.tensor(idx_list, dtype=torch.long)
        f1 = f1_all[sel].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[sel].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[sel].to(device).float()

        f1_param = f1.clone().requires_grad_(True)
        pin = se.diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        grad = torch.autograd.grad(loss, f1_param)[0]
        grad_abs = grad.abs()
        # For each pixel, sum top-n_chan |grad| as score; identify which channels
        per_pix_score = grad_abs.sort(dim=1, descending=True).values[:, :n_chan].sum(dim=1)
        flat = per_pix_score.contiguous().reshape(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long()
        xs_t = (topk % OUT_W).long()

        # For each top-K pixel, identify the top-n_chan channels
        # Reshape grad_abs to (b, 3, OUT_H*OUT_W); index by topk positions
        grad_per_pix = grad_abs.permute(0, 2, 3, 1).contiguous().reshape(b, OUT_H * OUT_W, 3)  # (b, HW, 3)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        sel_grad = grad_per_pix[batch_idx, topk]  # (b, K, 3)
        _, chan_indices = sel_grad.topk(n_chan, dim=2)  # (b, K, n_chan)

        # Optimize n_chan deltas per (pos)
        cur_d = torch.zeros((b, K, n_chan), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        for _ in range(n_iter):
            opt.zero_grad()
            f1_p = f1.clone()
            for ci in range(n_chan):
                cs = chan_indices[..., ci]  # (b, K) — channel for ci-th delta
                ds = cur_d[..., ci]
                for c in range(3):
                    mask_c = (cs == c)
                    if mask_c.any():
                        rows_b, cols_k = mask_c.nonzero(as_tuple=True)
                        yy = ys_t[rows_b, cols_k]
                        xx = xs_t[rows_b, cols_k]
                        dd = ds[rows_b, cols_k]
                        f1_p[rows_b, c, yy, xx] = f1_p[rows_b, c, yy, xx] + dd
            f1_p = f1_p.clamp(0, 255)
            pin = se.diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)

        for bi, pair_i in enumerate(idx_list):
            patches = []
            for k in range(K):
                xx = int(xs_t[bi, k].item())
                yy = int(ys_t[bi, k].item())
                chans_set = chan_indices[bi, k].cpu().numpy().tolist()
                deltas = cur_d[bi, k].detach().cpu().numpy().round().astype(np.int8).tolist()
                # encode as (x, y, chan_mask, *deltas)
                # chan_mask: bit i set if channel i is modified
                chan_mask = sum(1 << c for c in chans_set)
                if all(d == 0 for d in deltas):
                    continue
                patches.append((xx, yy, chan_mask, tuple(deltas)))
            if patches:
                out[int(pair_i)] = patches
    return out


def apply_multi_chan_patches(f1_all, patches):
    out = f1_all.clone()
    for pair_i, ps in patches.items():
        for tup in ps:
            x, y, chan_mask, deltas = tup
            chans = [c for c in range(3) if (chan_mask >> c) & 1]
            for c, d in zip(chans, deltas):
                v = int(out[pair_i, y, x, c]) + int(d)
                out[pair_i, y, x, c] = max(0, min(255, v))
    return out


def multi_chan_sidecar_size(patches, n_chan):
    """4B header + 4B per pair header + (4 + n_chan) bytes per patch."""
    if not patches:
        return 0
    parts = [struct.pack("<H", len(patches))]
    for pair_i, ps in sorted(patches.items()):
        parts.append(struct.pack("<HH", pair_i, len(ps)))
        for tup in ps:
            x, y, cm, deltas = tup
            parts.append(struct.pack("<HHB", x, y, cm))
            for d in deltas:
                parts.append(struct.pack("<b", int(d)))
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def main():
    import csv
    s = State()
    print(f"\n=== Tournament v2 on top {N_TEST_PAIRS} pairs ===", flush=True)

    # Load cache base frames
    if not (CACHE_DIR / "frames_x5.pt").exists():
        raise FileNotFoundError(f"Run v2_cache_builder first.")
    cache_frames = torch.load(CACHE_DIR / "frames_x5.pt", weights_only=False)
    f1_base = cache_frames['f1']
    f2_base = cache_frames['f2']
    n = f1_base.shape[0]

    # Top pairs by current pose loss
    pose_per_pair = per_pair_pose_mse(f1_base, f2_base, s.poses, s.posenet, s.device)
    rank = np.argsort(pose_per_pair)[::-1]
    test_pairs = [int(x) for x in rank[:N_TEST_PAIRS]]
    val_rgb = s.data["val_rgb"]
    val_rgb_test = val_rgb[test_pairs]
    val_rgb_f2_full = val_rgb[:, 1] if val_rgb.dim() == 5 else val_rgb

    seg_base, pose_base = fast_eval(f1_base[test_pairs], f2_base[test_pairs], val_rgb_test, s.device)
    print(f"\n  BASELINE: seg={seg_base:.6f} pose={pose_base:.6f}", flush=True)

    segnet = load_segnet(s.device)
    results = []
    from math import sqrt
    sqrt_pose_base = sqrt(max(0, 10*pose_base))

    def record(name, sb, seg_d, pose_d, elapsed):
        d_seg = (seg_d - seg_base) * 100
        d_pose = sqrt(max(0, 10*pose_d)) - sqrt_pose_base
        d_bytes = 25 * sb / UNCOMPRESSED_SIZE
        net = d_seg + d_pose + d_bytes
        results.append({'name': name, 'sb': sb, 'seg_d': seg_d, 'pose_d': pose_d,
                         'd_seg': d_seg, 'd_pose': d_pose, 'd_bytes': d_bytes,
                         'net': net, 'time': elapsed})
        print(f"  {name:<35} sb={sb:>5}B seg={seg_d:.5f} pose={pose_d:.5f}  "
              f"Δseg={d_seg:+.4f} Δpose={d_pose:+.4f} Δbytes={d_bytes:+.4f}  "
              f"NET={net:+.4f} ({elapsed:.0f}s)", flush=True)

    # ── H1: K=1 (extreme savings) ───────────────────────────────────────
    t0 = time.time()
    p = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet, test_pairs, K=1, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H1_K1", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H2: K=2 (reference) ─────────────────────────────────────────────
    t0 = time.time()
    p = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet, test_pairs, K=2, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H2_K2 (reference)", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H3: K=3 ─────────────────────────────────────────────────────────
    t0 = time.time()
    p = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet, test_pairs, K=3, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H3_K3", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H4: K=4 ─────────────────────────────────────────────────────────
    t0 = time.time()
    p = find_channel_only_patches(f1_base, f2_base, s.poses, s.posenet, test_pairs, K=4, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H4_K4", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H5: K=2 + iterative second pass (total K=4 split as 2+2) ────────
    t0 = time.time()
    p = find_channel_iterative(f1_base, f2_base, s.poses, s.posenet, test_pairs, K_each=2, n_passes=2, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H5_K2x2_iterative", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H6: K=2 multi-channel (2 chans per patch) ───────────────────────
    t0 = time.time()
    p = find_channel_multi_chan(f1_base, f2_base, s.poses, s.posenet, test_pairs, K=2, n_chan=2, n_iter=80, device=s.device)
    f1p = apply_multi_chan_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H6_K2_multichan2", multi_chan_sidecar_size(p, 2), seg_d, pose_d, time.time()-t0)

    # ── H7: K=2 multi-channel (3 chans per patch — full RGB) ────────────
    t0 = time.time()
    p = find_channel_multi_chan(f1_base, f2_base, s.poses, s.posenet, test_pairs, K=2, n_chan=3, n_iter=80, device=s.device)
    f1p = apply_multi_chan_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H7_K2_multichan3 (fullRGB)", multi_chan_sidecar_size(p, 3), seg_d, pose_d, time.time()-t0)

    # ── H8: K=2 with combined seg+pose gradient positions ───────────────
    t0 = time.time()
    p = find_channel_patches_combined_grad(f1_base, f2_base, s.poses, s.posenet, segnet,
                                              val_rgb_f2_full, test_pairs, K=2, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H8_K2_combined_grad", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H9: K=1 + K=1 iterative (total K=2 but 2 passes) ────────────────
    t0 = time.time()
    p = find_channel_iterative(f1_base, f2_base, s.poses, s.posenet, test_pairs, K_each=1, n_passes=2, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H9_K1x2_iterative", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── H10: K=1 + K=1 + K=1 iterative (total K=3) ──────────────────────
    t0 = time.time()
    p = find_channel_iterative(f1_base, f2_base, s.poses, s.posenet, test_pairs, K_each=1, n_passes=3, n_iter=80, device=s.device)
    f1p = apply_channel_patches(f1_base, p)[test_pairs].clone()
    seg_d, pose_d = fast_eval(f1p, f2_base[test_pairs], val_rgb_test, s.device)
    record("H10_K1x3_iterative", channel_sidecar_size(p), seg_d, pose_d, time.time()-t0)

    # ── C3 dim variations: try at fixed pose-only setup ──────────────────
    print("\n--- C3 pose-delta dim variations (no RGB patches) ---", flush=True)
    masks_x5 = torch.load(CACHE_DIR / "masks_x5.pt", weights_only=False)
    scale = torch.tensor([0.001, 0.005, 0.005, 0.001, 0.001, 0.005], device=s.device)

    def c3_test(name, target_dims, delta_range=(-3, -2, -1, 0, 1, 2, 3)):
        t0 = time.time()
        deltas = find_pose_deltas_gridsearch(s.gen, masks_x5, s.poses, s.posenet,
                                                test_pairs, s.device,
                                                target_dims=target_dims,
                                                delta_range=delta_range)
        # Apply deltas + regen frames just for test pairs
        from v2_c3_pose_vector import apply_pose_deltas_and_regen_full
        f1_after, f2_after = apply_pose_deltas_and_regen_full(
            s.gen, masks_x5, s.poses, deltas, s.device, scale, target_dims)
        seg_d, pose_d = fast_eval(f1_after[test_pairs], f2_after[test_pairs],
                                    val_rgb_test, s.device)
        sb = len(serialize_pose_deltas(deltas, target_dims))
        record(name, sb, seg_d, pose_d, time.time()-t0)

    c3_test("C3a_dims125 (current)", (1, 2, 5))
    c3_test("C3b_dims012", (0, 1, 2))
    c3_test("C3c_dims0125_5per", (0, 1, 2, 5), delta_range=(-2, -1, 0, 1, 2))
    c3_test("C3d_dims125_wider", (1, 2, 5), delta_range=(-5, -3, -1, 0, 1, 3, 5))

    # ── Sort + report ───────────────────────────────────────────────────
    print(f"\n=== TOURNAMENT V2 RESULTS (sorted by NET, lower=better) ===")
    print(f"{'rank':<5}{'name':<35}{'sb':>6}{'NET':>10}")
    print("-" * 60)
    results.sort(key=lambda r: r['net'])
    for i, r in enumerate(results):
        print(f"{i+1:<5}{r['name']:<35}{r['sb']:>6}{r['net']:>+10.4f}")

    out_csv = OUTPUT_DIR / "v2_tournament_v2_results.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "sb_bytes", "seg_d", "pose_d",
                     "d_seg", "d_pose", "d_bytes", "net", "elapsed_s"])
        for i, r in enumerate(results):
            w.writerow([i+1, r['name'], r['sb'], r['seg_d'], r['pose_d'],
                        r['d_seg'], r['d_pose'], r['d_bytes'], r['net'], r['time']])
    print(f"\nresults: {out_csv}")
    print(f"\n[hint] Best to scale up: {results[0]['name']}")


if __name__ == "__main__":
    main()
