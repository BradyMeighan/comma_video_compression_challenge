#!/usr/bin/env python
"""
Sidecar exploration: test multiple sidecar strategies on a trained model.

For each strategy:
  1. Generate frames from model
  2. Compute small pixel-patch sidecar (K points per pair, each with RGB delta)
  3. Apply patches at "decode time"
  4. Re-evaluate score
  5. Measure actual sidecar bytes after compression

Output: CSV showing method, K, sidecar_bytes, score, seg, pose, rate

Env vars:
  MODEL_PATH    : path to gen.pt (REQUIRED)
  OUTPUT_DIR    : where to save results (default: autoresearch/sidecar_results)
  N_ITER        : gradient steps per pair (default: 30)
"""
import sys, os, time, csv, struct, bz2
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import einops

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1")
os.environ.setdefault("CONFIG", "B")

from prepare import (load_data, evaluate, gpu_cleanup, MODEL_H, MODEL_W, OUT_H, OUT_W,
                     diff_round, pack_pair_yuv6, get_pose6, kl_on_logits, diff_rgb_to_yuv6,
                     load_segnet, load_posenet, MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE)
from train import Generator, load_data_full
import math


def diff_posenet_input(f1, f2):
    """Differentiable replacement for posenet.preprocess_input. f1, f2: (B, 3, H, W)."""
    f1_r = F.interpolate(f1, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
    f2_r = F.interpolate(f2, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
    return pack_pair_yuv6(f1_r, f2_r)

MODEL_PATH = os.environ.get("MODEL_PATH", "")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
N_ITER = int(os.environ.get("N_ITER", "30"))

if not MODEL_PATH or not Path(MODEL_PATH).exists():
    print(f"ERROR: MODEL_PATH not found: '{MODEL_PATH}'")
    sys.exit(1)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
csv_path = OUTPUT_DIR / "sidecar_results.csv"


# ── Generate all model output frames ONCE (caches) ──
def generate_all_frames(gen, data, device, batch_size=8):
    """Run gen on all val pairs, return f1, f2 at OUT_H × OUT_W as uint8."""
    rgb = data["val_rgb"]
    masks = data["val_masks"]
    poses = data["val_poses"]
    n = rgb.shape[0]
    f1_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    f2_all = torch.zeros(n, OUT_H, OUT_W, 3, dtype=torch.uint8)
    gen.eval()
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            m = masks[i:i+batch_size].to(device).long()
            p = poses[i:i+batch_size].to(device).float()
            p1, p2 = gen(m, p)
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode="bilinear", align_corners=False)
            f1_all[i:i+batch_size] = f1u.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
            f2_all[i:i+batch_size] = f2u.clamp(0, 255).round().permute(0, 2, 3, 1).to(torch.uint8).cpu()
    return f1_all, f2_all


def eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=None):
    """Run distortion eval on already-generated frames. Returns dict like evaluate()."""
    from modules import DistortionNet
    dist_net = DistortionNet().eval().to(device)
    from prepare import segnet_sd_path, posenet_sd_path
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    n = f1_all.shape[0]
    t_seg, t_pose, t_n = 0.0, 0.0, 0
    bs = 16
    with torch.inference_mode():
        for i in range(0, n, bs):
            f1 = f1_all[i:i+bs].to(device)
            f2 = f2_all[i:i+bs].to(device)
            comp = torch.stack([f1, f2], dim=1)
            gt = data["val_rgb"][i:i+bs].to(device)
            pd, sd = dist_net.compute_distortion(gt, comp)
            t_seg += sd.sum().item()
            t_pose += pd.sum().item()
            t_n += gt.shape[0]
    del dist_net; gpu_cleanup()
    avg_seg, avg_pose = t_seg / t_n, t_pose / t_n

    # model_bytes (assume same as original — sidecars added separately)
    model_bytes = 0
    if gen_for_bytes is not None:
        from prepare import estimate_model_bytes
        model_bytes = estimate_model_bytes(gen_for_bytes)

    return {"seg_dist": avg_seg, "pose_dist": avg_pose, "model_bytes": model_bytes}


def compose_score(seg_dist, pose_dist, model_bytes, sidecar_bytes):
    total = MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes
    rate = total / UNCOMPRESSED_SIZE
    return {
        "score": 100*seg_dist + math.sqrt(max(0, 10*pose_dist)) + 25*rate,
        "seg_term": 100 * seg_dist,
        "pose_term": math.sqrt(max(0, 10*pose_dist)),
        "rate_term": 25 * rate,
        "model_bytes": model_bytes,
        "sidecar_bytes": sidecar_bytes,
    }


# ─── Sidecar strategies ───────────────────────────────────────────────────

def find_seg_patches_on_f2(f2_all, gt_masks, segnet, K, n_iter, device):
    """Find K pixel patches on frame2 that reduce segnet disagreement."""
    n = f2_all.shape[0]
    pts_xy = np.zeros((n, K, 2), dtype=np.uint16)
    pts_d = np.zeros((n, K, 3), dtype=np.int8)
    bs = 4
    for i in range(0, n, bs):
        e = min(i + bs, n); b = e - i
        f2 = f2_all[i:e].to(device).float().permute(0, 3, 1, 2)
        gt_cls = gt_masks[i:e].to(device).long()  # (b, MODEL_H, MODEL_W)

        # Initial gradient through segnet
        f2_param = f2.clone().requires_grad_(True)
        f2_r = F.interpolate(f2_param, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
        logits = segnet(f2_r).float()  # (b, 5, MODEL_H, MODEL_W)
        loss = F.cross_entropy(logits, gt_cls, reduction='sum')
        grad = torch.autograd.grad(loss, f2_param)[0]
        gmag = grad.abs().sum(dim=1)

        flat = gmag.view(b, -1)
        _, topk = torch.topk(flat, K, dim=1)
        ys_t = (topk // OUT_W).long(); xs_t = (topk % OUT_W).long()
        pts_xy[i:e, :, 0] = xs_t.cpu().numpy().astype(np.uint16)
        pts_xy[i:e, :, 1] = ys_t.cpu().numpy().astype(np.uint16)

        cur_d = torch.zeros((b, K, 3), device=device, requires_grad=True)
        opt = torch.optim.Adam([cur_d], lr=2.0)
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)
        for _ in range(n_iter):
            opt.zero_grad()
            f2_p = f2.clone()
            for c in range(3):
                f2_p[batch_idx, c, ys_t, xs_t] = f2_p[batch_idx, c, ys_t, xs_t] + cur_d[..., c]
            f2_p = f2_p.clamp(0, 255)
            f2_pr = F.interpolate(f2_p, size=(MODEL_H, MODEL_W), mode='bilinear', align_corners=False)
            logits = segnet(f2_pr).float()
            loss = F.cross_entropy(logits, gt_cls, reduction='sum')
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d.clamp_(-127, 127)
        pts_d[i:e] = cur_d.detach().cpu().numpy().round().astype(np.int8)
    return pts_xy, pts_d


def find_universal_patch(f1_all, f2_all, gt_poses, posenet, K, n_iter, device):
    """Single patch positions+deltas SHARED across all pairs (no per-pair coords)."""
    n = f1_all.shape[0]
    bs = 8
    # Step 1: aggregate gradient across all pairs to find universal hot spots
    grad_acc = torch.zeros((3, OUT_H, OUT_W), device=device)
    for i in range(0, n, bs):
        e = min(i + bs, n)
        f1 = f1_all[i:e].to(device).float().permute(0, 3, 1, 2)
        f2 = f2_all[i:e].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[i:e].to(device).float()
        f1_param = f1.clone().requires_grad_(True)
        pin = diff_posenet_input(f1_param, f2)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        g = torch.autograd.grad(loss, f1_param)[0]
        grad_acc += g.abs().sum(dim=0)
    gmag = grad_acc.sum(dim=0)
    flat = gmag.view(-1)
    _, topk = torch.topk(flat, K)
    ys_t = (topk // OUT_W).long()
    xs_t = (topk % OUT_W).long()
    pts_xy = np.zeros((1, K, 2), dtype=np.uint16)
    pts_xy[0, :, 0] = xs_t.cpu().numpy().astype(np.uint16)
    pts_xy[0, :, 1] = ys_t.cpu().numpy().astype(np.uint16)

    # Step 2: optimize a single shared delta over all pairs
    # FIX: backward per-batch (free graph), accumulate gradients in cur_d.grad
    cur_d = torch.zeros((K, 3), device=device, requires_grad=True)
    opt = torch.optim.Adam([cur_d], lr=2.0)
    for it in range(n_iter):
        opt.zero_grad()
        for i in range(0, n, bs):
            e = min(i + bs, n); b = e - i
            f1 = f1_all[i:e].to(device).float().permute(0, 3, 1, 2)
            f2 = f2_all[i:e].to(device).float().permute(0, 3, 1, 2)
            gt_p = gt_poses[i:e].to(device).float()
            f1_p = f1.clone()
            for c in range(3):
                f1_p[:, c, ys_t, xs_t] = f1_p[:, c, ys_t, xs_t] + cur_d[:, c]
            f1_p = f1_p.clamp(0, 255)
            pin = diff_posenet_input(f1_p, f2)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            loss.backward()  # backward per-batch, frees graph immediately
        opt.step()
        with torch.no_grad():
            cur_d.clamp_(-127, 127)
        if it % 10 == 0:
            print(f"    [universal K={K}] iter {it}/{n_iter}", flush=True)
    pts_d = np.zeros((1, K, 3), dtype=np.int8)
    pts_d[0] = cur_d.detach().cpu().numpy().round().astype(np.int8)
    return pts_xy, pts_d  # broadcast at apply time


def apply_universal_patch_to_frames(f1_all, pts_xy, pts_d):
    """Apply shared patch to all frames."""
    out = f1_all.clone()
    K = pts_xy.shape[1]
    H, W = out.shape[1], out.shape[2]
    n = out.shape[0]
    for i in range(n):
        arr = out[i].float().numpy()
        for j in range(K):
            x, y = int(pts_xy[0, j, 0]), int(pts_xy[0, j, 1])
            d = pts_d[0, j].astype(np.float32)
            if not d.any(): continue
            arr[y, x] += d
        out[i] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def find_pose_patches(f1_all, f2_all, gt_poses, posenet, K, n_iter, lr, device,
                      target='f1'):
    """For each pair: find K pixel points whose modification minimizes pose error.
    target: 'f1' (only frame1), 'f2' (only frame2), or 'both' (both frames).

    Returns:
        pts_xy: (N, K, 2) uint16 coords (full res)
        pts_d:  (N, K, 3) int8 deltas (frame1 deltas)
        pts_d2: (N, K, 3) int8 deltas (frame2 deltas) — only used if target='both'
    """
    n = f1_all.shape[0]
    pts_xy = np.zeros((n, K, 2), dtype=np.uint16)
    pts_d = np.zeros((n, K, 3), dtype=np.int8)
    pts_d2 = np.zeros((n, K, 3), dtype=np.int8)

    bs = 4
    for i in range(0, n, bs):
        e = min(i + bs, n)
        b = e - i
        f1 = f1_all[i:e].to(device).float().permute(0, 3, 1, 2)  # (b, 3, H, W)
        f2 = f2_all[i:e].to(device).float().permute(0, 3, 1, 2)
        gt_p = gt_poses[i:e].to(device).float()

        # Initial gradient through posenet to identify high-impact pixels
        f1_param = f1.clone().requires_grad_(True)
        f2_param = f2.clone().requires_grad_(True) if target in ('f2', 'both') else f2
        pin = diff_posenet_input(f1_param, f2_param)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        if target == 'f2':
            grad = torch.autograd.grad(loss, f2_param)[0]
        elif target == 'both':
            g1, g2 = torch.autograd.grad(loss, [f1_param, f2_param])
            grad = g1.abs() + g2.abs()  # combined magnitude for ranking
        else:
            grad = torch.autograd.grad(loss, f1_param)[0]
        gmag = grad.abs().sum(dim=1)  # (b, H, W)

        # Top-K pixel coordinates (vectorized)
        flat = gmag.view(b, -1)
        _, topk = torch.topk(flat, K, dim=1)  # (b, K)
        ys_t = (topk // OUT_W).long()  # (b, K)
        xs_t = (topk % OUT_W).long()
        pts_xy[i:e, :, 0] = xs_t.cpu().numpy().astype(np.uint16)
        pts_xy[i:e, :, 1] = ys_t.cpu().numpy().astype(np.uint16)

        # Initialize deltas at zero (let optimizer find them)
        cur_d1 = torch.zeros((b, K, 3), device=device, requires_grad=True)
        params_to_opt = [cur_d1]
        cur_d2 = None
        if target in ('f2', 'both'):
            cur_d2 = torch.zeros((b, K, 3), device=device, requires_grad=True)
            params_to_opt.append(cur_d2)
        opt = torch.optim.Adam(params_to_opt, lr=2.0)

        # Vectorized index for scatter-add
        batch_idx = torch.arange(b, device=device).view(-1, 1).expand(-1, K)  # (b, K)

        for it in range(n_iter):
            opt.zero_grad()
            # Vectorized patch application
            f1_p = f1.clone()
            for c in range(3):
                f1_p[batch_idx, c, ys_t, xs_t] = f1_p[batch_idx, c, ys_t, xs_t] + cur_d1[..., c]
            f1_p = f1_p.clamp(0, 255)
            if cur_d2 is not None:
                f2_p = f2.clone()
                for c in range(3):
                    f2_p[batch_idx, c, ys_t, xs_t] = f2_p[batch_idx, c, ys_t, xs_t] + cur_d2[..., c]
                f2_p = f2_p.clamp(0, 255)
            else:
                f2_p = f2
            pin = diff_posenet_input(f1_p, f2_p)
            fp = get_pose6(posenet, pin).float()
            loss = ((fp - gt_p) ** 2).sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                cur_d1.clamp_(-127, 127)
                if cur_d2 is not None: cur_d2.clamp_(-127, 127)

        pts_d[i:e] = cur_d1.detach().cpu().numpy().round().astype(np.int8)
        if cur_d2 is not None:
            pts_d2[i:e] = cur_d2.detach().cpu().numpy().round().astype(np.int8)
    return pts_xy, pts_d, pts_d2


def apply_patches_to_frames(f_all, pts_xy, pts_d, radius=0):
    """Apply patches with small radius around each point. Returns new uint8 frames."""
    n, K = pts_xy.shape[:2]
    out = f_all.clone()
    H, W = out.shape[1], out.shape[2]
    for i in range(n):
        arr = out[i].float().numpy()
        for j in range(K):
            x, y = int(pts_xy[i, j, 0]), int(pts_xy[i, j, 1])
            d = pts_d[i, j].astype(np.float32)
            if not d.any(): continue
            if radius == 0:
                arr[y, x] += d
            else:
                x1, x2 = max(0, x - radius), min(W, x + radius + 1)
                y1, y2 = max(0, y - radius), min(H, y + radius + 1)
                arr[y1:y2, x1:x2] += d[None, None, :]
        out[i] = torch.from_numpy(np.clip(arr, 0, 255).astype(np.uint8))
    return out


def sidecar_size(pts_xy, pts_d, pts_d2=None, radius=0):
    parts = [struct.pack("<III", pts_xy.shape[0], pts_xy.shape[1], radius),
             pts_xy.astype(np.uint16).tobytes(),
             pts_d.astype(np.int8).tobytes()]
    if pts_d2 is not None and pts_d2.any():
        parts.append(pts_d2.astype(np.int8).tobytes())
    return len(bz2.compress(b''.join(parts), compresslevel=9))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print(f"[sidecar] Loading model from {MODEL_PATH}")
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)
    print(f"[sidecar] Loading data...")
    data = load_data_full(device)
    n_pairs = data["val_rgb"].shape[0]
    print(f"[sidecar] {n_pairs} eval pairs")

    # ── Baseline: no sidecars ──
    print("\n[baseline] Generating frames...")
    f1_all, f2_all = generate_all_frames(gen, data, device)
    base = eval_with_frames(f1_all, f2_all, data, device, gen_for_bytes=gen)
    base_score = compose_score(base["seg_dist"], base["pose_dist"], base["model_bytes"], 0)
    print(f"[baseline] score={base_score['score']:.4f} seg={base_score['seg_term']:.4f} "
          f"pose={base_score['pose_term']:.4f} rate={base_score['rate_term']:.4f} "
          f"model_bytes={base['model_bytes']}")

    # CSV header
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["method", "K", "radius", "sidecar_bytes", "score", "seg_term", "pose_term",
                    "rate_term", "model_bytes", "delta_score"])
        w.writerow(["baseline", 0, 0, 0, base_score["score"], base_score["seg_term"],
                    base_score["pose_term"], base_score["rate_term"], base["model_bytes"], 0])

    def write_row(method, K, radius, sidecar_bytes, result):
        delta = result["score"] - base_score["score"]
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([method, K, radius, sidecar_bytes, result["score"], result["seg_term"],
                        result["pose_term"], result["rate_term"], result["model_bytes"], delta])

    posenet = load_posenet(device)

    # ── Strategy sweep (round 3) ──
    strategies = [
        # Best from round 2 with more iterations
        ("pose_f1_n50",    "f1",  7, 0, 50),
        ("pose_f1_n100",   "f1",  7, 0, 100),
        ("pose_f1_K5_n100","f1",  5, 0, 100),
    ]
    # Special strategies (handled separately below)
    seg_strategies = [
        ("seg_f2_K10", 10, 30),
        ("seg_f2_K20", 20, 30),
    ]
    universal_strategies = [
        ("universal_K20",  20, 50),
        ("universal_K50",  50, 50),
        ("universal_K100", 100, 50),
    ]

    for label, target, K, radius, n_iter in strategies:
        print(f"\n[{label}] K={K} target={target} radius={radius}")
        t0 = time.time()
        pts_xy, pts_d, pts_d2 = find_pose_patches(
            f1_all, f2_all, data["val_poses"], posenet, K=K, n_iter=n_iter, lr=1.0, device=device, target=target)
        print(f"  patch search: {time.time()-t0:.1f}s", flush=True)
        t0 = time.time()
        if target == 'f1':
            f1_p = apply_patches_to_frames(f1_all, pts_xy, pts_d, radius=radius)
            f2_p = f2_all
            sb = sidecar_size(pts_xy, pts_d, None, radius)
        elif target == 'f2':
            f1_p = f1_all
            f2_p = apply_patches_to_frames(f2_all, pts_xy, pts_d, radius=radius)
            sb = sidecar_size(pts_xy, pts_d, None, radius)
        else:
            f1_p = apply_patches_to_frames(f1_all, pts_xy, pts_d, radius=radius)
            f2_p = apply_patches_to_frames(f2_all, pts_xy, pts_d2, radius=radius)
            sb = sidecar_size(pts_xy, pts_d, pts_d2, radius)
        result = eval_with_frames(f1_p, f2_p, data, device, gen_for_bytes=gen)
        full = compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        print(f"  apply+eval: {time.time()-t0:.1f}s")
        print(f"  K={K} target={target}: sidecar={sb}B score={full['score']:.4f} "
              f"seg={full['seg_term']:.4f} pose={full['pose_term']:.4f} rate={full['rate_term']:.4f} "
              f"delta={full['score']-base_score['score']:+.4f}", flush=True)
        write_row(f"{label}_K{K}", K, radius, sb, full)

    # ── Seg patches on f2 (different mechanism) ──
    segnet = load_segnet(device)
    for label, K, n_iter in seg_strategies:
        print(f"\n[{label}]")
        t0 = time.time()
        pts_xy, pts_d = find_seg_patches_on_f2(f2_all, data["val_masks"], segnet, K=K, n_iter=n_iter, device=device)
        print(f"  patch search: {time.time()-t0:.1f}s", flush=True)
        f2_p = apply_patches_to_frames(f2_all, pts_xy, pts_d, radius=0)
        sb = sidecar_size(pts_xy, pts_d, None, 0)
        result = eval_with_frames(f1_all, f2_p, data, device, gen_for_bytes=gen)
        full = compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        print(f"  K={K}: sidecar={sb}B score={full['score']:.4f} seg={full['seg_term']:.4f} "
              f"pose={full['pose_term']:.4f} delta={full['score']-base_score['score']:+.4f}", flush=True)
        write_row(label, K, 0, sb, full)
    del segnet; gpu_cleanup()

    # ── Universal patch (single shared patch for all pairs) ──
    posenet2 = load_posenet(device)
    for label, K, n_iter in universal_strategies:
        print(f"\n[{label}]")
        t0 = time.time()
        pts_xy, pts_d = find_universal_patch(f1_all, f2_all, data["val_poses"], posenet2, K=K, n_iter=n_iter, device=device)
        print(f"  patch search: {time.time()-t0:.1f}s", flush=True)
        f1_p = apply_universal_patch_to_frames(f1_all, pts_xy, pts_d)
        sb = sidecar_size(pts_xy, pts_d, None, 0)
        result = eval_with_frames(f1_p, f2_all, data, device, gen_for_bytes=gen)
        full = compose_score(result["seg_dist"], result["pose_dist"], result["model_bytes"], sb)
        print(f"  K={K}: sidecar={sb}B score={full['score']:.4f} pose={full['pose_term']:.4f} "
              f"delta={full['score']-base_score['score']:+.4f}", flush=True)
        write_row(label, K, 0, sb, full)
    del posenet2; gpu_cleanup()

    print(f"\n[done] Results in {csv_path}")


if __name__ == "__main__":
    main()
