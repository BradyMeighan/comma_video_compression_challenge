"""Shared utilities for v2 experiments — batched evaluators, cached state.

Speedups vs original explore_*.py:
  - Batched candidate evaluation (10x for CMA-ES populations)
  - Cached DistortionNet (already in sidecar_stack.py)
  - Cached baseline patches/frames (loaded once per script)
  - Batched mask flip verification (parallel over candidates)
"""
import sys, os, pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import (OUT_H, OUT_W, MODEL_H, MODEL_W, get_pose6,
                      load_posenet, estimate_model_bytes, apply_fp4_to_model,
                      MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE)
from train import Generator, load_data_full, coords
import sidecar_explore as se

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
MODEL_PATH = os.environ.get("MODEL_PATH", "autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt")


# ─── Common state loader ───────────────────────────────────────────────

class State:
    """Bundles everything experiments need. Loaded once, reused."""
    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        torch.backends.cudnn.benchmark = True
        print("Loading model + data...", flush=True)
        self.gen = Generator().to(self.device)
        sd = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.gen.load_state_dict(sd, strict=False)
        # Apply FP4 quantization (matches prepare.evaluate behavior).
        # Required when loading QAT-trained EMA weights — without this the
        # gen produces wildly wrong outputs because activations are scaled
        # for FP4 inference but weights are full-precision.
        self.gen.eval()
        apply_fp4_to_model(self.gen)
        self.data = load_data_full(self.device)
        self.posenet = load_posenet(self.device)
        self.model_bytes = estimate_model_bytes(self.gen)
        self.poses = self.data["val_poses"]
        self.masks_cpu = self.data["val_masks"].cpu()

        # Load baseline patches + frames
        bp_path = OUTPUT_DIR / "baseline_patches.pkl"
        with open(bp_path, 'rb') as f:
            self.bp = pickle.load(f)
        bf = torch.load(OUTPUT_DIR / "baseline_frames.pt", weights_only=False)
        self.f1_baseline = bf['f1_new']
        self.f2_baseline = bf['f2_new']
        self.baseline_masks = bf['new_masks']
        self.score_baseline = self.bp['score']
        self.sb_mask_baseline = self.bp['sb_mask_bz2']
        self.sb_rgb_baseline = self.bp['sb_rgb_bz2']
        self.sb_total_baseline = self.bp['sb_total_bz2']

        self.pose_per_pair = np.load(OUTPUT_DIR / "pose_per_pair.npy")
        self.rank = np.argsort(self.pose_per_pair)[::-1]
        print(f"State ready. model={MODEL_PATH}", flush=True)
        print(
            f"  legacy baseline_patches score={self.score_baseline:.4f} "
            f"(not raw e80 / not current unified best)",
            flush=True,
        )


# ─── Batched gen forward with one-hot mask (differentiable wrt mask) ──

def gen_forward_with_oh_mask_batch(gen, m_oh, p, device):
    """Same as sidecar_mask_verified.gen_forward_with_oh_mask but accepts batch."""
    co = coords(m_oh.shape[0], MODEL_H, MODEL_W, device)
    e_emb = m_oh @ gen.trunk.emb.weight
    e = F.interpolate(e_emb.permute(0, 3, 1, 2), co.shape[-2:], mode='bilinear', align_corners=False)
    s = gen.trunk.s1(gen.trunk.stem(torch.cat([e, co], 1)))
    z = gen.trunk.up(gen.trunk.d1(gen.trunk.down(s)))
    feat = gen.trunk.f1(gen.trunk.fuse(torch.cat([z, s], 1)))
    cond = gen.pose_mlp(p)
    f1u = F.interpolate(gen.h1(feat, cond), (OUT_H, OUT_W), mode='bilinear', align_corners=False)
    f2u = F.interpolate(gen.h2(feat), (OUT_H, OUT_W), mode='bilinear', align_corners=False)
    return f1u, f2u


def batch_pose_loss_for_mask_candidates(gen, base_m_int, candidate_perturbations,
                                          base_pose, gt_pose, posenet, device):
    """Evaluate pose loss for MULTIPLE mask perturbation candidates at once.

    base_m_int: (1, MH, MW) long, the current mask state
    candidate_perturbations: list of [(x, y, new_class), ...] — each candidate is a list of perturbations
    base_pose: (1, 6)
    gt_pose: (1, 6)
    posenet: frozen PoseNet
    Returns: tensor (N_candidates,) of pose loss values

    Speedup: instead of N gen forwards, do 1 batched forward with batch dim N.
    """
    N = len(candidate_perturbations)
    # Build a (N, MH, MW, 5) one-hot mask batch
    m_oh = F.one_hot(base_m_int, num_classes=5).float()  # (1, MH, MW, 5)
    m_oh_batch = m_oh.expand(N, -1, -1, -1).contiguous().clone()  # (N, MH, MW, 5)
    for i, perturbs in enumerate(candidate_perturbations):
        for (x, y, c) in perturbs:
            m_oh_batch[i, y, x] = 0
            m_oh_batch[i, y, x, c] = 1
    p_batch = base_pose.expand(N, -1)
    gt_batch = gt_pose.expand(N, -1)

    with torch.no_grad():
        f1u, f2u = gen_forward_with_oh_mask_batch(gen, m_oh_batch, p_batch, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        losses = ((fp - gt_batch) ** 2).sum(dim=1)  # (N,)
    return losses


def batch_pose_loss_for_block_candidates(gen, base_m_int, candidate_blocks,
                                          base_pose, gt_pose, posenet, device, block=2):
    """Like above but each candidate sets a BxB block to a single class.

    candidate_blocks: list of (x, y, class) — each is one block flip
    """
    N = len(candidate_blocks)
    m_oh = F.one_hot(base_m_int, num_classes=5).float()
    m_oh_batch = m_oh.expand(N, -1, -1, -1).contiguous().clone()
    for i, (x, y, c) in enumerate(candidate_blocks):
        # zero out block, set class c
        m_oh_batch[i, y:y+block, x:x+block] = 0
        m_oh_batch[i, y:y+block, x:x+block, c] = 1
    p_batch = base_pose.expand(N, -1)
    gt_batch = gt_pose.expand(N, -1)
    with torch.no_grad():
        f1u, f2u = gen_forward_with_oh_mask_batch(gen, m_oh_batch, p_batch, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        losses = ((fp - gt_batch) ** 2).sum(dim=1)
    return losses


def batch_pose_loss_for_pattern_candidates(gen, base_m_int, candidate_patterns,
                                              base_pose, gt_pose, posenet, device):
    """Each candidate = list of (x, y, pattern_id, class) where pattern_id encodes shape.

    Pattern IDs:
      0: 1x1 (single pixel)
      1: 3x3
      2: 1x4 (horizontal strip)
      3: 4x1 (vertical strip)
      4: 2x2
    """
    PATTERNS = {
        0: (1, 1), 1: (3, 3), 2: (1, 4), 3: (4, 1), 4: (2, 2),
    }
    N = len(candidate_patterns)
    m_oh = F.one_hot(base_m_int, num_classes=5).float()
    m_oh_batch = m_oh.expand(N, -1, -1, -1).contiguous().clone()
    for i, configs in enumerate(candidate_patterns):
        for (x, y, p_id, c) in configs:
            ph, pw = PATTERNS[int(p_id)]
            yy_end = min(y + ph, MODEL_H); xx_end = min(x + pw, MODEL_W)
            m_oh_batch[i, y:yy_end, x:xx_end] = 0
            m_oh_batch[i, y:yy_end, x:xx_end, c] = 1
    p_batch = base_pose.expand(N, -1)
    gt_batch = gt_pose.expand(N, -1)
    with torch.no_grad():
        f1u, f2u = gen_forward_with_oh_mask_batch(gen, m_oh_batch, p_batch, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        losses = ((fp - gt_batch) ** 2).sum(dim=1)
    return losses


def compose_score(seg_dist, pose_dist, model_bytes, sidecar_bytes):
    """Score formula."""
    import math
    total = MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes
    rate = total / UNCOMPRESSED_SIZE
    return {
        "score": 100 * seg_dist + math.sqrt(max(0, 10 * pose_dist)) + 25 * rate,
        "seg_term": 100 * seg_dist,
        "pose_term": math.sqrt(max(0, 10 * pose_dist)),
        "rate_term": 25 * rate,
    }


# ─── Mask format helpers ──────────────────────────────────────────────

def serialize_block_mask_v2(block_patches):
    """5 bytes per block patch: u16 x, u16 y, u8 class. Returns bz2-compressed bytes."""
    import struct, bz2
    if not block_patches:
        return b''
    parts = [struct.pack("<H", len(block_patches))]
    for pi in sorted(block_patches.keys()):
        ps = block_patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for tup in ps:
            x, y, c = tup[0], tup[1], tup[2]
            parts.append(struct.pack("<HHB", x, y, c))
    return bz2.compress(b''.join(parts), compresslevel=9)


# ─── Batched optimizers (replace single-candidate-per-forward originals) ─

def verified_greedy_block_mask_batched(gen, m_init, p, gt_p, posenet, device,
                                          K=1, n_candidates=10, block=2):
    """Batched verified-greedy block mask: 5x speedup vs original.

    For each K iter:
      1. Compute gradient → score top-N candidate (x,y) positions
      2. Build all (n_candidates × 5_classes) (pos, class) tuples
      3. BATCH all in one gen forward
      4. Pick best
    """
    from sidecar_mask_verified import gen_forward_with_oh_mask
    from prepare import get_pose6
    cur_m = m_init.clone()
    accepted = []
    for _ in range(K):
        m_oh = F.one_hot(cur_m, num_classes=5).float().requires_grad_(True)
        f1u, f2u = gen_forward_with_oh_mask(gen, m_oh, p, device)
        pin = se.diff_posenet_input(f1u, f2u)
        fp = get_pose6(posenet, pin).float()
        loss = ((fp - gt_p) ** 2).sum()
        baseline_loss = loss.item()
        grad = torch.autograd.grad(loss, m_oh)[0]
        cur_class = cur_m
        grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
        candidate_delta = grad - grad_cur.unsqueeze(-1)
        for cls in range(5):
            candidate_delta[..., cls][cur_class == cls] = float('inf')
        best_delta, _ = candidate_delta.min(dim=-1)
        neg_delta = (-best_delta).clamp_min(0)
        if block > 1:
            pooled = F.avg_pool2d(neg_delta.unsqueeze(0), kernel_size=block, stride=1) * (block * block)
            pooled = pooled.squeeze(0)
        else:
            pooled = neg_delta
        for (x, y, _) in accepted:
            for dy in range(block):
                for dx in range(block):
                    yy = y + dy; xx = x + dx
                    if 0 <= yy < pooled.shape[1] and 0 <= xx < pooled.shape[2]:
                        pooled[0, yy, xx] = 0
        flat = pooled.contiguous().reshape(-1)
        n_top = min(n_candidates, flat.numel())
        topk_vals, topk_idx = torch.topk(flat, n_top)
        H_p = pooled.shape[1]; W_p = pooled.shape[2]
        cand_ys = (topk_idx // W_p).long().cpu().numpy()
        cand_xs = (topk_idx % W_p).long().cpu().numpy()
        candidates = []
        for k in range(n_top):
            for new_cls in range(5):
                candidates.append((int(cand_xs[k]), int(cand_ys[k]), new_cls))
        if not candidates:
            break
        losses = batch_pose_loss_for_block_candidates(
            gen, cur_m, candidates, p, gt_p, posenet, device, block=block)
        deltas = (losses - baseline_loss).cpu().numpy()
        best_i = deltas.argmin()
        if deltas[best_i] >= 0:
            break
        x, y, new_cls = candidates[best_i]
        cur_m[0, y:y+block, x:x+block] = new_cls
        accepted.append((x, y, new_cls))
    return accepted, cur_m


def cma_es_mask_for_pair_batched(gen, m_init, p, gt_p, posenet, device,
                                    K=2, pop=12, gens=15, n_candidates=30):
    """Batched CMA-ES K single-pixel flips: ~10x faster than serial.

    Each generation's pop candidates evaluated in 1 batched forward.
    """
    from sidecar_mask_verified import gen_forward_with_oh_mask, pose_loss_for_pair
    from prepare import get_pose6
    # Get top-N candidate positions from gradient
    m_oh = F.one_hot(m_init, num_classes=5).float().requires_grad_(True)
    f1u, f2u = gen_forward_with_oh_mask(gen, m_oh, p, device)
    pin = se.diff_posenet_input(f1u, f2u)
    fp = get_pose6(posenet, pin).float()
    loss = ((fp - gt_p) ** 2).sum()
    grad = torch.autograd.grad(loss, m_oh)[0]
    cur_class = m_init
    grad_cur = grad.gather(3, cur_class.unsqueeze(-1)).squeeze(-1)
    candidate_delta = grad - grad_cur.unsqueeze(-1)
    for cls in range(5):
        candidate_delta[..., cls][cur_class == cls] = float('inf')
    best_delta, best_class = candidate_delta.min(dim=-1)
    flat = best_delta.contiguous().reshape(-1)
    _, top_idx = torch.topk(-flat, n_candidates)
    cand_ys = (top_idx // MODEL_W).long().cpu().numpy()
    cand_xs = (top_idx % MODEL_W).long().cpu().numpy()
    cand_classes = best_class.contiguous().reshape(-1)[top_idx].cpu().numpy()

    with torch.no_grad():
        base_loss = pose_loss_for_pair(
            gen, F.one_hot(m_init, num_classes=5).float(), p, gt_p, posenet, device)

    mu = np.random.uniform(0, n_candidates, K)
    sigma = n_candidates * 0.3
    best_choice = None
    best_loss = base_loss

    for _ in range(gens):
        # Sample pop candidate index sets
        candidate_perturbations = []
        sample_idx_sets = []
        for _ in range(pop):
            x = mu + sigma * np.random.randn(K)
            x = np.clip(x.round().astype(int), 0, n_candidates - 1)
            x = np.unique(x)
            if len(x) == 0:
                continue
            flips = []
            for idx in x:
                flips.append((int(cand_xs[idx]), int(cand_ys[idx]), int(cand_classes[idx])))
            candidate_perturbations.append(flips)
            sample_idx_sets.append(x)
        if not candidate_perturbations:
            continue

        # BATCHED eval
        losses = batch_pose_loss_for_mask_candidates(
            gen, m_init, candidate_perturbations, p, gt_p, posenet, device).cpu().numpy()

        # Track best
        best_i_local = losses.argmin()
        if losses[best_i_local] < best_loss:
            best_loss = losses[best_i_local]
            best_choice = candidate_perturbations[best_i_local]

        # ES update: keep top half, recompute mu and sigma
        sorted_idx = np.argsort(losses)
        elite_n = max(1, len(losses) // 2)
        elite_idx_sets = [sample_idx_sets[i] for i in sorted_idx[:elite_n]]
        # Recompute mu by averaging the K-element index sets (pad shorter)
        mu_acc = np.zeros(K); cnt = 0
        for s in elite_idx_sets:
            for j in range(min(K, len(s))):
                mu_acc[j] += s[j]
                cnt += 1
        if cnt > 0:
            mu = mu_acc / max(1, len(elite_idx_sets))
        sigma *= 0.85
    return best_choice if best_choice else []


def serialize_pattern_mask(pattern_patches):
    """6 bytes per: u16 x, u16 y, u8 pattern_id, u8 class."""
    import struct, bz2
    if not pattern_patches:
        return b''
    parts = [struct.pack("<H", len(pattern_patches))]
    for pi in sorted(pattern_patches.keys()):
        ps = pattern_patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, p_id, c) in ps:
            parts.append(struct.pack("<HHBB", x, y, p_id, c))
    return bz2.compress(b''.join(parts), compresslevel=9)
