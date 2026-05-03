#!/usr/bin/env python
"""
On-policy distillation: train MobileUNet on adversarial decode trajectory data.

Instead of training the student on natural video frames (where it gets 99.75%
but fails at adversarial decode), we train on the ACTUAL intermediate frames
that adversarial decode generates: flat colors -> noisy blobs -> converged frames.

Phase 0: Generate trajectory data (run adversarial decode with teacher, record checkpoints)
Phase 1: Train MobileUNet on trajectory + original frames
Phase 2: Validate adversarial decode transfer

Usage:
  python train_onpolicy.py --phase 0        # Generate trajectory data (~5 min)
  python train_onpolicy.py --phase 1        # Train student (~20 min)
  python train_onpolicy.py --phase 2        # Validate transfer (~5 min)
  python train_onpolicy.py --phase all      # Do everything
"""
import sys, time, math, argparse, os
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file

from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import (MobileUNet, PoseLUT, rgb_to_yuv6_diff,
                           posenet_preprocess_diff, compute_boundary_weights,
                           IDEAL_COLORS, margin_loss)

DISTILL_DIR = Path('distill_data')
TRAJ_DIR = Path('distill_data/trajectory')
MODEL_DIR = Path('tiny_models')


# ─── Phase 0: Generate trajectory data ──────────────────────────────────────

def phase_generate_trajectory(device, num_iters=150, batch_size=8):
    """Run adversarial decode with REAL teachers, record intermediate frames.

    At checkpoints [0, 10, 25, 50, 75, 100, 149], we snapshot the current
    frame and get the teacher's segnet output. This creates training data
    that spans the ENTIRE optimization trajectory.
    """
    print("=" * 70)
    print("PHASE 0: GENERATE TRAJECTORY DATA")
    print("=" * 70)
    print()
    print("Running adversarial decode with real teachers and recording")
    print("intermediate frames. The student will train on these exact inputs.")
    print()

    # Load teachers
    print("[1/3] Loading teachers...")
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in segnet.parameters(): p.requires_grad_(False)
    for p in posenet.parameters(): p.requires_grad_(False)
    print("  Loaded SegNet + PoseNet")

    # Load targets
    print("[2/3] Loading targets...")
    seg_logits_all = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits_all.argmax(1)
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True)
    N = teacher_argmax.shape[0]
    print(f"  {N} frame pairs")

    colors = IDEAL_COLORS.to(device)

    # Checkpoints to record (iteration numbers)
    checkpoints = [0, 5, 15, 30, 60, 100, num_iters - 1]
    print(f"  Recording at iterations: {checkpoints}")
    print()

    TRAJ_DIR.mkdir(parents=True, exist_ok=True)

    # Collect trajectory data
    all_traj_frames = []  # list of (checkpoint_iter, frame_tensor, teacher_logits)
    all_traj_logits = []

    print(f"[3/3] Running adversarial decode ({N} pairs, {num_iters} iters)...")
    t0 = time.time()

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        B = batch_end - batch_start
        tgt_s = teacher_argmax[batch_start:batch_end].to(device)
        tgt_p = pose_outputs[batch_start:batch_end].to(device)

        # Initialize (same as real inflate.py)
        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1), keepdim=True).expand_as(init).clone()
        f0 = f0.requires_grad_(True)

        optimizer = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_sched = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(num_iters-1, 1)))
                    for i in range(num_iters)]

        batch_frames = []
        batch_logits = []

        for it in range(num_iters):
            # Record checkpoint
            if it in checkpoints:
                with torch.no_grad():
                    snapshot = f1.data.clone()
                    teacher_out = segnet(snapshot)
                    batch_frames.append(snapshot.cpu().half())
                    batch_logits.append(teacher_out.cpu().half())

            for pg in optimizer.param_groups: pg['lr'] = lr_sched[it]
            optimizer.zero_grad(set_to_none=True)

            progress = it / max(num_iters - 1, 1)
            seg_l = margin_loss(segnet(f1), tgt_s, 0.1 if progress < 0.5 else 5.0)

            if progress >= 0.3:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(posenet(pn_in)['pose'][:, :6], tgt_p)
                a = 120.0; b = 0.2 if progress < 0.9 else 1.6
                total = a * seg_l + b * pose_l
            else:
                total = 120.0 * seg_l

            total.backward()
            optimizer.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)

        # Record final frame too (the last checkpoint is num_iters-1)
        all_traj_frames.extend(batch_frames)
        all_traj_logits.extend(batch_logits)

        del f0, f1, optimizer
        torch.cuda.empty_cache()

        elapsed = time.time() - t0
        pairs_done = batch_end
        eta = elapsed / pairs_done * (N - pairs_done)
        n_recorded = len(all_traj_frames)
        print(f"  [{pairs_done:3d}/{N}] {n_recorded} trajectory frames recorded "
              f"({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Stack and save
    # all_traj_frames: list of (B, 3, 384, 512) half tensors
    # Reshape: (N_pairs * n_checkpoints, 3, 384, 512)
    n_checkpoints = len(checkpoints)
    traj_frames = torch.cat(all_traj_frames, dim=0)  # (N*n_cp, 3, H, W)
    traj_logits = torch.cat(all_traj_logits, dim=0)  # (N*n_cp, 5, H, W)

    print(f"\n  Total trajectory frames: {traj_frames.shape[0]}")
    print(f"  Shape: frames={tuple(traj_frames.shape)}, logits={tuple(traj_logits.shape)}")

    torch.save(traj_frames, TRAJ_DIR / 'traj_frames.pt')
    torch.save(traj_logits, TRAJ_DIR / 'traj_logits.pt')

    size_mb = (traj_frames.element_size() * traj_frames.numel() +
               traj_logits.element_size() * traj_logits.numel()) / 1024**2
    print(f"  Saved to {TRAJ_DIR}/ ({size_mb:.0f} MB)")

    del segnet, posenet
    torch.cuda.empty_cache()


# ─── Phase 1: Train on trajectory data ──────────────────────────────────────

def phase_train(device, epochs=150, lr=3e-3, seg_base_ch=48, batch_size=2):
    """Train MobileUNet on trajectory data + original frames.

    The key difference from train_distill.py:
    - 70% of training is on TRAJECTORY frames (flat colors, noisy blobs, etc.)
    - 30% is on original video frames
    - This teaches the student what the teacher does on adversarial decode inputs
    """
    print("=" * 70)
    print(f"PHASE 1: ON-POLICY TRAINING (epochs={epochs})")
    print("=" * 70)
    print()
    print("Training MobileUNet on adversarial decode trajectory data.")
    print("70% trajectory frames (what it'll actually see during inflate)")
    print("30% original video frames (for stable baseline)")
    print()

    # Load original data
    print("[1/4] Loading original frames + teacher outputs...")
    seg_inputs = torch.load(DISTILL_DIR / 'seg_inputs.pt', weights_only=True)
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1)
    base_map = torch.from_numpy(np.load(DISTILL_DIR / 'base_map.npy'))
    N_orig = seg_inputs.shape[0]
    print(f"  {N_orig} original frames")

    # Load trajectory data
    print("[2/4] Loading trajectory data...")
    traj_frames = torch.load(TRAJ_DIR / 'traj_frames.pt', weights_only=True)  # half
    traj_logits = torch.load(TRAJ_DIR / 'traj_logits.pt', weights_only=True)  # half
    N_traj = traj_frames.shape[0]
    traj_argmax = traj_logits.float().argmax(1)
    print(f"  {N_traj} trajectory frames")
    print()

    # Student (no teacher needed — we have pre-extracted logits for everything)
    print("[3/3] Creating student MobileUNet...")
    student = MobileUNet(in_ch=3, n_classes=5, base_ch=seg_base_ch).to(device)
    student.init_base_map(base_map.to(device))
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  {n_params:,} params ({n_params*2/1024:.1f} KB at FP16)")
    print()

    # Training setup
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    T_kd = 6.0
    best_acc, best_traj_acc, best_state = 0.0, 0.0, None
    t0 = time.time()

    print(f"  {'Ep':>3} {'Loss':>7} {'OrigAcc':>8} {'TrajAcc':>8} {'BestOrig':>8} {'LR':>9} {'Time':>5}")
    print("  " + "-" * 60)

    for epoch in range(epochs):
        student.train()

        # ── Trajectory batch (70% of training, subsample to keep epochs fast) ──
        traj_perm = torch.randperm(N_traj)
        traj_loss_sum = 0.0
        traj_batches = 0
        traj_per_epoch = min(N_traj, 600)  # same size as original data

        for i in range(0, traj_per_epoch, batch_size):
            idx = traj_perm[i:i+batch_size]
            x = traj_frames[idx].float().to(device)
            tl = traj_logits[idx].float().to(device)
            ta = traj_argmax[idx].to(device)

            out = student(x)
            kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                          F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
            ce = F.cross_entropy(out, ta)
            mse = F.mse_loss(out, tl)
            loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            opt.zero_grad()
            loss.backward()
            opt.step()
            traj_loss_sum += loss.item()
            traj_batches += 1

        # ── Original frames batch (30% of training) ──
        orig_perm = torch.randperm(N_orig)
        for i in range(0, N_orig, batch_size):
            idx = orig_perm[i:i+batch_size]
            x = seg_inputs[idx].to(device)
            tl = seg_logits[idx].to(device)
            ta = teacher_argmax[idx].to(device)

            out = student(x)
            kd = F.kl_div(F.log_softmax(out / T_kd, 1),
                          F.softmax(tl / T_kd, 1), reduction='batchmean') * (T_kd**2)
            ce = F.cross_entropy(out, ta)
            mse = F.mse_loss(out, tl)
            loss = 0.4 * kd + 0.3 * ce + 0.3 * mse

            opt.zero_grad()
            loss.backward()
            opt.step()

        sched.step()

        # ── Evaluate ──
        student.eval()
        # Accuracy on original frames
        orig_acc = _check_acc(student, seg_inputs, teacher_argmax, device)
        # Accuracy on trajectory frames (sample)
        sample_n = min(600, N_traj)
        traj_acc = _check_acc(student, traj_frames[:sample_n].float(),
                              traj_argmax[:sample_n], device)

        if orig_acc > best_acc:
            best_acc = orig_acc
            best_traj_acc = traj_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            # Save immediately on new best
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, MODEL_DIR / 'onpolicy_segnet.pt')

        elapsed = time.time() - t0
        avg_loss = traj_loss_sum / max(traj_batches, 1)
        saved = "*" if orig_acc >= best_acc else " "
        print(f" {saved}{epoch:3d} {avg_loss:7.3f} {orig_acc*100:7.3f}% {traj_acc*100:7.3f}% "
              f"{best_acc*100:7.3f}% {sched.get_last_lr()[0]:.2e} {elapsed:4.0f}s",
              flush=True)

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if best_state:
        student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save({k: v.cpu() for k, v in student.state_dict().items()},
               MODEL_DIR / 'onpolicy_segnet.pt')
    size_kb = os.path.getsize(MODEL_DIR / 'onpolicy_segnet.pt') / 1024
    print(f"\n  Best: orig_acc={best_acc*100:.3f}% traj_acc={best_traj_acc*100:.3f}%")
    print(f"  Saved to {MODEL_DIR / 'onpolicy_segnet.pt'} ({size_kb:.1f} KB)")
    return student, best_acc


def _check_acc(model, inputs, targets, device, batch_size=32):
    correct = total = 0
    N = inputs.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = inputs[i:i+batch_size].to(device)
            ta = targets[i:i+batch_size].to(device)
            pred = model(x).argmax(1)
            correct += (pred == ta).sum().item()
            total += ta.numel()
    return correct / total


# ─── Phase 2: Validate transfer ─────────────────────────────────────────────

def phase_validate(device, seg_base_ch=48, num_batches=7, iters=200):
    """THE test. Run adversarial decode with student, evaluate with teacher."""
    print("=" * 70)
    print("PHASE 2: ADVERSARIAL DECODE TRANSFER TEST")
    print("=" * 70)
    print()

    # Load teachers (evaluation only)
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    # Load student seg
    print("Loading on-policy MobileUNet...")
    base_map = torch.from_numpy(np.load(DISTILL_DIR / 'base_map.npy'))
    s_seg = MobileUNet(in_ch=3, n_classes=5, base_ch=seg_base_ch).to(device)
    s_seg.init_base_map(base_map.to(device))
    sd_path = MODEL_DIR / 'onpolicy_segnet.pt'
    s_seg.load_state_dict(torch.load(sd_path, weights_only=True, map_location=device))
    s_seg.eval()
    for p in s_seg.parameters(): p.requires_grad_(False)
    seg_params = sum(p.numel() for p in s_seg.parameters())
    print(f"  MobileUNet: {seg_params:,} params")

    # Load PoseLUT
    print("Loading PoseLUT...")
    pose_outputs = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True)
    s_pose = PoseLUT(n_frames=pose_outputs.shape[0], pose_dim=6, embed_dim=32,
                     base_ch=16).to(device)
    s_pose.load_state_dict(torch.load(MODEL_DIR / 'pose_lut.pt', weights_only=True,
                                       map_location=device))
    s_pose.eval()
    for p in s_pose.parameters(): p.requires_grad_(False)
    pose_params = sum(p.numel() for p in s_pose.parameters())
    print(f"  PoseLUT: {pose_params:,} params")
    print()

    # Load targets
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)
    pose_targets = pose_outputs.to(device)
    del seg_logits

    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size
    bs = 4
    starts = [0, 50, 100, 200, 300, 400, 500][:num_batches]

    print(f"Running adversarial decode with STUDENT models...")
    print(f"  {num_batches} batches x {bs} pairs x {iters} iters")
    print()

    results = {'ss': [], 'ts': [], 'tp': []}

    for bi, st in enumerate(starts):
        end = min(st + bs, len(teacher_argmax))
        tgt_s = teacher_argmax[st:end]
        tgt_p = pose_targets[st:end]
        B = tgt_s.shape[0]

        init = colors[tgt_s].permute(0, 3, 1, 2).clone()
        f1 = init.requires_grad_(True)
        f0 = init.detach().mean(dim=(-2,-1), keepdim=True).expand_as(init).clone().requires_grad_(True)

        optimizer = torch.optim.AdamW([f0, f1], lr=1.2, weight_decay=0)
        lr_sched = [0.06 + 0.57 * (1 + math.cos(math.pi * i / max(iters-1,1)))
                    for i in range(iters)]

        print(f"  Batch {bi+1}/{num_batches} (pairs {st}-{end-1}): ", end='', flush=True)

        for it in range(iters):
            for pg in optimizer.param_groups: pg['lr'] = lr_sched[it]
            optimizer.zero_grad(set_to_none=True)

            progress = it / max(iters-1, 1)

            # Student seg
            seg_l = margin_loss(s_seg(f1), tgt_s, 5.0)

            # Student pose
            run_pose = (progress >= 0.3)
            if run_pose:
                both = torch.stack([f0, f1], dim=1)
                pn_in = posenet_preprocess_diff(both)
                pose_l = F.smooth_l1_loss(s_pose(pn_in)['pose'], tgt_p)
            else:
                pose_l = seg_l.new_zeros(())

            a = 120.0 if progress < 0.9 else 24.0
            b = 0.06 if progress < 0.3 else (0.3 if progress < 0.9 else 1.6)
            total = a * seg_l + b * pose_l

            total.backward()
            optimizer.step()
            with torch.no_grad():
                f0.data.clamp_(0, 255)
                f1.data.clamp_(0, 255)

            if (it+1) % 50 == 0:
                print(f"{it+1}", end=' ', flush=True)

        print("done", flush=True)

        # Evaluate with real teacher
        with torch.no_grad():
            # Student self-assessment
            ss = (s_seg(f1.data).argmax(1) != tgt_s).float().mean((1,2))
            results['ss'].extend(ss.cpu().tolist())

            # Teacher assessment (with resolution round-trip)
            f1_up = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0,255).round().byte().float()
            f0_up = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                  align_corners=False).clamp(0,255).round().byte().float()

            ts_in = F.interpolate(f1_up, (mH, mW), mode='bilinear')
            ts = (t_seg(ts_in).argmax(1) != tgt_s).float().mean((1,2))
            results['ts'].extend(ts.cpu().tolist())

            tp_pair = F.interpolate(
                torch.stack([f0_up, f1_up], 1).reshape(-1, 3, Hc, Wc),
                (mH, mW), mode='bilinear'
            ).reshape(B, 2, 3, mH, mW)
            tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:, :6]
            tp = (tpo - tgt_p).pow(2).mean(1)
            results['tp'].extend(tp.cpu().tolist())

        print(f"    Student seg={np.mean(results['ss'][-B:]):.6f} | "
              f"Teacher seg={np.mean(results['ts'][-B:]):.6f} pose={np.mean(results['tp'][-B:]):.6f}")

        del f0, f1, optimizer
        torch.cuda.empty_cache()

    # Summary
    avg_ss = np.mean(results['ss'])
    avg_ts = np.mean(results['ts'])
    avg_tp = np.mean(results['tp'])
    gap = avg_ts - avg_ss

    score_seg = 100 * avg_ts
    score_pose = math.sqrt(10 * avg_tp) if avg_tp > 0 else 0
    distortion = score_seg + score_pose

    print()
    print("=" * 70)
    print("TRANSFER RESULTS")
    print("=" * 70)
    print(f"  Student seg_dist: {avg_ss:.6f}")
    print(f"  Teacher seg_dist: {avg_ts:.6f}  (gap: {gap:.6f})")
    print(f"  Teacher pose_mse: {avg_tp:.6f}")
    print()
    print(f"  100*seg    = {score_seg:.4f}")
    print(f"  sqrt(10*p) = {score_pose:.4f}")
    print(f"  distortion = {distortion:.4f}")
    print()

    # Estimate archive: MobileUNet FP16 + PoseLUT + targets
    seg_size = seg_params * 2  # FP16
    pose_size = 216 * 1024  # PoseLUT
    target_size = 300 * 1024  # estimate
    archive = seg_size + pose_size + target_size
    rate = archive / 37_545_489
    total = distortion + 25 * rate
    print(f"  Archive: seg={seg_size/1024:.0f}KB + pose={pose_size/1024:.0f}KB + tgt={target_size/1024:.0f}KB = {archive/1024:.0f}KB")
    print(f"  25*rate = {25*rate:.3f}")
    print(f"  TOTAL SCORE = {total:.4f}")
    print(f"  Leader: 1.95 | Baseline: 4.39")
    print()

    if gap < 0.005:
        print("  Transfer gap is SMALL -- student is a good proxy!")
    elif gap < 0.02:
        print("  Transfer gap is MODERATE -- might work, could improve with more training")
    else:
        print("  Transfer gap is LARGE -- student gradients are misleading the optimization")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='all', choices=['0', '1', '2', 'all'])
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--seg-ch', type=int, default=48)
    parser.add_argument('--lr', type=float, default=3e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    if args.phase in ('0', 'all'):
        phase_generate_trajectory(device)
        print()

    if args.phase in ('1', 'all'):
        phase_train(device, epochs=args.epochs, seg_base_ch=args.seg_ch, lr=args.lr)
        print()

    if args.phase in ('2', 'all'):
        phase_validate(device, seg_base_ch=args.seg_ch)


if __name__ == '__main__':
    main()
