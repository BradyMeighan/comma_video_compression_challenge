#!/usr/bin/env python
"""
HYBRID: Codec initialization + adversarial refinement.

Instead of starting adversarial decode from flat-colored blobs (maximally OOD),
start from codec-decoded frames (near-natural). The MobileUNet only needs to
provide correct gradients on near-natural inputs — which it achieves at 99.75%.

Test: decode compressed video → refine with MobileUNet for 20-50 iters → evaluate.
"""
import sys, time, math, subprocess
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from frame_utils import camera_size, segnet_model_input_size
from train_distill import (MobileUNet, PoseLUT, posenet_preprocess_diff,
                           margin_loss, IDEAL_COLORS)

DISTILL_DIR = Path('distill_data')


def decode_video_frames(video_path, device):
    """Decode video to raw frames using ffmpeg, return as tensor at model resolution."""
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]  # 384, 512
    Wc, Hc = camera_size  # 1164, 874

    # Get video dimensions via ffprobe
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', str(video_path)],
        capture_output=True, text=True)
    parts = probe.stdout.strip().split(',')
    if len(parts) < 2:
        print(f"  ERROR: Could not probe {video_path}")
        return None
    vid_w, vid_h = int(parts[0]), int(parts[1])

    # Decode to raw RGB at native resolution
    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-v', 'quiet', '-'
    ]
    result = subprocess.run(cmd, capture_output=True)
    raw = result.stdout

    frame_bytes = vid_w * vid_h * 3
    n_frames = len(raw) // frame_bytes
    if n_frames == 0:
        print(f"  ERROR: No frames decoded from {video_path}")
        return None

    frames = np.frombuffer(raw, dtype=np.uint8)[:n_frames * frame_bytes].reshape(n_frames, vid_h, vid_w, 3)
    print(f"  Decoded {n_frames} frames at {vid_w}x{vid_h}")
    frames_t = torch.from_numpy(frames.copy()).float().permute(0, 3, 1, 2)
    frames_resized = F.interpolate(frames_t, (mH, mW), mode='bilinear', align_corners=False)
    return frames_resized.to(device)


def main():
    device = torch.device('cuda')
    print(f"Device: {device}\n")

    # Load teachers for evaluation
    print("Loading teachers...")
    t_seg = SegNet().eval().to(device)
    t_seg.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    t_pose = PoseNet().eval().to(device)
    t_pose.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))
    for p in t_seg.parameters(): p.requires_grad_(False)
    for p in t_pose.parameters(): p.requires_grad_(False)

    # Load targets
    seg_logits = torch.load(DISTILL_DIR / 'seg_logits.pt', weights_only=True)
    teacher_argmax = seg_logits.argmax(1).to(device)
    pose_targets = torch.load(DISTILL_DIR / 'pose_outputs.pt', weights_only=True).to(device)
    base_map = torch.from_numpy(np.load(DISTILL_DIR / 'base_map.npy'))
    del seg_logits

    # Load MobileUNet student
    print("Loading MobileUNet student...")
    student = MobileUNet(in_ch=3, n_classes=5, base_ch=48).to(device)
    student.init_base_map(base_map.to(device))
    sd_path = Path('tiny_models/mobile_segnet.pt')
    if sd_path.exists():
        student.load_state_dict(torch.load(sd_path, weights_only=True, map_location=device))
        print(f"  Loaded from {sd_path}")
    else:
        # Try on-policy version
        sd_path = Path('tiny_models/onpolicy_segnet.pt')
        student.load_state_dict(torch.load(sd_path, weights_only=True, map_location=device))
        print(f"  Loaded from {sd_path}")
    student.eval()
    for p in student.parameters(): p.requires_grad_(False)

    # Also test with the SVD non-factored model (proven 1.31 distortion)
    print("Loading SVD SegNet...")
    svd_seg = SegNet().eval().to(device)
    svd_sd = torch.load('compressed_models/segnet_svd_finetuned.pt',
                         weights_only=True, map_location=device)
    svd_seg.load_state_dict(svd_sd)
    for p in svd_seg.parameters(): p.requires_grad_(False)

    colors = IDEAL_COLORS.to(device)
    mH, mW = segnet_model_input_size[1], segnet_model_input_size[0]
    Wc, Hc = camera_size

    # Try available compressed videos
    videos = [
        ('_temporal_roi_tmp/0.mkv', '256KB temporal ROI'),
        ('submissions/phase1/archive/0.mkv', '584KB phase1'),
        ('_val_tmp/0.mkv', '848KB val'),
    ]

    for video_path, label in videos:
        if not Path(video_path).exists():
            continue

        size_kb = Path(video_path).stat().st_size / 1024
        print(f"\n{'='*70}")
        print(f"VIDEO: {label} ({size_kb:.0f} KB)")
        print(f"{'='*70}")

        # Decode video
        print("  Decoding video...")
        decoded = decode_video_frames(video_path, device)
        if decoded is None:
            continue

        n_decoded = decoded.shape[0]
        print(f"  Got {n_decoded} decoded frames at {mW}x{mH}")

        # Check: how good is the codec alone? (no refinement)
        print("\n  Evaluating codec alone (no refinement)...")
        codec_seg_dists = []
        with torch.no_grad():
            for i in range(0, min(n_decoded, 600), 32):
                frames_batch = decoded[i:i+32]
                # Need to handle frame pair structure: SegNet uses last frame of pair
                # For simplicity, evaluate each frame against its target
                n = min(frames_batch.shape[0], 600 - i)
                pred = t_seg(frames_batch[:n]).argmax(1)
                tgt = teacher_argmax[i:i+n]
                dist = (pred != tgt).float().mean((1, 2))
                codec_seg_dists.extend(dist.cpu().tolist())
        codec_seg = np.mean(codec_seg_dists)
        print(f"  Codec-only seg_dist: {codec_seg:.6f} ({(1-codec_seg)*100:.2f}% agreement)")
        print(f"  Codec-only 100*seg: {100*codec_seg:.4f}")

        # Now test hybrid: codec init + refinement
        for model_name, model in [("MobileUNet (1MB)", student),
                                   ("SVD SegNet (5.5MB)", svd_seg)]:
            for n_iters in [10, 30]:
                print(f"\n  --- Hybrid: {model_name} x {n_iters} iters ---")

                results = {'ts': [], 'tp': []}
                for st in [0, 100, 200, 300, 450]:
                    end = min(st + 4, 600, n_decoded)
                    if end <= st: break
                    tgt_s = teacher_argmax[st:end]
                    tgt_p = pose_targets[st:end]
                    B = tgt_s.shape[0]

                    # Initialize from DECODED VIDEO frames (not flat blobs!)
                    f1 = decoded[st:end].clone().requires_grad_(True)
                    f0 = decoded[max(0,st-1):max(0,st-1)+B].clone() if st > 0 else \
                         decoded[st:end].clone()
                    f0 = f0.mean(dim=(-2,-1), keepdim=True).expand_as(f1).clone().requires_grad_(True)

                    opt = torch.optim.AdamW([f0, f1], lr=0.5, weight_decay=0)  # lower LR for refinement
                    lr_s = [0.05 + 0.25*(1+math.cos(math.pi*i/max(n_iters-1,1)))
                            for i in range(n_iters)]

                    for it in range(n_iters):
                        for pg in opt.param_groups: pg['lr'] = lr_s[it]
                        opt.zero_grad(set_to_none=True)
                        p = it / max(n_iters-1, 1)

                        seg_l = margin_loss(model(f1), tgt_s, 0.1 if p < 0.5 else 5.0)
                        if p >= 0.3:
                            both = torch.stack([f0, f1], dim=1)
                            pn_in = posenet_preprocess_diff(both)
                            pose_l = F.smooth_l1_loss(t_pose(pn_in)['pose'][:,:6], tgt_p)
                            total = 120*seg_l + 0.2*pose_l
                        else:
                            total = 120*seg_l
                        total.backward()
                        opt.step()
                        with torch.no_grad():
                            f0.data.clamp_(0, 255)
                            f1.data.clamp_(0, 255)

                    with torch.no_grad():
                        f1u = F.interpolate(f1.data, (Hc, Wc), mode='bicubic',
                                            align_corners=False).clamp(0,255).round().byte().float()
                        f0u = F.interpolate(f0.data, (Hc, Wc), mode='bicubic',
                                            align_corners=False).clamp(0,255).round().byte().float()
                        ts_in = F.interpolate(f1u, (mH, mW), mode='bilinear')
                        ts = (t_seg(ts_in).argmax(1) != tgt_s).float().mean((1,2))
                        results['ts'].extend(ts.cpu().tolist())
                        tp_pair = F.interpolate(
                            torch.stack([f0u,f1u],1).reshape(-1,3,Hc,Wc),
                            (mH,mW),mode='bilinear').reshape(B,2,3,mH,mW)
                        tpo = t_pose(posenet_preprocess_diff(tp_pair))['pose'][:,:6]
                        results['tp'].extend((tpo-tgt_p).pow(2).mean(1).cpu().tolist())
                    del f0, f1, opt; torch.cuda.empty_cache()

                seg_d = np.mean(results['ts'])
                pose_d = np.mean(results['tp'])
                s_seg = 100 * seg_d
                s_pose = math.sqrt(10 * pose_d)
                dist = s_seg + s_pose

                model_kb = 1000 if 'Mobile' in model_name else 5500
                archive_kb = size_kb + model_kb + 300
                rate = (archive_kb * 1024) / 37_545_489
                score = dist + 25 * rate

                improvement = codec_seg - seg_d
                print(f"  seg_dist={seg_d:.6f} (codec was {codec_seg:.6f}, "
                      f"{'improved' if improvement > 0 else 'WORSE'} by {improvement:.6f})")
                print(f"  pose_mse={pose_d:.6f}")
                print(f"  distortion={dist:.4f}  archive={archive_kb:.0f}KB  "
                      f"TOTAL={score:.3f}  (leader: 1.95)")

    print(f"\n  REFERENCE: flat-blob init + SVD SegNet = 1.31 distortion at 5.5MB")


if __name__ == '__main__':
    main()
