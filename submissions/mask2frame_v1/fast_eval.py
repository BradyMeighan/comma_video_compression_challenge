#!/usr/bin/env python
"""
Quick local evaluation: inflate from archive/ dir, compute score.
Usage: python -m submissions.mask2frame_v1.fast_eval [--device cuda:0]
"""
import sys, os, math, argparse, mmap
from pathlib import Path

import torch
import einops
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, seq_len
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

UNCOMPRESSED = 37_545_489

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--archive-dir", type=Path, default=Path(__file__).parent / "archive")
    parser.add_argument("--inflated-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    archive_dir = args.archive_dir

    if args.inflated_dir is None:
        inflated_dir = archive_dir.parent / "_eval_inflated"
    else:
        inflated_dir = args.inflated_dir

    # Step 1: inflate if needed
    raw_path = inflated_dir / "0.raw"
    names_file = ROOT / "public_test_video_names.txt"
    if not raw_path.exists():
        print("Inflating...")
        inflated_dir.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run([
            sys.executable, "-m", "submissions.mask2frame_v1.inflate",
            str(archive_dir), str(inflated_dir), str(names_file)
        ], check=True, cwd=str(ROOT))

    # Step 2: compute size
    total_size = 0
    for f in ["model.pt.br", "mask.obu.br", "pose.npy.br"]:
        p = archive_dir / f
        if p.exists():
            total_size += p.stat().st_size
    rate = total_size / UNCOMPRESSED
    print(f"Archive: {total_size:,} bytes, rate={rate:.6f}")

    # Step 3: load GT cache
    cache_dir = archive_dir.parent / "_cache"
    cache_dir.mkdir(exist_ok=True)
    gt_cache = cache_dir / "gt.pt"

    dist_net = DistortionNet().eval().to(device)
    dist_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    files = [l.strip() for l in names_file.read_text().splitlines() if l.strip()]

    if gt_cache.exists():
        print("Loading GT cache...")
        gt_data = torch.load(gt_cache, map_location="cpu", weights_only=True)
    else:
        print("Building GT cache from videos...")
        from frame_utils import AVVideoDataset
        ds = AVVideoDataset(files, data_dir=ROOT / "videos", batch_size=args.batch_size, device=device, num_threads=2, seed=1234, prefetch_queue_depth=2)
        ds.prepare_data()
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0)
        gt_seg, gt_pose = [], []
        with torch.inference_mode():
            for _, _, batch in tqdm(dl, desc="GT"):
                batch = batch.to(device)
                po, so = dist_net(batch)
                gt_seg.append(so.cpu())
                gt_pose.append({k: v.cpu() for k, v in po.items()})
        gt_data = {"seg": torch.cat(gt_seg), "pose_keys": list(gt_pose[0].keys()),
                   **{f"pose_{k}": torch.cat([p[k] for p in gt_pose]) for k in gt_pose[0].keys()}}
        torch.save(gt_data, gt_cache)

    # Step 4: eval reconstructed
    W, H = camera_size
    n_frames = 1200
    n_pairs = n_frames // seq_len
    frame_bytes = H * W * 3

    raw = raw_path
    assert raw.exists(), f"Missing {raw}"
    raw_size = raw.stat().st_size
    assert raw_size == n_frames * frame_bytes, f"Size mismatch: {raw_size} vs {n_frames * frame_bytes}"

    mm = open(raw, 'rb')
    total_seg, total_pose, total_n = 0.0, 0.0, 0

    with torch.inference_mode():
        for start in tqdm(range(0, n_pairs, args.batch_size), desc="Eval"):
            bs = min(args.batch_size, n_pairs - start)
            frames = []
            for i in range(bs):
                pair_idx = start + i
                offset = pair_idx * seq_len * frame_bytes
                mm.seek(offset)
                pair_bytes = mm.read(seq_len * frame_bytes)
                pair = torch.frombuffer(bytearray(pair_bytes), dtype=torch.uint8).reshape(seq_len, H, W, 3)
                frames.append(pair)
            batch = torch.stack(frames).to(device)
            po, so = dist_net(batch)

            gt_seg_batch = gt_data["seg"][start:start+bs].to(device)
            gt_pose_batch = {k: gt_data[f"pose_{k}"][start:start+bs].to(device) for k in gt_data["pose_keys"]}

            seg_dist = (so.argmax(dim=1) != gt_seg_batch.argmax(dim=1)).float().mean(dim=tuple(range(1, so.argmax(dim=1).ndim)))
            pose_dist = sum(
                (po[k][..., :h//2] - gt_pose_batch[k][..., :h//2]).pow(2).mean(dim=tuple(range(1, po[k].ndim)))
                for k, h in [("pose", 12)]
            )
            total_seg += seg_dist.sum().item()
            total_pose += pose_dist.sum().item()
            total_n += bs

    mm.close()

    avg_seg = total_seg / total_n
    avg_pose = total_pose / total_n
    score = 100 * avg_seg + math.sqrt(max(0, 10 * avg_pose)) + 25 * rate
    print(f"\n=== Results ===")
    print(f"  seg={avg_seg:.8f}  pose={avg_pose:.8f}")
    print(f"  100*seg={100*avg_seg:.4f}  sqrt(10*pose)={math.sqrt(max(0,10*avg_pose)):.4f}  25*rate={25*rate:.4f}")
    print(f"  SCORE = {score:.4f}")

if __name__ == "__main__":
    main()
