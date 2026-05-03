#!/usr/bin/env python
"""
Fast eval: run models on GPU without DALI. Takes any .raw file + archive size.
Caches ground truth on first run.

Usage:
  python -m submissions.av1_repro.fast_eval <raw_path> <archive_bytes>
  python -m submissions.av1_repro.fast_eval submissions/av1_repro/inflated/0.raw 928340
"""
import sys, math, time, struct
import numpy as np, torch, torch.nn.functional as F, einops
from pathlib import Path
from safetensors.torch import load_file

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from frame_utils import camera_size, segnet_model_input_size, AVVideoDataset
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path

W_CAM, H_CAM = camera_size
CACHE = ROOT / 'submissions' / 'av1_repro' / '_cache'
UNCOMPRESSED = 37_545_489


def get_ground_truth(device):
    CACHE.mkdir(exist_ok=True)
    f = CACHE / 'gt.pt'
    if f.exists():
        return torch.load(f, weights_only=True)

    print("Caching ground truth (one-time)...", flush=True)
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    ds = AVVideoDataset(['0.mkv'], data_dir=ROOT / 'videos',
                        batch_size=16, device=torch.device('cpu'))
    ds.prepare_data()

    seg_list, pose_list = [], []
    with torch.inference_mode():
        for _, idx, batch in ds:
            batch = batch.to(device)
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
            seg_in = segnet.preprocess_input(x)
            seg_list.append(segnet(seg_in).argmax(1).cpu())
            pn_in = posenet.preprocess_input(x)
            pose_list.append(posenet(pn_in)['pose'][:, :6].cpu())

    gt = {'seg': torch.cat(seg_list), 'pose': torch.cat(pose_list)}
    torch.save(gt, f)
    del segnet, posenet; torch.cuda.empty_cache()
    return gt


def main():
    raw_path = Path(sys.argv[1])
    archive_bytes = int(sys.argv[2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.time()

    gt = get_ground_truth(device)
    N = gt['seg'].shape[0]

    # Load compressed .raw
    raw = np.fromfile(raw_path, dtype=np.uint8).reshape(N * 2, H_CAM, W_CAM, 3)

    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    seg_dists, pose_dists = [], []
    bs = 16

    with torch.inference_mode():
        for i in range(0, N, bs):
            end = min(i + bs, N)
            # Each pair: frame_0 at 2*i, frame_1 at 2*i+1
            f0 = torch.from_numpy(raw[2*i:2*end:2].copy()).to(device).float()
            f1 = torch.from_numpy(raw[2*i+1:2*end:2].copy()).to(device).float()
            x = torch.stack([f0, f1], dim=1)  # (B, 2, H, W, 3)
            x = einops.rearrange(x, 'b t h w c -> b t c h w')

            # Seg (last frame)
            seg_in = segnet.preprocess_input(x)
            seg_pred = segnet(seg_in).argmax(1)
            gt_seg = gt['seg'][i:end].to(device)
            seg_dists.extend((seg_pred != gt_seg).float().mean((1,2)).cpu().tolist())

            # Pose (both frames)
            pn_in = posenet.preprocess_input(x)
            pn_out = posenet(pn_in)['pose'][:, :6]
            gt_pose = gt['pose'][i:end].to(device)
            pose_dists.extend((pn_out - gt_pose).pow(2).mean(1).cpu().tolist())

    seg_d = np.mean(seg_dists)
    pose_d = np.mean(pose_dists)
    rate = archive_bytes / UNCOMPRESSED
    score = 100 * seg_d + math.sqrt(10 * pose_d) + 25 * rate

    print(f"seg={seg_d:.8f} pose={pose_d:.8f} "
          f"size={archive_bytes/1024:.1f}KB "
          f"100s={100*seg_d:.4f} sqrtp={math.sqrt(10*pose_d):.4f} "
          f"25r={25*rate:.4f} score={score:.4f} "
          f"({time.time()-t0:.1f}s)")


if __name__ == '__main__':
    main()
