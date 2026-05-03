"""Recompute pose_per_pair.npy for the current MODEL_PATH.
The rank of 'hardest pairs' shifts when the model changes.
Backs up the existing file first.
"""
import sys, os, shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import OUT_H, OUT_W, get_pose6, load_posenet
from train import Generator, load_data_full
import sidecar_explore as se

MODEL_PATH = os.environ.get("MODEL_PATH")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def main():
    assert MODEL_PATH and Path(MODEL_PATH).exists(), f"MODEL_PATH bad: {MODEL_PATH}"
    out_path = OUTPUT_DIR / "pose_per_pair.npy"
    if out_path.exists():
        backup = OUTPUT_DIR / "pose_per_pair.OLD_MODEL.npy"
        if not backup.exists():
            shutil.copy(out_path, backup)
            print(f"[recompute] backed up old → {backup}")

    device = torch.device("cuda")
    gen = Generator().to(device)
    sd = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    gen.load_state_dict(sd, strict=False)
    gen.eval()
    posenet = load_posenet(device)
    data = load_data_full(device)
    masks = data["val_masks"]; poses = data["val_poses"]
    n = poses.shape[0]
    print(f"[recompute] computing pose loss per pair for {n} pairs ...", flush=True)

    pose_per_pair = np.zeros(n, dtype=np.float32)
    bs = 8
    with torch.inference_mode():
        for i in range(0, n, bs):
            j = min(i + bs, n)
            m = masks[i:j].to(device).long()
            p = poses[i:j].to(device).float()
            p1, p2 = gen(m, p)
            f1u = F.interpolate(p1, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            f2u = F.interpolate(p2, (OUT_H, OUT_W), mode='bilinear', align_corners=False)
            pin = se.diff_posenet_input(f1u, f2u)
            fp = get_pose6(posenet, pin).float()
            losses = ((fp - p) ** 2).sum(dim=1).cpu().numpy()
            pose_per_pair[i:j] = losses
    np.save(out_path, pose_per_pair)
    print(f"[recompute] saved {out_path}")
    print(f"[recompute] mean={pose_per_pair.mean():.6f} max={pose_per_pair.max():.6f} "
          f"top10_mean={np.sort(pose_per_pair)[-10:].mean():.6f}")


if __name__ == "__main__":
    main()
