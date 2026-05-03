"""Visualize frames from Phase 2 adversarial .raw output."""
import numpy as np
from pathlib import Path

raw_path = Path("submissions/phase2/inflated/0.raw")
out_dir = Path("viz_phase2")
out_dir.mkdir(exist_ok=True)

H, W, C = 874, 1164, 3
frame_bytes = H * W * C

file_size = raw_path.stat().st_size
num_frames = file_size // frame_bytes
print(f"Raw file: {file_size / 1e6:.1f} MB, {num_frames} frames")

# Extract sample frames
sample_indices = [0, 1, 10, 50, 100, 200, min(300, num_frames - 1)]
sample_indices = [i for i in sample_indices if i < num_frames]

from PIL import Image

for idx in sample_indices:
    offset = idx * frame_bytes
    with open(raw_path, 'rb') as f:
        f.seek(offset)
        data = f.read(frame_bytes)
    frame = np.frombuffer(data, dtype=np.uint8).reshape(H, W, C)
    img = Image.fromarray(frame)
    img.save(out_dir / f"frame_{idx:04d}.png")
    print(f"  Saved frame {idx}: mean RGB = ({frame[:,:,0].mean():.0f}, {frame[:,:,1].mean():.0f}, {frame[:,:,2].mean():.0f}), "
          f"unique colors = {len(np.unique(frame.reshape(-1, 3), axis=0))}")

# Also make a short video from first ~200 frames
print(f"\nMaking video from first {min(200, num_frames)} frames...")
try:
    import av
    container = av.open(str(out_dir / "phase2_adversarial.mp4"), mode='w')
    stream = container.add_stream('libx264', rate=20)
    stream.width = W
    stream.height = H
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '18', 'preset': 'fast'}

    for i in range(min(200, num_frames)):
        offset = i * frame_bytes
        with open(raw_path, 'rb') as f:
            f.seek(offset)
            data = f.read(frame_bytes)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(H, W, C)
        av_frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"  Saved {out_dir / 'phase2_adversarial.mp4'}")
except Exception as e:
    print(f"  Video creation failed: {e}")

# Also render seg map with ideal colors for comparison
print("\nRendering seg map visualization...")
try:
    import struct
    seg_path = Path("submissions/phase2/archive/0/seg.bin")
    colors_path = Path("submissions/phase2/archive/0/colors.bin")

    with open(seg_path, 'rb') as f:
        n, sH, sW, flags = struct.unpack('<IIII', f.read(16))
        has_perm = bool(flags & (1 << 4))
        is_ri = bool(flags & (1 << 3))
        compress_method = flags & 0x3

        perm_lut = None
        if has_perm:
            fwd_perm = list(f.read(5))
            inv_lut = np.zeros(5, dtype=np.uint8)
            for orig, enc in enumerate(fwd_perm):
                inv_lut[enc] = orig
            perm_lut = inv_lut

        import bz2
        raw = bz2.decompress(f.read())
        if is_ri:
            seg_maps = np.frombuffer(raw, dtype=np.uint8).reshape(sH, n, sW)
            seg_maps = np.ascontiguousarray(seg_maps.transpose(1, 0, 2))
        else:
            seg_maps = np.frombuffer(raw, dtype=np.uint8).reshape(n, sH, sW)
        if perm_lut is not None:
            seg_maps = perm_lut[seg_maps]

    colors = np.frombuffer(colors_path.read_bytes(), dtype=np.float32).reshape(5, 3)
    print(f"  Seg maps: {seg_maps.shape}, Ideal colors: {colors}")

    # Render a few seg maps with ideal colors
    for idx in [0, 50, 100]:
        if idx >= n:
            break
        seg = seg_maps[idx]
        vis = colors[seg].astype(np.uint8)
        img = Image.fromarray(vis)
        img_up = img.resize((W, H), Image.NEAREST)
        img_up.save(out_dir / f"segmap_ideal_{idx:04d}.png")
        print(f"  Saved segmap_ideal_{idx:04d}.png")

    # Side-by-side: seg map init vs optimized frame
    if 0 < num_frames:
        with open(raw_path, 'rb') as f:
            data = f.read(frame_bytes)
        opt_frame = np.frombuffer(data, dtype=np.uint8).reshape(H, W, C)
        seg_init = colors[seg_maps[0]].astype(np.uint8)
        seg_init_up = np.array(Image.fromarray(seg_init).resize((W, H), Image.NEAREST))

        side_by_side = np.concatenate([seg_init_up, opt_frame], axis=1)
        Image.fromarray(side_by_side).save(out_dir / "comparison_init_vs_optimized.png")
        print(f"  Saved comparison_init_vs_optimized.png (left=init, right=optimized)")

except Exception as e:
    print(f"  Seg map visualization failed: {e}")

print(f"\nAll outputs in {out_dir.resolve()}")
