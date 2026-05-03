# Phase 4: The Hybrid Pipeline -- Combining Codec Initialization with Adversarial Optimization

**Status:** Implementation task document
**Goal:** Achieve a score below 1.5, potentially approaching 1.0
**Strategy:** Use an aggressively compressed AV1/VVC stream as initialization for gradient-based adversarial optimization, combined with stored segmentation maps and pose vectors

---

## 1. Why a Hybrid Approach

The scoring formula (`evaluate.py:92`) is:

```
score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
```

Each prior phase attacks one or two terms effectively but leaves the others suboptimal. The hybrid combines all advantages:

| Approach | SegNet Term | Rate Term | PoseNet Term | Total (est.) | Weakness |
|----------|-------------|-----------|--------------|--------------|----------|
| Phase 1: Codec only (SVT-AV1, CRF 28-32) | 0.5-0.9 | 0.5-0.7 | 1.5-2.0 | 2.5-3.0 | Cannot push SegNet below ~0.005 distortion; PoseNet hard to control |
| Phase 2: Adversarial decode (from scratch) | 0.1-0.3 | 1.0-1.5 | 0.7-1.0 | 1.5-2.5 | Seg maps are large (1.5-3 MB); generating from noise/flat color needs 30-50 iterations per frame; risky on timing |
| Phase 3: INR-based | 0.5-1.0 | 0.5-1.0 | 1.0-1.5 | 2.5-3.0 | Model capacity vs archive size tradeoff; hard to get both small and accurate |
| **Phase 4: Hybrid** | **0.1-0.3** | **0.4-0.9** | **0.7-1.2** | **1.2-2.3** | Most complexity, but each component is well-understood |

The key insight: a heavily compressed video at CRF 35-50 looks terrible to the human eye, but it is already very close to the correct answer in the evaluation networks' feature spaces. Starting gradient descent from this initialization means only 5-10 iterations are needed instead of 30-50, which:

1. Fits easily within the 30-minute T4 budget
2. Produces more stable convergence (smaller perturbations from a good starting point)
3. Allows more aggressive compression (smaller archive) because we fix distortion at decode time

---

## 2. Architecture Overview

### Encoder (offline, unlimited time)

```
Original video (1164x874, 1200 frames, 37.5 MB)
    |
    v
[1] Downscale to 512x384 via Lanczos
    |
    +---> [2] Run SegNet on all 600 odd frames --> argmax maps (values 0-4)
    |         Shape per frame: (384, 512), dtype uint8 (3 bits used)
    |
    +---> [3] Run PoseNet on all 600 frame pairs --> 6-dim pose vectors
    |         Shape per pair: (6,), dtype float32
    |
    +---> [4] SegNet-aware preprocessing (optional: blur sky/uniform regions)
    |
    +---> [5] Encode with SVT-AV1 at aggressive CRF (35-45)
    |         Input: 512x384 @ 20fps
    |         Output: compressed_video.mkv (~100-300 KB)
    |
    v
[6] Package everything:
    archive/
      0.mkv            # aggressively compressed video
      seg_maps.bin      # compressed segmentation maps
      pose.bin          # compressed pose vectors
      config.json       # metadata
    --> zip --> archive.zip (~400 KB - 1.2 MB)
```

### Decoder (T4 GPU, 30-minute budget)

```
archive.zip
    |
    v
[1] Unzip --> archive/0.mkv + seg_maps.bin + pose.bin + config.json
    |
    v
[2] Decode compressed video via PyAV
    |   Result: 1200 frames at 512x384, heavily degraded but structurally close
    |
    v
[3] Load SegNet and PoseNet from repo models/ directory
    |   segnet = SegNet(); segnet.load_state_dict(load_file('models/segnet.safetensors'))
    |   posenet = PoseNet(); posenet.load_state_dict(load_file('models/posenet.safetensors'))
    |
    v
[4] Load seg_maps.bin --> 600 target segmentation maps, shape (600, 384, 512)
    Load pose.bin --> 600 target pose vectors, shape (600, 6)
    |
    v
[5] For each frame pair (even_frame, odd_frame):
    |   a. Initialize: decoded frames as starting tensors (already close)
    |   b. Run 5-10 gradient descent steps:
    |      - SegNet margin loss on odd frame (match stored argmax map)
    |      - PoseNet MSE loss on frame pair (match stored pose vector)
    |      - Regularization loss (stay close to decoded frame)
    |   c. Clamp to [0, 255], round to uint8
    |   d. Verify SegNet argmax matches target; fix any mismatches
    |   e. For even frame: 3-5 PoseNet-only optimization steps
    |
    v
[6] Upscale all 1200 frames from 512x384 to 1164x874 via bicubic interpolation
    |
    v
[7] Write flat binary .raw file (1200 * 874 * 1164 * 3 = 3,661,113,600 bytes)
```

---

## 3. Detailed Data Flow with Tensor Shapes

This section traces exact tensor shapes through every operation, referencing the actual code in `modules.py` and `frame_utils.py`.

### 3.1 SegNet Data Path

From `modules.py:103-113` and `modules.py:143-148`:

```
Raw input: (B, T=2, H=874, W=1164, C=3) uint8
    |
    v  DistortionNet.preprocess_input (modules.py:143-148)
    |  rearrange 'b t h w c -> b t c h w' --> (B, 2, 3, 874, 1164) float32
    |
    v  SegNet.preprocess_input (modules.py:107-109)
    |  x = x[:, -1, ...]  --> takes ONLY the last frame (odd frame)
    |  Shape: (B, 3, 874, 1164) float32
    |
    v  F.interpolate(x, size=(384, 512), mode='bilinear')
    |  Shape: (B, 3, 384, 512) float32
    |
    v  SegNet.forward (smp.Unet with efficientnet_b2 encoder)
    |  Output: (B, 5, 384, 512) float32 -- raw logits for 5 classes
    |
    v  SegNet.compute_distortion (modules.py:111-113)
    |  argmax(dim=1) --> (B, 384, 512) int64 -- per-pixel class labels
    |  Compare original vs reconstructed: fraction of disagreeing pixels
    |  Output: (B,) float32
```

Critical facts:
- SegNet only sees the **odd-indexed frame** (index 1 in each pair): frames 1, 3, 5, ..., 1199
- Input resolution is **bilinear-downscaled to (384, 512)** -- the exact `segnet_model_input_size`
- Distortion is binary per pixel: argmax match or not. Logit magnitudes do not matter.

### 3.2 PoseNet Data Path

From `modules.py:61-84` and `frame_utils.py:51-78`:

```
Raw input: (B, T=2, H=874, W=1164, C=3) uint8
    |
    v  DistortionNet.preprocess_input (modules.py:143-148)
    |  rearrange 'b t h w c -> b t c h w' --> (B, 2, 3, 874, 1164) float32
    |
    v  PoseNet.preprocess_input (modules.py:70-74)
    |  rearrange 'b t c h w -> (b*t) c h w' --> (2B, 3, 874, 1164)
    |  F.interpolate to (384, 512) --> (2B, 3, 384, 512)
    |
    v  rgb_to_yuv6 (frame_utils.py:51-78)
    |  BT.601: Y = 0.299*R + 0.587*G + 0.114*B
    |  Chroma 4:2:0 subsampling (2x2 box average)
    |  6 channels: [y00, y10, y01, y11, U_sub, V_sub]
    |  Output: (2B, 6, 192, 256)
    |
    v  rearrange '(b t) c h w -> b (t c) h w' with t=2, c=6
    |  Stack 2 frames' 6 channels --> 12 channels
    |  Output: (B, 12, 192, 256)
    |
    v  PoseNet.forward (modules.py:76-80)
    |  Normalize: (x - 127.5) / 63.75
    |  fastvit_t12 backbone --> (B, 2048)
    |  Summarizer: Linear(2048, 512) + ReLU + ResBlock(512) --> (B, 512)
    |  Hydra head: ResBlock(512) --> Linear(512, 32) + residual --> Linear(32, 12) --> (B, 12)
    |  Output: {'pose': (B, 12)}
    |
    v  PoseNet.compute_distortion (modules.py:82-84)
    |  Uses ONLY first 6 of 12 dims: out[..., :6]
    |  MSE between original and reconstructed
    |  Output: (B,) float32
```

Critical facts:
- PoseNet sees **both frames** of each pair (even and odd)
- Input goes through RGB-to-YUV6 conversion with 4:2:0 subsampling
- Final input resolution to the backbone is **(192, 256)** due to the 2x downsampling in YUV6
- Only the first 6 of 12 output dimensions contribute to distortion

### 3.3 Working Resolution for Optimization

Since both networks resize to `(384, 512)` before any processing, we operate the gradient descent loop at **512x384 RGB**. This is crucial:

- We encode the video at 512x384 (or nearby even dimensions)
- We run optimization at 512x384
- We upscale to 1164x874 only at the very end for the .raw output
- The upscaling is deterministic (bicubic), so the evaluation pipeline's bilinear downscale of our 1164x874 output will closely match our 512x384 optimized frames

Note: there is a subtle mismatch. We optimize at 512x384 and then upscale to 1164x874. The evaluator then downscales 1164x874 back to 512x384 via bilinear. The round-trip (upscale then downscale) introduces small interpolation errors. To minimize this:
- Use `align_corners=False` consistently (matching `F.interpolate` defaults)
- Consider optimizing at 1164x874 directly if VRAM allows (probably too expensive)
- Or: after optimization, upscale to 1164x874, then simulate the evaluator's downscale, check if SegNet argmax is preserved, fix if not

---

## 4. Task 1: Aggressive Codec Compression

### Rationale

In a codec-only approach, CRF must be conservative (28-32) to avoid flipping SegNet argmax classes. But since the hybrid approach fixes distortion at decode time via gradient descent, we can encode at CRF 35-50 where the video looks terrible but the structural layout is preserved.

### Encoding Command (SVT-AV1)

```bash
# Downscale to 512x384 first
ffmpeg -y -i videos/0.mkv \
  -vf "scale=512:384:flags=lanczos" \
  -pix_fmt yuv420p \
  -f rawvideo /tmp/input_512x384.yuv

# Encode with SVT-AV1 at aggressive settings
SvtAv1EncApp \
  -i /tmp/input_512x384.yuv \
  -w 512 -h 384 \
  --fps 20 \
  --input-depth 8 \
  --crf 40 \
  --preset 2 \
  --keyint 120 \
  --irefresh-type 2 \
  --film-grain 0 \
  --tile-rows 0 --tile-columns 0 \
  -b /tmp/encoded.ivf

# Or via ffmpeg wrapper:
ffmpeg -y -i videos/0.mkv \
  -vf "scale=512:384:flags=lanczos" \
  -c:v libsvtav1 \
  -crf 40 \
  -preset 2 \
  -svtav1-params "keyint=120:irefresh-type=2:film-grain=0" \
  -pix_fmt yuv420p \
  archive/0.mkv
```

### CRF Sweep Targets

| CRF | Expected Size (512x384, 60s @ 20fps) | Visual Quality | Suitable for Hybrid? |
|-----|---------------------------------------|----------------|---------------------|
| 30  | ~400-600 KB | Acceptable | Yes, but larger than needed |
| 35  | ~200-400 KB | Noticeable artifacts | Yes, good sweet spot |
| 40  | ~100-250 KB | Severe blocking | Yes, still recognizable structure |
| 45  | ~60-150 KB | Very degraded | Borderline -- may need more iterations |
| 50  | ~30-80 KB | Extremely degraded | Risky -- structure may be too damaged |

### Why Aggressive CRF Works Here

At CRF 40, the decoded frame will have:
- Severe blocking artifacts
- Washed-out colors
- Blurred edges and loss of fine detail

But critically, it will still preserve:
- The overall scene layout (sky at top, road at bottom, cars in middle)
- Approximate color distributions per region
- Frame-to-frame temporal consistency

This means the decoded frame, when fed through SegNet, will get maybe 80-90% of pixels correct (vs 99%+ at CRF 30). The gradient descent loop only needs to fix the remaining 10-20% of pixels, which requires far fewer iterations than generating from scratch.

### Encoding Script

```python
#!/usr/bin/env python
"""compress_video.py -- Encode video at aggressive CRF for hybrid pipeline."""
import subprocess
import sys

def encode_video(input_path, output_path, crf=40, preset=2, keyint=120):
    """Encode video at 512x384 with SVT-AV1 at aggressive CRF."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', input_path,
        '-vf', 'scale=512:384:flags=lanczos',
        '-c:v', 'libsvtav1',
        '-crf', str(crf),
        '-preset', str(preset),
        '-svtav1-params', f'keyint={keyint}:irefresh-type=2:film-grain=0',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(cmd, check=True)

    import os
    size = os.path.getsize(output_path)
    print(f"Encoded video: {size:,} bytes ({size/1024:.1f} KB)")
    return size

if __name__ == '__main__':
    encode_video(sys.argv[1], sys.argv[2], crf=int(sys.argv[3]) if len(sys.argv) > 3 else 40)
```

---

## 5. Task 2: Compact Segmentation Map Encoding

### What We Need to Store

- 600 segmentation maps (one per odd frame: frames 1, 3, 5, ..., 1199)
- Each map: 384 rows x 512 columns = 196,608 pixels
- Each pixel: one of 5 classes (values 0-4), requiring 3 bits
- Raw uncompressed: 600 * 196,608 * 3 bits = 44,236,800 bytes (44.2 MB) at 8 bits per pixel, or 600 * 196,608 * 3/8 = 44.2 MB at packed 3 bits

Clearly, raw storage is way too large. But there is massive redundancy:

### Temporal Redundancy

Consecutive dashcam frames have very similar segmentation. Between frame 1 and frame 3 (which are 2 frames apart in the original video), typically:
- 95-98% of pixels have the same class
- Changes concentrate at object boundaries and moving objects
- The sky, road surface, and static background are nearly identical

### Strategy A: Delta-Coded + Entropy Compression

```python
import numpy as np
import zlib

def encode_seg_maps(seg_maps):
    """
    seg_maps: np.ndarray of shape (600, 384, 512), dtype uint8, values 0-4
    Returns: bytes
    """
    # Pack each pixel into 4 bits (wastes 1 bit but aligns better)
    # Or pack 2 pixels per byte (each pixel needs 3 bits, but 4-bit nibble is simpler)

    encoded_frames = []

    # First frame: store directly
    first = seg_maps[0].astype(np.uint8)  # (384, 512)
    encoded_frames.append(first.tobytes())

    # Subsequent frames: XOR with previous, then compress
    for i in range(1, len(seg_maps)):
        delta = (seg_maps[i] != seg_maps[i-1]).astype(np.uint8)  # Binary: changed or not
        # For changed pixels, store the new class
        changed_mask = delta  # (384, 512), 0 or 1

        # Encode: delta map (1 bit per pixel) + new values for changed pixels
        # Pack delta into bits
        delta_bits = np.packbits(changed_mask.flatten())

        # Extract new values where changed
        new_values = seg_maps[i][changed_mask.astype(bool)]  # 1D array of new classes

        # Pack new values: 2 pixels per byte (4 bits each)
        if len(new_values) % 2 == 1:
            new_values = np.append(new_values, 0)
        packed_values = (new_values[0::2] << 4) | new_values[1::2]

        frame_data = delta_bits.tobytes() + packed_values.astype(np.uint8).tobytes()
        encoded_frames.append(frame_data)

    # Concatenate with length headers
    result = len(seg_maps).to_bytes(4, 'little')
    for frame_data in encoded_frames:
        result += len(frame_data).to_bytes(4, 'little')
        result += frame_data

    # Final zlib compression
    compressed = zlib.compress(result, level=9)
    return compressed
```

### Strategy B: FFV1 Lossless Video Codec

FFV1 is a lossless video codec that excels at compressing low-color-count video. Since our maps have only 5 distinct values, this is ideal.

```bash
# Convert seg maps to a video with 5-color palette, then encode with FFV1
ffmpeg -y -f rawvideo -pix_fmt gray -s 512x384 -r 20 \
  -i seg_maps_raw.bin \
  -c:v ffv1 -level 3 -slicecrc 1 \
  -pix_fmt gray \
  seg_maps.mkv
```

Alternatively, in Python:

```python
import av
import numpy as np

def encode_seg_maps_ffv1(seg_maps, output_path):
    """
    seg_maps: np.ndarray of shape (600, 384, 512), dtype uint8, values 0-4
    Encode as FFV1 lossless video. Values 0-4 stored in grayscale channel.
    """
    container = av.open(output_path, mode='w')
    stream = container.add_stream('ffv1', rate=20)
    stream.width = 512
    stream.height = 384
    stream.pix_fmt = 'gray'
    stream.options = {'level': '3', 'slicecrc': '1'}

    for i in range(len(seg_maps)):
        # Scale values to spread across byte range for better compression
        # 0->0, 1->51, 2->102, 3->153, 4->204 (multiply by 51)
        frame_data = (seg_maps[i] * 51).astype(np.uint8)
        frame = av.VideoFrame.from_ndarray(frame_data, format='gray')
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
```

### Strategy C: Reduced Resolution Maps

SegNet boundaries are relatively smooth. We can store maps at half resolution (256x192) and upscale at decode time via nearest-neighbor interpolation:

```python
import torch.nn.functional as F

def downsample_seg_map(seg_map_384x512):
    """Downsample seg map using mode (most common class) in each 2x2 block."""
    # seg_map: (384, 512), values 0-4
    h, w = seg_map_384x512.shape
    # Reshape into 2x2 blocks
    blocks = seg_map_384x512.reshape(h//2, 2, w//2, 2)  # (192, 2, 256, 2)
    # For each block, take the most common value (mode)
    # Simple approach: just take top-left pixel (nearest-neighbor downsample)
    downsampled = blocks[:, 0, :, 0]  # (192, 256)
    return downsampled

def upsample_seg_map(seg_map_192x256):
    """Upsample via nearest-neighbor to restore 384x512."""
    t = torch.from_numpy(seg_map_192x256).unsqueeze(0).unsqueeze(0).float()
    up = F.interpolate(t, size=(384, 512), mode='nearest').squeeze().numpy().astype(np.uint8)
    return up
```

This halves the data volume but may introduce boundary errors. Must be validated against the full-resolution SegNet output.

### Strategy D: Hybrid -- Full Resolution for Key Frames, Half for Others

Store every 10th map at full resolution, others at half resolution. At decode time, use gradient descent to fix any boundary errors from the upscaling. This is self-correcting because the optimization loop verifies SegNet output.

### Expected Sizes

| Method | Estimated Total Size |
|--------|---------------------|
| Raw (uint8, 600 * 384 * 512) | 117.9 MB |
| Delta + zlib (Strategy A) | 300-800 KB |
| FFV1 lossless video (Strategy B) | 200-600 KB |
| Half-res + delta + zlib (Strategy C + A) | 100-300 KB |
| zstd on raw (simple baseline) | 2-5 MB |

Strategy B (FFV1) or Strategy A (delta + zlib) are the recommended approaches. FFV1 handles temporal redundancy natively and requires less custom code.

### Extraction Code

```python
def extract_seg_maps(video_path, segnet, device):
    """
    Run SegNet on all odd frames of the original video.
    Returns: np.ndarray of shape (600, 384, 512), dtype uint8
    """
    import av
    from frame_utils import segnet_model_input_size

    container = av.open(video_path)
    stream = container.streams.video[0]

    seg_maps = []
    frame_idx = 0

    for frame in container.decode(stream):
        if frame_idx % 2 == 1:  # Odd frames only (1, 3, 5, ..., 1199)
            # Convert to RGB tensor
            from frame_utils import yuv420_to_rgb
            rgb = yuv420_to_rgb(frame)  # (874, 1164, 3) uint8

            # Resize to SegNet input size
            rgb_tensor = rgb.permute(2, 0, 1).unsqueeze(0).float().to(device)
            # (1, 3, 874, 1164)
            rgb_resized = F.interpolate(
                rgb_tensor,
                size=(segnet_model_input_size[1], segnet_model_input_size[0]),  # (384, 512)
                mode='bilinear',
                align_corners=False
            )
            # (1, 3, 384, 512)

            with torch.no_grad():
                logits = segnet(rgb_resized)  # (1, 5, 384, 512)
                seg_map = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                # (384, 512), values 0-4

            seg_maps.append(seg_map)

        frame_idx += 1

    container.close()
    return np.stack(seg_maps)  # (600, 384, 512)
```

---

## 6. Task 3: Pose Vector Extraction and Compression

### What We Need to Store

- 600 pose vectors (one per frame pair)
- Each vector: 6 float32 values (the first 6 of 12 PoseNet output dims)
- Raw: 600 * 6 * 4 bytes = 14,400 bytes (14.4 KB)

This is already tiny. Further compression options:

### Float16 Quantization

```python
import numpy as np

def compress_pose_vectors(pose_vectors):
    """
    pose_vectors: np.ndarray of shape (600, 6), dtype float32
    Returns: bytes
    """
    # Convert to float16 -- 50% size reduction with minimal precision loss
    fp16 = pose_vectors.astype(np.float16)  # (600, 6), 7,200 bytes

    # zlib compress
    compressed = zlib.compress(fp16.tobytes(), level=9)
    return compressed  # Expected: ~4-6 KB

def decompress_pose_vectors(data):
    """Returns: np.ndarray of shape (600, 6), dtype float32"""
    raw = zlib.decompress(data)
    fp16 = np.frombuffer(raw, dtype=np.float16).reshape(600, 6)
    return fp16.astype(np.float32)
```

### Extraction Code

```python
def extract_pose_vectors(video_path, posenet, device):
    """
    Run PoseNet on all frame pairs of the original video.
    Returns: np.ndarray of shape (600, 6), dtype float32
    """
    import av
    from frame_utils import yuv420_to_rgb, segnet_model_input_size, rgb_to_yuv6
    import einops

    container = av.open(video_path)
    stream = container.streams.video[0]

    pose_vectors = []
    frames_buffer = []

    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (874, 1164, 3) uint8
        frames_buffer.append(rgb)

        if len(frames_buffer) == 2:
            even_frame, odd_frame = frames_buffer
            # Stack into batch: (1, 2, 874, 1164, 3)
            pair = torch.stack([even_frame, odd_frame]).unsqueeze(0).float().to(device)
            # (1, 2, 874, 1164, 3) --> rearrange to (1, 2, 3, 874, 1164)
            pair_chw = einops.rearrange(pair, 'b t h w c -> b t c h w')

            # Run PoseNet preprocessing
            posenet_in = posenet.preprocess_input(pair_chw)  # (1, 12, 192, 256)

            with torch.no_grad():
                out = posenet(posenet_in)  # {'pose': (1, 12)}
                pose_vec = out['pose'][0, :6].cpu().numpy()  # (6,) -- first 6 dims only

            pose_vectors.append(pose_vec)
            frames_buffer = []

    container.close()
    return np.stack(pose_vectors)  # (600, 6)
```

### Important: PoseNet Distortion Computation

From `modules.py:82-84`, the distortion is computed as MSE between the outputs of PoseNet on the original vs reconstructed frame pairs, using only the first 6 dimensions. We store the PoseNet output on the **original** frames, then at decode time we optimize the reconstructed frames so their PoseNet output matches.

The distortion enters the score as `sqrt(10 * posenet_dist)`. The baseline has `posenet_dist = 0.38`, contributing `sqrt(3.8) = 1.95` to the score. If we can achieve `posenet_dist = 0.05`, the contribution drops to `sqrt(0.5) = 0.71` -- a saving of 1.24 points.

---

## 7. Task 3: The Gradient Descent Optimization Loop

This is the most critical component of the hybrid pipeline. It takes degraded decoded frames and optimizes them to match stored SegNet/PoseNet targets.

### 7.1 Loss Functions

#### SegNet Margin Loss

The goal: ensure that for every pixel, the target class has the highest logit with a comfortable margin. This makes the optimization robust to uint8 rounding and the upscale/downscale round-trip.

```python
def margin_loss(logits, target_map, margin=2.0):
    """
    Computes margin loss for segmentation matching.

    Args:
        logits: (B, 5, H, W) -- raw SegNet output logits
        target_map: (B, H, W) -- target class per pixel, values 0-4 (int64)
        margin: desired gap between target class logit and runner-up

    Returns:
        scalar loss (mean over all pixels and batch)
    """
    B, C, H, W = logits.shape  # C=5

    # Gather the logit for the target class at each pixel
    # target_map: (B, H, W) --> (B, 1, H, W) for gather
    target_logit = logits.gather(1, target_map.unsqueeze(1).long())  # (B, 1, H, W)

    # Create mask to exclude the target class, then find the max among remaining classes
    # mask: (B, 5, H, W), 1 everywhere except at the target class channel
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target_map.unsqueeze(1).long(), False)

    # Set target class logits to -inf so they don't win the max
    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9
    runner_up = masked_logits.max(dim=1, keepdim=True).values  # (B, 1, H, W)

    # Loss: penalize when target_logit - runner_up < margin
    loss = F.relu(margin - (target_logit - runner_up)).mean()

    return loss
```

Why margin loss instead of cross-entropy:
- Cross-entropy keeps pushing logits apart indefinitely, wasting optimization budget
- Margin loss stops once the gap is sufficient, letting the optimizer focus on harder pixels
- A margin of 2.0 provides buffer against uint8 rounding perturbations
- Once `target_logit - runner_up >= margin` for a pixel, that pixel contributes zero loss

#### PoseNet MSE Loss

```python
def posenet_loss(posenet, img_even, img_odd, target_pose, device):
    """
    Compute MSE loss between PoseNet output and target pose vector.

    Args:
        posenet: loaded PoseNet model (eval mode, on device)
        img_even: (1, 3, 384, 512) float32 -- even frame being optimized
        img_odd: (1, 3, 384, 512) float32 -- odd frame being optimized
        target_pose: (1, 6) float32 -- target pose vector from original video

    Returns:
        scalar loss
    """
    # Stack frames: (1, 2, 3, 384, 512)
    pair = torch.stack([img_even, img_odd], dim=1)

    # Run PoseNet preprocessing (handles YUV conversion internally)
    posenet_in = posenet.preprocess_input(pair)  # (1, 12, 192, 256)

    # Forward pass
    out = posenet(posenet_in)  # {'pose': (1, 12)}
    pred_pose = out['pose'][:, :6]  # (1, 6) -- first 6 dims only

    # MSE loss against target
    loss = F.mse_loss(pred_pose, target_pose)

    return loss
```

#### Regularization Loss

Keeps the optimized frame close to the decoded frame, preventing adversarial drift:

```python
def regularization_loss(img_current, img_decoded):
    """
    L2 regularization to keep optimized frame close to decoded initialization.

    Args:
        img_current: (1, 3, 384, 512) float32 -- current optimized frame
        img_decoded: (1, 3, 384, 512) float32 -- original decoded frame (detached)

    Returns:
        scalar loss
    """
    return F.mse_loss(img_current, img_decoded)
```

### 7.2 Complete Optimization Loop

```python
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

def optimize_frame_pair(
    decoded_even,      # (1, 3, 384, 512) float32, decoded from compressed video
    decoded_odd,       # (1, 3, 384, 512) float32, decoded from compressed video
    target_seg_map,    # (1, 384, 512) int64, target segmentation for odd frame
    target_pose,       # (1, 6) float32, target pose vector
    segnet,            # loaded SegNet model (eval mode)
    posenet,           # loaded PoseNet model (eval mode)
    device,
    n_steps=10,        # number of gradient descent steps
    lr=2.0,            # learning rate (per pixel)
    seg_weight=100.0,  # weight for SegNet margin loss
    pose_weight=10.0,  # weight for PoseNet MSE loss
    reg_weight=0.01,   # weight for regularization loss
    margin=2.0,        # margin for SegNet loss
    use_amp=True,      # mixed precision
):
    """
    Optimize a frame pair starting from decoded video frames.

    Returns:
        optimized_even: (1, 3, 384, 512) uint8
        optimized_odd: (1, 3, 384, 512) uint8
    """
    # Clone decoded frames as starting point
    img_odd = decoded_odd.clone().detach().requires_grad_(True)
    img_even = decoded_even.clone().detach().requires_grad_(True)

    # Keep detached copies for regularization
    decoded_odd_detached = decoded_odd.clone().detach()
    decoded_even_detached = decoded_even.clone().detach()

    optimizer = torch.optim.Adam([img_odd, img_even], lr=lr)
    scaler = GradScaler(enabled=use_amp)

    # Learning rate schedule: cosine decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.1)

    for step in range(n_steps):
        optimizer.zero_grad()

        with autocast(enabled=use_amp, dtype=torch.float16):
            # --- SegNet loss (odd frame only) ---
            # SegNet expects (B, 3, 384, 512) float input
            seg_logits = segnet(img_odd)  # (1, 5, 384, 512)
            seg_loss = margin_loss(seg_logits, target_seg_map, margin=margin)

            # --- PoseNet loss (both frames) ---
            pose_loss = posenet_loss(posenet, img_even, img_odd, target_pose, device)

            # --- Regularization ---
            reg_loss_odd = F.mse_loss(img_odd, decoded_odd_detached)
            reg_loss_even = F.mse_loss(img_even, decoded_even_detached)
            reg_loss = reg_loss_odd + reg_loss_even

            # --- Total loss ---
            total_loss = seg_weight * seg_loss + pose_weight * pose_loss + reg_weight * reg_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Clamp pixel values to valid range
        with torch.no_grad():
            img_odd.data.clamp_(0.0, 255.0)
            img_even.data.clamp_(0.0, 255.0)

        scheduler.step()

    # Round to uint8
    optimized_odd = img_odd.detach().round().clamp(0, 255).to(torch.uint8)
    optimized_even = img_even.detach().round().clamp(0, 255).to(torch.uint8)

    return optimized_even, optimized_odd
```

### 7.3 Handling the PoseNet Preprocessing Differentiability

A critical implementation detail: `posenet.preprocess_input` calls `rgb_to_yuv6` which involves arithmetic operations on the RGB tensor. For gradient flow, we need these operations to be differentiable. Let us trace:

```python
# rgb_to_yuv6 (frame_utils.py:51-78):
# Y = R * 0.299 + G * 0.587 + B * 0.114  --> differentiable (linear)
# U = (B - Y) / 1.772 + 128.0            --> differentiable (linear)
# V = (R - Y) / 1.402 + 128.0            --> differentiable (linear)
# .clamp_(0.0, 255.0)                    --> differentiable everywhere except at boundaries
# y00 = Y[..., 0::2, 0::2] etc.          --> indexing, differentiable
# U_sub = box average                    --> differentiable (linear)
```

All operations in `rgb_to_yuv6` are differentiable with respect to the RGB input. The `clamp_` operations have zero gradient at the boundaries but are differentiable elsewhere. This is fine for optimization.

The `F.interpolate` with `mode='bilinear'` in `preprocess_input` is also differentiable.

However, the PoseNet's internal normalization uses `(x - self._mean) / self._std` which is a simple affine transform -- fully differentiable.

**The entire pipeline from RGB pixels through PoseNet output is differentiable.** This is essential for the gradient descent to work.

For SegNet, the forward pass through the UNet is a standard neural network -- fully differentiable with respect to input pixels.

### 7.4 Handling BatchNorm in Eval Mode

Both SegNet and PoseNet contain BatchNorm layers. In eval mode (`.eval()`), BatchNorm uses running statistics and behaves as a fixed affine transform -- fully differentiable. We must ensure both models are in eval mode:

```python
segnet.eval()
posenet.eval()
```

This is critical. In train mode, BatchNorm computes batch statistics which would make the gradient computation incorrect for our single-sample optimization.

### 7.5 The `clamp_` Gradient Issue

When `img_odd` or `img_even` hit the [0, 255] boundary and we `clamp_` in-place, the gradient for those pixels becomes zero (the gradient of clamp at the boundary is 0). This could stall optimization for boundary pixels.

Mitigation: use a soft clamp or simply accept that boundary pixels converge more slowly. In practice, most pixel values are well within [0, 255] and this is not a significant issue. The margin in our SegNet loss provides sufficient buffer.

---

## 8. Task 4: Post-Optimization Verification and Fixup

After gradient descent and uint8 rounding, we must verify that every pixel in every odd frame still produces the correct SegNet argmax. Rounding from float32 to uint8 can flip borderline pixels.

### Verification Loop

```python
def verify_and_fix_segnet(
    optimized_odd,    # (1, 3, 384, 512) uint8 tensor
    target_seg_map,   # (1, 384, 512) int64
    segnet,
    device,
    max_fix_iters=5,
    lr=5.0,
    margin=3.0,       # larger margin for fixup phase
):
    """
    Verify SegNet argmax matches target. If not, run targeted fixup iterations.

    Returns:
        fixed_odd: (1, 3, 384, 512) uint8
        n_mismatched: int -- number of mismatched pixels before fixup
    """
    with torch.no_grad():
        logits = segnet(optimized_odd.float())  # (1, 5, 384, 512)
        pred_map = logits.argmax(dim=1)  # (1, 384, 512)
        mismatch = (pred_map != target_seg_map)  # (1, 384, 512) bool
        n_mismatched = mismatch.sum().item()

    if n_mismatched == 0:
        return optimized_odd, 0

    print(f"  {n_mismatched} mismatched pixels ({n_mismatched / (384*512) * 100:.2f}%), running fixup...")

    # Run targeted optimization with higher margin
    img = optimized_odd.float().clone().detach().requires_grad_(True)
    decoded_ref = optimized_odd.float().clone().detach()
    optimizer = torch.optim.Adam([img], lr=lr)

    for fix_step in range(max_fix_iters):
        optimizer.zero_grad()

        logits = segnet(img)

        # Only penalize mismatched pixels -- use a pixel-weighted margin loss
        target_logit = logits.gather(1, target_seg_map.unsqueeze(1).long())
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, target_seg_map.unsqueeze(1).long(), False)
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9
        runner_up = masked_logits.max(dim=1, keepdim=True).values

        # Weight: focus on mismatched pixels (10x weight)
        pixel_weight = 1.0 + 9.0 * mismatch.unsqueeze(1).float()
        per_pixel_loss = F.relu(margin - (target_logit - runner_up)) * pixel_weight
        loss = per_pixel_loss.mean() + 0.001 * F.mse_loss(img, decoded_ref)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            img.data.clamp_(0.0, 255.0)

        # Check progress
        with torch.no_grad():
            new_logits = segnet(img)
            new_pred = new_logits.argmax(dim=1)
            new_mismatch = (new_pred != target_seg_map)
            new_n = new_mismatch.sum().item()

            if new_n == 0:
                break
            mismatch = new_mismatch  # Update mask for next iteration

    fixed = img.detach().round().clamp(0, 255).to(torch.uint8)

    # Final check
    with torch.no_grad():
        final_logits = segnet(fixed.float())
        final_pred = final_logits.argmax(dim=1)
        final_mismatch = (final_pred != target_seg_map).sum().item()

    return fixed, final_mismatch
```

### The Round-Trip Problem

There is a subtlety: we optimize at 512x384, but the final .raw file is 1164x874. The evaluator then resizes back to 512x384 via bilinear interpolation. The round-trip `512x384 -> bicubic up to 1164x874 -> bilinear down to 512x384` is not the identity transform.

To handle this properly:

```python
def round_trip_simulation(frame_512x384):
    """
    Simulate the evaluator's data path:
    1. Our frame at 512x384
    2. Upscale to 1164x874 via bicubic (what our inflate.py does)
    3. Downscale to 512x384 via bilinear (what the evaluator does)

    Returns the frame as the evaluator's SegNet/PoseNet will see it.
    """
    # (1, 3, 384, 512) -> upscale to (1, 3, 874, 1164)
    up = F.interpolate(frame_512x384, size=(874, 1164), mode='bicubic', align_corners=False)
    up = up.clamp(0, 255).round()  # uint8 simulation

    # (1, 3, 874, 1164) -> downscale to (1, 3, 384, 512) -- what SegNet preprocessing does
    down = F.interpolate(up, size=(384, 512), mode='bilinear', align_corners=False)

    return down
```

The optimization should ideally include this round-trip in the forward pass. However, this adds computational cost (two interpolation operations per iteration). A pragmatic approach:

1. Run the main optimization at 512x384 (fast, 5-10 iterations)
2. After optimization, simulate the round-trip
3. Run SegNet on the round-tripped frame
4. If any pixels flipped, run 2-3 fixup iterations with the round-trip in the loop

```python
def optimize_with_roundtrip_aware_fixup(
    optimized_odd_512,   # (1, 3, 384, 512) uint8 -- from main optimization
    target_seg_map,      # (1, 384, 512) int64
    segnet,
    device,
):
    """
    After main optimization, check if the upscale-downscale round-trip
    preserves SegNet argmax. Fix if not.
    """
    # Simulate round-trip
    frame_float = optimized_odd_512.float()
    round_tripped = round_trip_simulation(frame_float)

    with torch.no_grad():
        logits_rt = segnet(round_tripped)
        pred_rt = logits_rt.argmax(dim=1)
        mismatch_rt = (pred_rt != target_seg_map).sum().item()

    if mismatch_rt == 0:
        return optimized_odd_512, 0

    print(f"  Round-trip caused {mismatch_rt} mismatches, running round-trip-aware fixup...")

    # Optimize with round-trip in the loop
    img = optimized_odd_512.float().clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([img], lr=3.0)

    for step in range(5):
        optimizer.zero_grad()

        # Simulate the full round-trip
        rt = round_trip_simulation(img)

        # SegNet on round-tripped frame
        logits = segnet(rt)
        loss = margin_loss(logits, target_seg_map, margin=3.0)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            img.data.clamp_(0.0, 255.0)

    fixed = img.detach().round().clamp(0, 255).to(torch.uint8)

    # Final verification
    with torch.no_grad():
        rt_final = round_trip_simulation(fixed.float())
        logits_final = segnet(rt_final)
        final_mismatch = (logits_final.argmax(dim=1) != target_seg_map).sum().item()

    return fixed, final_mismatch
```

---

## 9. Task 5: Timing Analysis and Optimization

### Per-Iteration Timing on T4

| Operation | Estimated Time | Notes |
|-----------|---------------|-------|
| SegNet forward (B=1, 384x512, FP16) | ~15-25 ms | EfficientNet-B2 UNet |
| SegNet backward (same) | ~25-40 ms | ~1.5-2x forward |
| PoseNet forward (B=1, 192x256, FP16) | ~3-8 ms | FastViT-T12 |
| PoseNet backward (same) | ~5-12 ms | ~1.5-2x forward |
| rgb_to_yuv6 + interpolation | ~1-2 ms | Simple arithmetic |
| Adam optimizer step | ~1-2 ms | |
| Clamp + overhead | ~1 ms | |
| **Total per iteration** | **~50-90 ms** | |

### Per-Frame-Pair Timing

| Configuration | Iters | Time per pair | 600 pairs total | Margin |
|--------------|-------|---------------|-----------------|--------|
| 5 iterations | 5 | 250-450 ms | 2.5-4.5 min | Huge (25+ min left) |
| 10 iterations | 10 | 500-900 ms | 5-9 min | Very comfortable |
| 15 iterations | 15 | 750-1350 ms | 7.5-13.5 min | Comfortable |
| 20 iterations | 20 | 1.0-1.8 s | 10-18 min | OK |
| 30 iterations | 30 | 1.5-2.7 s | 15-27 min | Tight |

With 10 iterations, total optimization time is approximately 5-9 minutes. Adding:
- Video decoding: ~5-10 seconds
- Model loading: ~5-10 seconds
- Seg map + pose vector loading: ~1 second
- Verification pass: ~2-3 minutes (600 SegNet forward passes)
- Fixup iterations: ~1-2 minutes (only for mismatched frames)
- Upscaling + .raw writing: ~30-60 seconds

**Total estimated decode time: 10-16 minutes** -- well within the 30-minute budget.

### Batching Strategy

If VRAM allows, processing multiple frame pairs simultaneously speeds things up significantly:

```python
# T4 has 16 GB VRAM
# Model sizes (in memory):
#   SegNet (EfficientNet-B2 UNet): ~40 MB parameters + ~100-200 MB activations at B=1
#   PoseNet (FastViT-T12): ~56 MB parameters + ~50-100 MB activations at B=1
#   Both models: ~100 MB params + ~200-300 MB activations at B=1
#
# Per frame pair being optimized:
#   img_even + img_odd: 2 * 3 * 384 * 512 * 4 bytes = 4.7 MB (float32 + gradients ~14 MB)
#   SegNet activations for backward: ~200-400 MB at B=1
#   PoseNet activations for backward: ~100-200 MB at B=1
#
# Conservative estimate per batch element: ~400-600 MB
# Available VRAM (after models): ~15.5 GB
# Max batch size: ~15.5 GB / 500 MB = ~31 pairs
#
# Practical recommendation: batch size 4-8 to leave safety margin

BATCH_SIZE = 4  # Process 4 frame pairs simultaneously
```

With batch_size=4 and 10 iterations:
- 600 / 4 = 150 batches
- Each batch: ~200-400 ms (parallelism helps)
- Total: 30-60 seconds for optimization (dramatic speedup!)

### Mixed Precision (AMP)

Using `torch.cuda.amp` for FP16 forward/backward reduces memory and increases throughput:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast(dtype=torch.float16):
    seg_logits = segnet(img_odd)
    seg_loss = margin_loss(seg_logits, target_seg_map)
    # ... other losses

scaler.scale(total_loss).backward()
scaler.step(optimizer)
scaler.update()
```

T4 has good FP16 performance (65 TFLOPS FP16 vs 8.1 TFLOPS FP32). Mixed precision gives approximately a 2-4x speedup for model inference/backprop.

---

## 10. Task 6: Archive Structure and File Formats

### Archive Layout

```
archive/
  0.mkv              # AV1-encoded video at 512x384, CRF 35-45
  seg_maps.bin        # Compressed segmentation maps (FFV1 or delta+zlib)
  pose.bin            # Compressed pose vectors (float16 + zlib)
  config.json         # Metadata
```

### config.json

```json
{
  "version": 1,
  "resolution": [512, 384],
  "n_frames": 1200,
  "n_pairs": 600,
  "seg_map_encoding": "ffv1",
  "seg_map_resolution": [512, 384],
  "pose_encoding": "float16_zlib",
  "pose_dims": 6,
  "optimization": {
    "n_steps": 10,
    "lr": 2.0,
    "seg_weight": 100.0,
    "pose_weight": 10.0,
    "reg_weight": 0.01,
    "margin": 2.0
  }
}
```

### Size Budget

| Component | Size Estimate | Notes |
|-----------|--------------|-------|
| 0.mkv (CRF 40) | 100-250 KB | Aggressively compressed AV1 |
| seg_maps.bin | 200-600 KB | FFV1 or delta+zlib encoded |
| pose.bin | 4-6 KB | Float16 + zlib |
| config.json | <1 KB | Metadata |
| **Total (pre-zip)** | **300 KB - 860 KB** | |
| **archive.zip** | **300 KB - 800 KB** | zip adds ~2% overhead; already compressed data doesn't shrink much |

### Rate Contribution to Score

```
rate = archive_size / 37,545,489
score_rate = 25 * rate
```

| Archive Size | Rate | Score Contribution |
|-------------|------|-------------------|
| 400 KB | 0.0107 | 0.27 |
| 600 KB | 0.0160 | 0.40 |
| 800 KB | 0.0213 | 0.53 |
| 1.0 MB | 0.0273 | 0.68 |
| 1.2 MB | 0.0327 | 0.82 |

---

## 11. Expected Score Breakdown

### Optimistic Scenario

| Component | Value | Score Contribution | How |
|-----------|-------|-------------------|-----|
| SegNet dist | 0.001 | 100 * 0.001 = **0.10** | Near-perfect after verification + round-trip fixup |
| Rate | 600 KB / 37.5 MB = 0.016 | 25 * 0.016 = **0.40** | CRF 40 video + compact seg maps |
| PoseNet dist | 0.05 | sqrt(10 * 0.05) = sqrt(0.5) = **0.71** | Good init + direct optimization |
| **Total** | | **1.21** | |

### Realistic Scenario

| Component | Value | Score Contribution | How |
|-----------|-------|-------------------|-----|
| SegNet dist | 0.003 | 100 * 0.003 = **0.30** | Most pixels correct, some boundary issues survive round-trip |
| Rate | 800 KB / 37.5 MB = 0.021 | 25 * 0.021 = **0.53** | Slightly larger seg maps |
| PoseNet dist | 0.08 | sqrt(10 * 0.08) = sqrt(0.8) = **0.89** | Good improvement from baseline's 0.38 |
| **Total** | | **1.72** | |

### Conservative Scenario

| Component | Value | Score Contribution | How |
|-----------|-------|-------------------|-----|
| SegNet dist | 0.005 | 100 * 0.005 = **0.50** | Round-trip errors + some fixup failures |
| Rate | 1.2 MB / 37.5 MB = 0.032 | 25 * 0.032 = **0.80** | Larger-than-expected seg maps |
| PoseNet dist | 0.15 | sqrt(10 * 0.15) = sqrt(1.5) = **1.22** | Moderate improvement from baseline |
| **Total** | | **2.52** | |

### Comparison with Competition

- Current leader (PR #17, EthanYangTW): **2.90**
- Baseline: **4.39**
- Our optimistic: **1.21** (would be 1st place by a wide margin)
- Our realistic: **1.72** (still 1st place)
- Our conservative: **2.52** (still 1st place)

---

## 12. Complete Encoder Script

```python
#!/usr/bin/env python
"""
compress.py -- Phase 4 Hybrid Pipeline Encoder

Encodes the original video into a compact archive containing:
1. Aggressively compressed AV1 video
2. Compressed segmentation maps
3. Compressed pose vectors

Usage:
    python compress.py --input videos/0.mkv --output-dir archive/ --crf 40
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import av
import einops
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# Add repo root to path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # Adjust based on actual submission location
sys.path.insert(0, str(ROOT))

from frame_utils import (
    camera_size,
    rgb_to_yuv6,
    segnet_model_input_size,
    yuv420_to_rgb,
)
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path


def load_models(device):
    """Load SegNet and PoseNet models."""
    segnet = SegNet().eval().to(device)
    segnet_sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(segnet_sd)

    posenet = PoseNet().eval().to(device)
    posenet_sd = load_file(str(posenet_sd_path), device=str(device))
    posenet.load_state_dict(posenet_sd)

    return segnet, posenet


def decode_video_frames(video_path):
    """Decode all frames from video, return as list of (H, W, 3) uint8 tensors."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8 tensor
        frames.append(rgb)
    container.close()
    return frames


def extract_segmentation_maps(frames, segnet, device, batch_size=8):
    """
    Run SegNet on all odd frames (indices 1, 3, 5, ...).

    Args:
        frames: list of 1200 (H, W, 3) uint8 tensors
        segnet: loaded SegNet model
        device: torch device
        batch_size: batch size for inference

    Returns:
        np.ndarray of shape (600, 384, 512), dtype uint8, values 0-4
    """
    W_in, H_in = segnet_model_input_size  # (512, 384)
    odd_frames = [frames[i] for i in range(1, len(frames), 2)]  # 600 frames

    seg_maps = []

    for batch_start in range(0, len(odd_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(odd_frames))
        batch_frames = odd_frames[batch_start:batch_end]

        # Stack and preprocess: (B, H, W, 3) -> (B, 3, H, W) -> resize to (B, 3, 384, 512)
        batch_tensor = torch.stack(batch_frames).to(device)  # (B, H, W, 3)
        batch_chw = batch_tensor.permute(0, 3, 1, 2).float()  # (B, 3, H, W)
        batch_resized = F.interpolate(batch_chw, size=(H_in, W_in), mode='bilinear', align_corners=False)
        # (B, 3, 384, 512)

        with torch.no_grad():
            logits = segnet(batch_resized)  # (B, 5, 384, 512)
            seg_map = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            # (B, 384, 512)

        seg_maps.append(seg_map)

    return np.concatenate(seg_maps, axis=0)  # (600, 384, 512)


def extract_pose_vectors(frames, posenet, device, batch_size=8):
    """
    Run PoseNet on all frame pairs (0,1), (2,3), ..., (1198,1199).

    Args:
        frames: list of 1200 (H, W, 3) uint8 tensors
        posenet: loaded PoseNet model
        device: torch device
        batch_size: batch size for inference

    Returns:
        np.ndarray of shape (600, 6), dtype float32
    """
    pairs = [(frames[i], frames[i+1]) for i in range(0, len(frames), 2)]

    pose_vectors = []

    for batch_start in range(0, len(pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        # Stack: (B, 2, H, W, 3)
        even_batch = torch.stack([p[0] for p in batch_pairs]).to(device)  # (B, H, W, 3)
        odd_batch = torch.stack([p[1] for p in batch_pairs]).to(device)   # (B, H, W, 3)

        # (B, 2, H, W, 3) -> (B, 2, 3, H, W)
        pair_tensor = torch.stack([even_batch, odd_batch], dim=1)  # (B, 2, H, W, 3)
        pair_chw = einops.rearrange(pair_tensor, 'b t h w c -> b t c h w').float()
        # (B, 2, 3, H, W)

        # PoseNet preprocessing
        posenet_in = posenet.preprocess_input(pair_chw)  # (B, 12, 192, 256)

        with torch.no_grad():
            out = posenet(posenet_in)  # {'pose': (B, 12)}
            pose_vec = out['pose'][:, :6].cpu().numpy()  # (B, 6)

        pose_vectors.append(pose_vec)

    return np.concatenate(pose_vectors, axis=0)  # (600, 6)


def compress_seg_maps(seg_maps):
    """
    Compress segmentation maps using delta coding + zlib.

    Args:
        seg_maps: np.ndarray of shape (600, 384, 512), dtype uint8

    Returns:
        bytes
    """
    n_frames, h, w = seg_maps.shape

    # Header: n_frames, h, w
    header = struct.pack('<III', n_frames, h, w)

    # Delta encode: first frame raw, subsequent frames as XOR with previous
    delta_frames = [seg_maps[0].tobytes()]
    for i in range(1, n_frames):
        delta = np.bitwise_xor(seg_maps[i], seg_maps[i-1])
        delta_frames.append(delta.tobytes())

    # Concatenate all frames
    all_data = header + b''.join(delta_frames)

    # Compress with zlib level 9
    compressed = zlib.compress(all_data, level=9)

    return compressed


def decompress_seg_maps(data):
    """
    Decompress segmentation maps.

    Returns:
        np.ndarray of shape (n_frames, h, w), dtype uint8
    """
    decompressed = zlib.decompress(data)

    # Parse header
    n_frames, h, w = struct.unpack('<III', decompressed[:12])

    frame_size = h * w
    offset = 12

    # First frame
    frames = [np.frombuffer(decompressed[offset:offset+frame_size], dtype=np.uint8).reshape(h, w)]
    offset += frame_size

    # Delta-decode subsequent frames
    for i in range(1, n_frames):
        delta = np.frombuffer(decompressed[offset:offset+frame_size], dtype=np.uint8).reshape(h, w)
        frame = np.bitwise_xor(frames[-1], delta)
        frames.append(frame)
        offset += frame_size

    return np.stack(frames)  # (n_frames, h, w)


def compress_pose_vectors(pose_vectors):
    """
    Compress pose vectors using float16 + zlib.

    Args:
        pose_vectors: np.ndarray of shape (600, 6), dtype float32

    Returns:
        bytes
    """
    n_pairs, n_dims = pose_vectors.shape
    header = struct.pack('<II', n_pairs, n_dims)

    fp16 = pose_vectors.astype(np.float16)
    data = header + fp16.tobytes()

    return zlib.compress(data, level=9)


def decompress_pose_vectors(data):
    """
    Decompress pose vectors.

    Returns:
        np.ndarray of shape (n_pairs, n_dims), dtype float32
    """
    decompressed = zlib.decompress(data)
    n_pairs, n_dims = struct.unpack('<II', decompressed[:8])
    fp16 = np.frombuffer(decompressed[8:], dtype=np.float16).reshape(n_pairs, n_dims)
    return fp16.astype(np.float32)


def encode_video(input_path, output_path, crf=40, preset=2):
    """Encode video at 512x384 using SVT-AV1."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-i', str(input_path),
        '-vf', 'scale=512:384:flags=lanczos',
        '-c:v', 'libsvtav1',
        '-crf', str(crf),
        '-preset', str(preset),
        '-svtav1-params', 'keyint=120:irefresh-type=2:film-grain=0',
        '-pix_fmt', 'yuv420p',
        str(output_path)
    ]

    # Fallback to libx265 if SVT-AV1 not available
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("SVT-AV1 not available, falling back to libx265...")
        cmd = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-i', str(input_path),
            '-vf', 'scale=512:384:flags=lanczos',
            '-c:v', 'libx265',
            '-crf', str(crf),
            '-preset', 'slow',
            '-x265-params', 'keyint=120:min-keyint=120:scenecut=0',
            '-pix_fmt', 'yuv420p',
            str(output_path)
        ]
        subprocess.run(cmd, check=True)

    size = os.path.getsize(output_path)
    print(f"Encoded video: {size:,} bytes ({size/1024:.1f} KB)")
    return size


def main():
    parser = argparse.ArgumentParser(description='Phase 4 Hybrid Encoder')
    parser.add_argument('--input', type=Path, default=Path('videos/0.mkv'))
    parser.add_argument('--output-dir', type=Path, default=Path('archive'))
    parser.add_argument('--crf', type=int, default=40)
    parser.add_argument('--preset', type=int, default=2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load models
    print("Loading models...")
    segnet, posenet = load_models(device)

    # Step 2: Decode original video
    print("Decoding original video...")
    frames = decode_video_frames(args.input)
    print(f"  Decoded {len(frames)} frames, shape: {frames[0].shape}")

    # Step 3: Extract segmentation maps
    print("Extracting segmentation maps...")
    seg_maps = extract_segmentation_maps(frames, segnet, device, args.batch_size)
    print(f"  Extracted {seg_maps.shape[0]} seg maps, shape: {seg_maps.shape[1:]}")
    print(f"  Unique classes: {np.unique(seg_maps)}")

    # Step 4: Extract pose vectors
    print("Extracting pose vectors...")
    pose_vectors = extract_pose_vectors(frames, posenet, device, args.batch_size)
    print(f"  Extracted {pose_vectors.shape[0]} pose vectors, shape: {pose_vectors.shape[1:]}")
    print(f"  Pose range: [{pose_vectors.min():.4f}, {pose_vectors.max():.4f}]")

    # Step 5: Encode video
    print(f"Encoding video at CRF {args.crf}...")
    video_path = args.output_dir / '0.mkv'
    video_size = encode_video(args.input, video_path, args.crf, args.preset)

    # Step 6: Compress and save seg maps
    print("Compressing segmentation maps...")
    seg_data = compress_seg_maps(seg_maps)
    seg_path = args.output_dir / 'seg_maps.bin'
    seg_path.write_bytes(seg_data)
    print(f"  Compressed seg maps: {len(seg_data):,} bytes ({len(seg_data)/1024:.1f} KB)")

    # Step 7: Compress and save pose vectors
    print("Compressing pose vectors...")
    pose_data = compress_pose_vectors(pose_vectors)
    pose_path = args.output_dir / 'pose.bin'
    pose_path.write_bytes(pose_data)
    print(f"  Compressed pose vectors: {len(pose_data):,} bytes ({len(pose_data)/1024:.1f} KB)")

    # Step 8: Save config
    config = {
        'version': 1,
        'resolution': [512, 384],
        'n_frames': len(frames),
        'n_pairs': len(frames) // 2,
        'seg_map_encoding': 'delta_xor_zlib',
        'seg_map_resolution': [512, 384],
        'pose_encoding': 'float16_zlib',
        'pose_dims': 6,
        'crf': args.crf,
        'optimization': {
            'n_steps': 10,
            'lr': 2.0,
            'seg_weight': 100.0,
            'pose_weight': 10.0,
            'reg_weight': 0.01,
            'margin': 2.0
        }
    }
    config_path = args.output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Step 9: Create archive.zip
    print("Creating archive.zip...")
    import zipfile
    zip_path = args.output_dir.parent / 'archive.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in [video_path, seg_path, pose_path, config_path]:
            zf.write(file, file.relative_to(args.output_dir))

    total_size = os.path.getsize(zip_path)
    rate = total_size / 37_545_489
    print(f"\nArchive created: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    print(f"Rate: {rate:.6f}")
    print(f"Rate score contribution: 25 * {rate:.6f} = {25 * rate:.2f}")


if __name__ == '__main__':
    main()
```

---

## 13. Complete Decoder Script

```python
#!/usr/bin/env python
"""
inflate.py -- Phase 4 Hybrid Pipeline Decoder

Decodes the compressed archive and runs adversarial optimization
to produce frames that match stored SegNet/PoseNet targets.

Usage (called by inflate.sh):
    python -m submissions.phase4_hybrid.inflate <archive_dir>/0.mkv <output_dir>/0.raw

The script expects the archive directory to contain:
    0.mkv          -- compressed video
    seg_maps.bin   -- compressed segmentation maps
    pose.bin       -- compressed pose vectors
    config.json    -- metadata
"""

import json
import os
import struct
import sys
import zlib
from pathlib import Path

import av
import einops
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from frame_utils import (
    camera_size,
    rgb_to_yuv6,
    segnet_model_input_size,
    yuv420_to_rgb,
)
from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path


# ============================================================
# Data loading utilities
# ============================================================

def decompress_seg_maps(data):
    """Decompress delta-XOR-zlib encoded segmentation maps."""
    decompressed = zlib.decompress(data)
    n_frames, h, w = struct.unpack('<III', decompressed[:12])
    frame_size = h * w
    offset = 12

    frames = [np.frombuffer(decompressed[offset:offset+frame_size], dtype=np.uint8).reshape(h, w).copy()]
    offset += frame_size

    for i in range(1, n_frames):
        delta = np.frombuffer(decompressed[offset:offset+frame_size], dtype=np.uint8).reshape(h, w)
        frame = np.bitwise_xor(frames[-1], delta)
        frames.append(frame)
        offset += frame_size

    return np.stack(frames)


def decompress_pose_vectors(data):
    """Decompress float16-zlib encoded pose vectors."""
    decompressed = zlib.decompress(data)
    n_pairs, n_dims = struct.unpack('<II', decompressed[:8])
    fp16 = np.frombuffer(decompressed[8:], dtype=np.float16).reshape(n_pairs, n_dims)
    return fp16.astype(np.float32)


def decode_compressed_video(video_path):
    """Decode compressed video frames via PyAV."""
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8 tensor
        frames.append(rgb)
    container.close()
    return frames


# ============================================================
# Loss functions
# ============================================================

def margin_loss(logits, target_map, margin=2.0):
    """
    SegNet margin loss: penalize pixels where target class logit
    does not exceed runner-up by at least `margin`.

    Args:
        logits: (B, 5, H, W) float -- SegNet raw logits
        target_map: (B, H, W) long -- target class indices (0-4)
        margin: desired logit gap

    Returns:
        scalar loss
    """
    target_logit = logits.gather(1, target_map.unsqueeze(1))  # (B, 1, H, W)

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, target_map.unsqueeze(1), False)
    masked_logits = logits.masked_fill(~mask, -1e9)
    runner_up = masked_logits.max(dim=1, keepdim=True).values  # (B, 1, H, W)

    loss = F.relu(margin - (target_logit - runner_up)).mean()
    return loss


def compute_posenet_loss(posenet, img_even, img_odd, target_pose):
    """
    PoseNet MSE loss between predicted and target pose vectors.

    Args:
        posenet: loaded PoseNet (eval mode)
        img_even: (B, 3, 384, 512) float -- even frames
        img_odd: (B, 3, 384, 512) float -- odd frames
        target_pose: (B, 6) float -- target pose vectors

    Returns:
        scalar loss
    """
    B = img_even.shape[0]
    # Stack as (B, 2, 3, 384, 512)
    pair = torch.stack([img_even, img_odd], dim=1)

    # PoseNet.preprocess_input expects (B, T, C, H, W)
    posenet_in = posenet.preprocess_input(pair)  # (B, 12, 192, 256)

    out = posenet(posenet_in)  # {'pose': (B, 12)}
    pred_pose = out['pose'][:, :6]  # (B, 6)

    loss = F.mse_loss(pred_pose, target_pose)
    return loss


# ============================================================
# Optimization loop
# ============================================================

def optimize_batch(
    decoded_even,      # (B, 3, 384, 512) float32
    decoded_odd,       # (B, 3, 384, 512) float32
    target_seg_maps,   # (B, 384, 512) int64
    target_poses,      # (B, 6) float32
    segnet,
    posenet,
    config,
):
    """
    Optimize a batch of frame pairs.

    Returns:
        optimized_even: (B, 3, 384, 512) uint8
        optimized_odd: (B, 3, 384, 512) uint8
    """
    opt_cfg = config['optimization']
    n_steps = opt_cfg['n_steps']
    lr = opt_cfg['lr']
    seg_weight = opt_cfg['seg_weight']
    pose_weight = opt_cfg['pose_weight']
    reg_weight = opt_cfg['reg_weight']
    seg_margin = opt_cfg['margin']

    # Clone as optimization variables
    img_odd = decoded_odd.clone().detach().requires_grad_(True)
    img_even = decoded_even.clone().detach().requires_grad_(True)

    # Reference frames for regularization
    ref_odd = decoded_odd.clone().detach()
    ref_even = decoded_even.clone().detach()

    optimizer = torch.optim.Adam([img_odd, img_even], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.1
    )

    use_amp = img_odd.device.type == 'cuda'

    for step in range(n_steps):
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                seg_logits = segnet(img_odd)
                seg_loss = margin_loss(seg_logits, target_seg_maps, margin=seg_margin)

                pose_loss = compute_posenet_loss(posenet, img_even, img_odd, target_poses)

                reg_loss = F.mse_loss(img_odd, ref_odd) + F.mse_loss(img_even, ref_even)

                total_loss = seg_weight * seg_loss + pose_weight * pose_loss + reg_weight * reg_loss
        else:
            seg_logits = segnet(img_odd)
            seg_loss = margin_loss(seg_logits, target_seg_maps, margin=seg_margin)

            pose_loss = compute_posenet_loss(posenet, img_even, img_odd, target_poses)

            reg_loss = F.mse_loss(img_odd, ref_odd) + F.mse_loss(img_even, ref_even)

            total_loss = seg_weight * seg_loss + pose_weight * pose_loss + reg_weight * reg_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            img_odd.data.clamp_(0.0, 255.0)
            img_even.data.clamp_(0.0, 255.0)

        scheduler.step()

    # Round to uint8
    opt_odd = img_odd.detach().round().clamp(0, 255).to(torch.uint8)
    opt_even = img_even.detach().round().clamp(0, 255).to(torch.uint8)

    return opt_even, opt_odd


def verify_and_fix_batch(opt_odd, target_seg_maps, segnet, max_fix_iters=3, margin=3.0):
    """
    Verify SegNet argmax and fix mismatched pixels.

    Args:
        opt_odd: (B, 3, 384, 512) uint8
        target_seg_maps: (B, 384, 512) int64

    Returns:
        fixed_odd: (B, 3, 384, 512) uint8
        total_mismatches: int
    """
    with torch.no_grad():
        logits = segnet(opt_odd.float())
        pred = logits.argmax(dim=1)
        mismatch_mask = (pred != target_seg_maps)
        total_mismatches = mismatch_mask.sum().item()

    if total_mismatches == 0:
        return opt_odd, 0

    # Run fixup
    img = opt_odd.float().clone().detach().requires_grad_(True)
    ref = opt_odd.float().clone().detach()
    optimizer = torch.optim.Adam([img], lr=5.0)

    for fix_step in range(max_fix_iters):
        optimizer.zero_grad()

        logits = segnet(img)
        loss = margin_loss(logits, target_seg_maps, margin=margin)
        reg = 0.001 * F.mse_loss(img, ref)
        (loss + reg).backward()
        optimizer.step()

        with torch.no_grad():
            img.data.clamp_(0.0, 255.0)
            new_pred = segnet(img).argmax(dim=1)
            new_mismatches = (new_pred != target_seg_maps).sum().item()
            if new_mismatches == 0:
                break

    fixed = img.detach().round().clamp(0, 255).to(torch.uint8)

    with torch.no_grad():
        final_pred = segnet(fixed.float()).argmax(dim=1)
        final_mismatches = (final_pred != target_seg_maps).sum().item()

    return fixed, final_mismatches


# ============================================================
# Main decode pipeline
# ============================================================

def main(src_path, dst_path):
    """
    Main decode pipeline.

    Args:
        src_path: path to compressed video (archive_dir/0.mkv)
        dst_path: path to output .raw file (inflated_dir/0.raw)
    """
    archive_dir = Path(src_path).parent

    # Determine device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Phase 4 Hybrid Decoder -- device: {device}")

    # --------------------------------------------------------
    # Step 1: Load config
    # --------------------------------------------------------
    config_path = archive_dir / 'config.json'
    with open(config_path) as f:
        config = json.load(f)

    W_work, H_work = config['resolution']  # [512, 384]
    n_frames = config['n_frames']           # 1200
    n_pairs = config['n_pairs']             # 600

    print(f"Config: {n_frames} frames, {n_pairs} pairs, working resolution {W_work}x{H_work}")

    # --------------------------------------------------------
    # Step 2: Load models
    # --------------------------------------------------------
    print("Loading models...")
    segnet = SegNet().eval().to(device)
    segnet_sd = load_file(str(segnet_sd_path), device=str(device))
    segnet.load_state_dict(segnet_sd)

    posenet = PoseNet().eval().to(device)
    posenet_sd = load_file(str(posenet_sd_path), device=str(device))
    posenet.load_state_dict(posenet_sd)

    # Freeze models (no gradients for model parameters)
    for param in segnet.parameters():
        param.requires_grad_(False)
    for param in posenet.parameters():
        param.requires_grad_(False)

    # --------------------------------------------------------
    # Step 3: Decode compressed video
    # --------------------------------------------------------
    print("Decoding compressed video...")
    video_path = archive_dir / '0.mkv'
    decoded_frames = decode_compressed_video(video_path)
    print(f"  Decoded {len(decoded_frames)} frames, shape: {decoded_frames[0].shape}")

    # Resize decoded frames to working resolution if needed
    resized_frames = []
    for frame in decoded_frames:
        # frame: (H_enc, W_enc, 3) uint8 tensor
        frame_chw = frame.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
        if frame_chw.shape[2] != H_work or frame_chw.shape[3] != W_work:
            frame_chw = F.interpolate(frame_chw, size=(H_work, W_work), mode='bilinear', align_corners=False)
        resized_frames.append(frame_chw.squeeze(0))  # (3, 384, 512)

    print(f"  Resized to {W_work}x{H_work}")

    # --------------------------------------------------------
    # Step 4: Load seg maps and pose vectors
    # --------------------------------------------------------
    print("Loading segmentation maps...")
    seg_data = (archive_dir / 'seg_maps.bin').read_bytes()
    seg_maps = decompress_seg_maps(seg_data)
    seg_maps_tensor = torch.from_numpy(seg_maps).long().to(device)
    print(f"  Loaded {seg_maps.shape[0]} seg maps, shape: {seg_maps.shape[1:]}")

    print("Loading pose vectors...")
    pose_data = (archive_dir / 'pose.bin').read_bytes()
    pose_vectors = decompress_pose_vectors(pose_data)
    pose_vectors_tensor = torch.from_numpy(pose_vectors).float().to(device)
    print(f"  Loaded {pose_vectors.shape[0]} pose vectors, shape: {pose_vectors.shape[1:]}")

    # --------------------------------------------------------
    # Step 5: Gradient descent optimization
    # --------------------------------------------------------
    print("Running adversarial optimization...")

    BATCH_SIZE = 4  # Adjust based on VRAM
    if device.type != 'cuda':
        BATCH_SIZE = 1  # CPU is slow, no batching benefit

    optimized_frames = [None] * n_frames  # Will hold (3, 384, 512) uint8 tensors

    total_seg_mismatches = 0

    for batch_start in range(0, n_pairs, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_pairs)
        B = batch_end - batch_start

        # Gather batch data
        even_indices = [2 * (batch_start + i) for i in range(B)]
        odd_indices = [2 * (batch_start + i) + 1 for i in range(B)]

        decoded_even = torch.stack([resized_frames[idx] for idx in even_indices]).to(device)
        decoded_odd = torch.stack([resized_frames[idx] for idx in odd_indices]).to(device)
        # (B, 3, 384, 512)

        target_segs = seg_maps_tensor[batch_start:batch_end]  # (B, 384, 512)
        target_poses = pose_vectors_tensor[batch_start:batch_end]  # (B, 6)

        # Optimize
        opt_even, opt_odd = optimize_batch(
            decoded_even, decoded_odd,
            target_segs, target_poses,
            segnet, posenet, config
        )

        # Verify and fix SegNet
        opt_odd, mismatches = verify_and_fix_batch(
            opt_odd, target_segs, segnet
        )
        total_seg_mismatches += mismatches

        # Store results
        for i in range(B):
            optimized_frames[even_indices[i]] = opt_even[i].cpu()  # (3, 384, 512)
            optimized_frames[odd_indices[i]] = opt_odd[i].cpu()

        if (batch_start // BATCH_SIZE) % 50 == 0:
            pair_idx = batch_start
            print(f"  Processed pairs {pair_idx}-{batch_end-1}/{n_pairs}"
                  f" (seg mismatches so far: {total_seg_mismatches})")

    print(f"  Optimization complete. Total seg mismatches: {total_seg_mismatches}")

    # --------------------------------------------------------
    # Step 6: Upscale to camera resolution and write .raw
    # --------------------------------------------------------
    W_out, H_out = camera_size  # (1164, 874)
    print(f"Upscaling to {W_out}x{H_out} and writing .raw file...")

    frame_bytes = H_out * W_out * 3

    with open(dst_path, 'wb') as f:
        for frame_idx in range(n_frames):
            frame = optimized_frames[frame_idx]  # (3, 384, 512) uint8

            # Upscale: (1, 3, 384, 512) -> (1, 3, 874, 1164)
            frame_float = frame.unsqueeze(0).float()
            upscaled = F.interpolate(
                frame_float,
                size=(H_out, W_out),
                mode='bicubic',
                align_corners=False
            )
            upscaled = upscaled.clamp(0, 255).round().to(torch.uint8)

            # (1, 3, 874, 1164) -> (874, 1164, 3)
            frame_hwc = upscaled.squeeze(0).permute(1, 2, 0).contiguous()

            assert frame_hwc.shape == (H_out, W_out, 3), f"Bad shape: {frame_hwc.shape}"

            f.write(frame_hwc.numpy().tobytes())

    output_size = os.path.getsize(dst_path)
    expected_size = n_frames * frame_bytes
    assert output_size == expected_size, f"Output size {output_size} != expected {expected_size}"

    print(f"  Wrote {dst_path}: {output_size:,} bytes ({n_frames} frames)")
    print("Done!")


if __name__ == '__main__':
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    main(src_path, dst_path)
```

---

## 14. Shell Scripts

### compress.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Hybrid Pipeline -- Compression Script
# Usage: bash compress.sh [input_video] [output_dir]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

INPUT="${1:-$ROOT/videos/0.mkv}"
OUTPUT_DIR="${2:-$HERE/archive}"

echo "=== Phase 4 Hybrid Pipeline Encoder ==="
echo "Input: $INPUT"
echo "Output: $OUTPUT_DIR"

cd "$ROOT"

# Run the Python encoder
python -m "submissions.phase4_hybrid.compress" \
    --input "$INPUT" \
    --output-dir "$OUTPUT_DIR" \
    --crf 40 \
    --preset 2

echo "=== Compression complete ==="
ls -lh "$OUTPUT_DIR"/*
ls -lh "$HERE/archive.zip"
```

### inflate.sh

```bash
#!/usr/bin/env bash
set -euo pipefail

# Phase 4 Hybrid Pipeline -- Inflation Script
# Called by evaluate.sh as: bash inflate.sh <archive_dir> <inflated_dir> <video_names_file>

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

cd "$ROOT"

while IFS= read -r VIDEO_NAME || [[ -n "$VIDEO_NAME" ]]; do
    BASENAME="${VIDEO_NAME%.*}"
    SRC="$DATA_DIR/${BASENAME}.mkv"
    DST="$OUTPUT_DIR/${BASENAME}.raw"

    echo "Inflating: $SRC -> $DST"
    python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"
done < "$FILE_LIST"

echo "=== Inflation complete ==="
ls -lh "$OUTPUT_DIR"/*.raw
```

---

## 15. Risk Analysis and Mitigations

### Risk 1: Gradient Descent Does Not Converge

**Probability:** Low (when starting from decoded video)
**Impact:** High (SegNet distortion would remain elevated)
**Mitigation:**
- The decoded video already gets 80-90% of pixels correct, so convergence only needs to fix 10-20%
- The margin loss is well-conditioned and has clear gradients at boundary pixels
- Use Adam optimizer which handles per-parameter learning rates well
- If convergence is slow, increase the number of iterations (we have ample time budget)
- Fallback: even without full convergence, the result will still be much better than codec-only

### Risk 2: VRAM Exhaustion on T4

**Probability:** Medium
**Impact:** High (process crashes)
**Analysis:**
- T4 has 16 GB VRAM
- SegNet model: ~40 MB
- PoseNet model: ~56 MB
- Per frame pair (B=1): ~400-600 MB for activations + gradients
- At B=4: ~1.6-2.4 GB for activations
- Total: ~2-3 GB at B=4
- Safety margin: ~13 GB free
**Mitigation:**
- Start with B=4, reduce to B=2 or B=1 if OOM
- Use `torch.cuda.amp` (FP16) to halve activation memory
- Clear gradient computation graph between batches via `torch.cuda.empty_cache()`
- Monitor with `torch.cuda.max_memory_allocated()`

### Risk 3: uint8 Rounding Flips SegNet Pixels

**Probability:** Medium (expected for some boundary pixels)
**Impact:** Low (verification + fixup loop handles this)
**Mitigation:**
- The margin loss with margin=2.0 provides buffer (the logit gap must be at least 2.0)
- Post-optimization verification catches all mismatches
- Targeted fixup iterations with higher margin (3.0) and focused pixel weighting
- After fixup, a final verification confirms zero mismatches

### Risk 4: Upscale/Downscale Round-Trip Breaks SegNet

**Probability:** Medium
**Impact:** Medium (could add 0.1-0.3 to SegNet term)
**Mitigation:**
- Round-trip simulation in the verification step
- Round-trip-aware fixup iterations if needed
- Alternative: optimize directly at 1164x874 (much more expensive but eliminates the issue)
- Alternative: use the same interpolation method for upscale as the evaluator uses for downscale

### Risk 5: PoseNet Optimization Conflicts with SegNet

**Probability:** Low
**Impact:** Low (PoseNet weight can be reduced)
**Analysis:**
- PoseNet sees both frames at (192, 256) after YUV6 conversion
- SegNet sees only the odd frame at (384, 512) in RGB
- The two networks operate at different resolutions and color spaces
- Small pixel changes to satisfy PoseNet are unlikely to flip SegNet argmax at boundary pixels
**Mitigation:**
- Weight SegNet loss much higher than PoseNet (100:10 ratio)
- Run SegNet verification after joint optimization
- If conflicts arise, reduce pose_weight or separate optimization: first SegNet-only on odd frames, then PoseNet-only on both

### Risk 6: 30-Minute Time Budget

**Probability:** Very low
**Impact:** High (evaluation fails)
**Analysis from Section 9:**
- 10 iterations at B=4: ~1-2 minutes for optimization
- Full pipeline including model loading, decoding, verification, upscaling: ~10-16 minutes
- Safety margin: 14-20 minutes
**Mitigation:**
- Implement a timer that reduces iterations if running behind schedule
- Process the most important frames first (those with highest SegNet mismatch)
- The pipeline can produce a valid .raw file even if optimization is cut short (just use decoded frames for remaining pairs)

---

## 16. Hyperparameter Tuning Guide

### Key Hyperparameters and Their Effects

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| CRF (encoding) | 30-50 | 40 | Lower = larger file + better init; Higher = smaller file + worse init |
| n_steps | 3-30 | 10 | More steps = better distortion but more time |
| lr | 0.5-10.0 | 2.0 | Higher = faster convergence but may overshoot |
| seg_weight | 10-500 | 100.0 | Higher = prioritize SegNet over PoseNet |
| pose_weight | 1-50 | 10.0 | Higher = better PoseNet score but may hurt SegNet |
| reg_weight | 0.001-0.1 | 0.01 | Higher = stay closer to decoded frame; too high prevents convergence |
| margin | 1.0-5.0 | 2.0 | Higher = more robust to rounding but harder to achieve |
| BATCH_SIZE | 1-8 | 4 | Higher = faster but more VRAM |

### Tuning Strategy

1. **Fix CRF first:** Try CRF 35, 40, 45. Measure archive size and number of initially-mismatched SegNet pixels.
2. **Tune n_steps:** Start with 5, increase until SegNet mismatch reaches zero (or close). With CRF 40, expect 5-10 steps.
3. **Tune lr:** If convergence is slow, increase lr. If overshooting (loss oscillates), decrease.
4. **Tune weights:** The 100:10 seg/pose ratio is a good starting point. If PoseNet score is too high, increase pose_weight. If SegNet score is too high, increase seg_weight.
5. **Tune margin:** If uint8 rounding frequently breaks SegNet, increase margin to 3.0 or 4.0.

### Tradeoff: CRF vs Optimization Steps

There is a direct tradeoff between CRF aggressiveness and the number of optimization steps needed:

| CRF | ~Archive Size | ~Initial SegNet Mismatch | Steps Needed | Net Score Impact |
|-----|---------------|--------------------------|--------------|-----------------|
| 30 | 500 KB | 1-2% | 3-5 | Rate: 0.33, less optimization time |
| 35 | 300 KB | 3-5% | 5-8 | Rate: 0.20, moderate optimization |
| 40 | 150 KB | 5-10% | 8-12 | Rate: 0.10, good sweet spot |
| 45 | 80 KB | 10-20% | 12-20 | Rate: 0.05, more optimization needed |
| 50 | 50 KB | 20-40% | 20-30 | Rate: 0.03, risky convergence |

The sweet spot depends on seg map size. If seg maps are 500 KB, the video being 150 KB vs 300 KB saves only (300-150)/37545 * 25 = 0.10 points, which is small. The dominant size component is the seg maps.

### Optimal Budget Allocation

Given the scoring formula sensitivities:
- Each 100 KB of archive costs 25 * (100000/37545489) = **0.066 points**
- Each 0.001 SegNet distortion costs **0.10 points**
- Reducing PoseNet from 0.10 to 0.05 saves only **sqrt(0.5) - sqrt(1.0) = -0.29 points**

Therefore: prioritize SegNet distortion (aim for zero), then minimize archive size, then optimize PoseNet.

---

## 17. Implementation Checklist

### Encoder Side (compress.py + compress.sh)

- [ ] Implement video encoding with SVT-AV1 at configurable CRF
- [ ] Implement fallback to libx265 if SVT-AV1 unavailable
- [ ] Implement SegNet map extraction (batch processing for speed)
- [ ] Implement PoseNet vector extraction (batch processing for speed)
- [ ] Implement seg map compression (delta XOR + zlib)
- [ ] Implement pose vector compression (float16 + zlib)
- [ ] Implement archive creation (zip all components)
- [ ] CRF sweep: test 30, 35, 40, 45 and measure sizes + initial mismatches
- [ ] Validate round-trip: compressed -> decoded -> SegNet == original SegNet output?
- [ ] Measure total archive size at each CRF level

### Decoder Side (inflate.py + inflate.sh)

- [ ] Implement video decoding via PyAV
- [ ] Implement seg map decompression
- [ ] Implement pose vector decompression
- [ ] Implement model loading (SegNet + PoseNet)
- [ ] Implement margin loss function
- [ ] Implement PoseNet loss function
- [ ] Implement the optimization loop with Adam + cosine LR
- [ ] Implement SegNet verification + fixup
- [ ] Implement round-trip-aware verification
- [ ] Implement bicubic upscaling to 1164x874
- [ ] Implement .raw file writing
- [ ] Test on local machine (CPU mode)
- [ ] Test on GPU (if available)
- [ ] Time the full pipeline and verify it fits in 30 minutes
- [ ] Measure final SegNet distortion, PoseNet distortion, and rate

### Integration Testing

- [ ] Run full encoder -> archive.zip -> inflate.sh -> evaluate.py pipeline
- [ ] Verify .raw file size matches expected (3,661,113,600 bytes)
- [ ] Verify SegNet distortion is near zero
- [ ] Verify PoseNet distortion is improved from baseline
- [ ] Verify rate is competitive
- [ ] Calculate total score and compare with targets

---

## 18. Alternative Approaches to Explore

### 18.1 Even Frames: Copy from Odd Frame

Since even frames only affect PoseNet (SegNet ignores them), one option is to make even frames identical to odd frames (or a simple transform of them). This eliminates half the optimization work. PoseNet sees both frames, so identical frames would give a specific (likely non-zero) pose output. We would then store pose vectors assuming this strategy. Risk: PoseNet distortion might be worse because the original even/odd frames are genuinely different.

### 18.2 SegNet Map at Reduced Resolution

If SegNet boundaries are smooth enough, storing maps at 256x192 (half resolution) and using nearest-neighbor upsampling could work. This would halve the seg map size. The optimization loop would then need to fix any boundary errors from the upsampling. Worth testing.

### 18.3 Skip PoseNet Optimization

If PoseNet distortion from the decoded video alone is acceptable (say, 0.15 vs baseline's 0.38), we could skip PoseNet optimization entirely. The score savings from PoseNet optimization are modest: sqrt(10*0.15) - sqrt(10*0.05) = 1.22 - 0.71 = 0.51 points. If this saves enough time to allow more SegNet iterations, it could be a net win.

### 18.4 Learned Initialization Instead of Codec

Instead of a traditional codec, use a tiny neural network (e.g., a small MLP per frame, or a shared convolutional network with per-frame latent codes) as the initialization. The network weights + latent codes could potentially be smaller than a compressed video. This blurs the line between Phase 3 (INR) and Phase 4 (hybrid), and is worth exploring if the codec approach hits a size floor.

### 18.5 Class-Conditioned Initialization

Instead of starting from the decoded video, start from a "class-color" image where each pixel is set to an ideal RGB value for its target class. This requires only the seg map (no compressed video). Combined with PoseNet optimization, this could be viable. The risk is that PoseNet optimization from a flat-colored image takes many more iterations. But if we have the seg map anyway, and if the class colors are chosen to minimize PoseNet distortion globally, it could work.

### 18.6 Storing SegNet Logits Instead of ArgMax Maps

Instead of storing the argmax class (3 bits), store quantized logit differences or the top-2 class probabilities. This gives the optimization loop information about how confident the segmentation is at each pixel, allowing it to prioritize uncertain pixels. However, this significantly increases storage requirements and may not be worth the complexity.

---

## 19. Key Code References

| What | File | Lines | Notes |
|------|------|-------|-------|
| Scoring formula | `evaluate.py` | 92 | `100*segnet + sqrt(10*posenet) + 25*rate` |
| SegNet architecture | `modules.py` | 103-113 | `smp.Unet('tu-efficientnet_b2', classes=5)` |
| SegNet preprocessing | `modules.py` | 107-109 | Takes last frame, resize to 512x384 |
| SegNet distortion | `modules.py` | 111-113 | argmax disagreement fraction |
| PoseNet architecture | `modules.py` | 61-80 | FastViT-T12 + Hydra head |
| PoseNet preprocessing | `modules.py` | 70-74 | Both frames, resize, RGB->YUV6, stack 12ch |
| PoseNet distortion | `modules.py` | 82-84 | MSE on first 6 of 12 output dims |
| RGB to YUV6 | `frame_utils.py` | 51-78 | BT.601, 4:2:0 subsampling |
| YUV420 to RGB | `frame_utils.py` | 159-183 | BT.601 limited range, bilinear chroma |
| Camera size | `frame_utils.py` | 11 | (1164, 874) |
| SegNet input size | `frame_utils.py` | 13 | (512, 384) |
| Seq len | `frame_utils.py` | 10 | 2 |
| SegNet weights | `modules.py` | 17 | `models/segnet.safetensors` (38 MB) |
| PoseNet weights | `modules.py` | 18 | `models/posenet.safetensors` (56 MB) |
| .raw format | `frame_utils.py` | 218-253 | Flat uint8 (N, H, W, 3), no header |
| Original video size | `evaluate.py` | 64 | 37,545,489 bytes |
| Baseline inflate | `submissions/baseline_fast/inflate.py` | all | Decode + bicubic upscale |
| Baseline inflate.sh | `submissions/baseline_fast/inflate.sh` | all | Shell wrapper pattern |

---

## 20. Summary

The Phase 4 Hybrid Pipeline combines three strategies for maximum effect:

1. **Aggressive codec compression** (CRF 40 AV1) provides a structurally-close initialization for ~150 KB
2. **Stored segmentation maps** (delta-coded + zlib) provide exact SegNet targets for ~400 KB
3. **Stored pose vectors** (float16 + zlib) provide exact PoseNet targets for ~5 KB
4. **Gradient descent at decode time** (5-10 iterations per frame pair) bridges the gap from degraded decoded frames to near-perfect evaluation network outputs

The total archive is expected to be 400 KB - 800 KB, giving a rate score of 0.27-0.53. SegNet distortion should approach zero (score contribution 0.1-0.3). PoseNet distortion should be significantly reduced from the baseline (score contribution 0.7-1.2).

**Expected total score: 1.2-2.3**, beating the current leader at 2.90 by a significant margin.

The implementation is straightforward: one Python encoder script, one Python decoder script, two shell wrappers. All required libraries (torch, av, safetensors, einops, timm, segmentation-models-pytorch) are already in the repo's dependency list. The decode time budget of 30 minutes on a T4 is more than sufficient, with an estimated 10-16 minutes for the full pipeline.
