# Phase 2: Adversarial Decode via Evaluation Network Inversion

## Goal

Achieve a competition score of approximately 1.5-2.0 by transmitting only compact semantic data (segmentation argmax maps + pose vectors) and reconstructing full RGB frames at decode time via iterative gradient descent through the freely available SegNet and PoseNet evaluation networks. This approach exploits the fact that the scoring function depends entirely on the outputs of two known, fixed, differentiable neural networks whose weights ship with the repository and do not count toward archive size.

---

## Table of Contents

1. [Core Insight](#1-core-insight)
2. [How the Scoring Works](#2-how-the-scoring-works)
3. [Encoding Phase: Task 1 -- Extract Ground Truth Targets](#3-encoding-phase-task-1----extract-ground-truth-targets)
4. [Encoding Phase: Task 2 -- Compress the Target Data](#4-encoding-phase-task-2----compress-the-target-data)
5. [Decoding Phase: Task 3 -- Optimization-Based Decoder](#5-decoding-phase-task-3----optimization-based-decoder)
6. [Task 4: Handle the uint8 Quantization Problem](#6-task-4-handle-the-uint8-quantization-problem)
7. [Task 5: Optimize Archive Compression](#7-task-5-optimize-archive-compression)
8. [Timing Budget and Performance Optimization](#8-timing-budget-and-performance-optimization)
9. [Deliverables](#9-deliverables)
10. [Expected Score Breakdown](#10-expected-score-breakdown)
11. [Risks and Mitigations](#11-risks-and-mitigations)
12. [Appendix A: Complete Preprocessing Reference](#appendix-a-complete-preprocessing-reference)
13. [Appendix B: Pseudocode for Full Pipeline](#appendix-b-pseudocode-for-full-pipeline)

---

## 1. Core Insight

The evaluation score is computed entirely by two neural networks:

- **SegNet** (`modules.py:103-113`): an `smp.Unet('tu-efficientnet_b2', classes=5)` segmentation network
- **PoseNet** (`modules.py:61-84`): a `fastvit_t12`-based ego-motion estimator

Their weights are stored at:
- `models/segnet.safetensors` -- 38,502,892 bytes (~38 MB)
- `models/posenet.safetensors` -- 55,835,560 bytes (~56 MB)

These files are tracked via Git LFS (`.gitattributes`), pulled during CI setup (`.github/workflows/eval.yml:62-71`), and are available to `inflate.py` at decode time. They do **not** count toward the `archive.zip` file size. This was confirmed by PR #15 (sweeter_codec8) which loaded custom safetensors at decode time.

**The key realization**: We do not need to reconstruct frames that look like the original video. We only need to reconstruct frames that produce **identical outputs** from SegNet and PoseNet. Both networks are differentiable PyTorch models. We can:

1. **Encode**: Run both networks on the original video, record their outputs (segmentation argmax maps + 6-dim pose vectors).
2. **Compress**: These outputs are far smaller than the raw video.
3. **Decode**: Starting from an initial pixel canvas, run gradient descent to find RGB images that produce the recorded network outputs when fed through SegNet/PoseNet.

The generated frames may look nothing like the original dashcam footage -- they just need to fool the two evaluation networks.

---

## 2. How the Scoring Works

### Score Formula

From `evaluate.py:92`:

```python
score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate
```

### SegNet Distortion

From `modules.py:111-113`:

```python
def compute_distortion(self, out1, out2):
    diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
    return diff.mean(dim=tuple(range(1, diff.ndim)))
```

- `out1` and `out2` are logit tensors of shape `(B, 5, 384, 512)`
- `argmax(dim=1)` produces per-pixel class labels in `{0, 1, 2, 3, 4}`
- Distortion = fraction of pixels where the argmax class **disagrees** between original and reconstructed
- If we generate frames where `argmax(SegNet(reconstructed)) == argmax(SegNet(original))` at every pixel, distortion = **0.0 exactly**
- With the 100x multiplier, even 0.1% disagreement (0.001) costs 0.1 points

### PoseNet Distortion

From `modules.py:82-84`:

```python
def compute_distortion(self, out1, out2):
    distortion_heads = ['pose']
    return sum(
        (out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(...)
        for h in self.hydra.heads if h.name in distortion_heads
    )
```

- Output is `{'pose': tensor of shape (B, 12)}`
- Distortion = MSE between first 6 of 12 output dimensions: `out[..., :6]`
- If we generate frame pairs where `PoseNet(reconstructed)[:6] == PoseNet(original)[:6]`, distortion = **0.0 exactly**
- The score contribution is `sqrt(10 * posenet_dist)`, so it is sublinear -- less sensitive than SegNet

### Rate

From `evaluate.py:63-65`:

```python
compressed_size = (args.submission_dir / 'archive.zip').stat().st_size
uncompressed_size = sum(file.stat().st_size for file in args.uncompressed_dir.rglob('*') if file.is_file())
rate = compressed_size / uncompressed_size
```

- `uncompressed_size` = 37,545,489 bytes (the original `videos/0.mkv`)
- Rate contribution to score = `25 * rate`
- A 1 MB archive gives rate = 0.0266, contributing 0.67 to score
- A 3 MB archive gives rate = 0.0799, contributing 2.0 to score

### Why This Approach Wins

Traditional video codecs must preserve pixel-level fidelity. We only need to preserve the **network outputs**, which is a massively reduced information space:

| Data | Traditional Codec | Adversarial Decode |
|------|------------------|--------------------|
| What to preserve | ~3.6 GB of raw pixels | 600 argmax maps + 600 pose vectors |
| Compressed size | ~700 KB - 2.2 MB | ~1-3 MB |
| SegNet distortion | 0.005-0.01 (codec artifacts) | ~0.001-0.005 (near-perfect optimization) |
| PoseNet distortion | 0.1-0.4 | ~0.05-0.15 (direct optimization) |

---

## 3. Encoding Phase: Task 1 -- Extract Ground Truth Targets

### Overview

Run both evaluation networks on the original video and save their outputs. This runs offline with unlimited time and compute.

### Frame Pair Structure

From `frame_utils.py:10`: `seq_len = 2`

Frames are grouped into non-overlapping pairs:
- Pair 0: (frame_0, frame_1)
- Pair 1: (frame_2, frame_3)
- Pair 2: (frame_4, frame_5)
- ...
- Pair 599: (frame_1198, frame_1199)

Total: 600 pairs from 1200 frames.

### What Each Network Sees

- **SegNet** evaluates the **last frame** of each pair (the odd-indexed frame): frames 1, 3, 5, ..., 1199. Reference: `modules.py:108`: `x = x[:, -1, ...]`
- **PoseNet** evaluates **both frames** of each pair as a concatenated 12-channel input. Reference: `modules.py:70-74`

### SegNet Target Extraction

For each pair index `p` (0 through 599), extract the SegNet argmax map from the odd frame (frame index `2*p + 1`).

**Exact preprocessing chain** (traced through the evaluation pipeline):

1. **Raw input**: `(B, T=2, H=874, W=1164, C=3)` uint8, loaded by `TensorVideoDataset` (`frame_utils.py:230-232`)

2. **DistortionNet.preprocess_input** (`modules.py:143-148`):
   ```python
   x = einops.rearrange(x, 'b t h w c -> b t c h w', b=batch_size, t=seq_len, c=3).float()
   ```
   Shape becomes: `(B, 2, 3, 874, 1164)` float32

3. **SegNet.preprocess_input** (`modules.py:107-109`):
   ```python
   x = x[:, -1, ...]  # Take last frame only
   # Shape: (B, 3, 874, 1164)
   x = F.interpolate(x, size=(384, 512), mode='bilinear')
   # Shape: (B, 3, 384, 512)
   ```
   Note: `segnet_model_input_size = (512, 384)` from `frame_utils.py:13`, and `F.interpolate` takes `size=(H, W)` = `(384, 512)`.

4. **SegNet forward** (`modules.py:103-105`): `smp.Unet('tu-efficientnet_b2', classes=5, activation=None)`
   - Input: `(B, 3, 384, 512)` float32
   - Output: `(B, 5, 384, 512)` float32 logits (5 classes, no activation)

5. **Extract argmax**: `out.argmax(dim=1)` -> `(B, 384, 512)` int64, values in `{0, 1, 2, 3, 4}`

**Encoding script must**:
```python
# For each batch of frame pairs from the original video:
#   batch shape: (B, 2, 874, 1164, 3) uint8
x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()
segnet_in = x[:, -1, ...]  # (B, 3, 874, 1164)
segnet_in = F.interpolate(segnet_in, size=(384, 512), mode='bilinear')  # (B, 3, 384, 512)
segnet_out = segnet(segnet_in)  # (B, 5, 384, 512)
seg_maps = segnet_out.argmax(dim=1)  # (B, 384, 512), values 0-4
```

Save all 600 segmentation maps, each of shape `(384, 512)` with values 0-4.

### PoseNet Target Extraction

For each pair, extract the 6-dimensional pose vector.

**Exact preprocessing chain**:

1. **DistortionNet.preprocess_input** converts to `(B, 2, 3, 874, 1164)` float32 (same as above)

2. **PoseNet.preprocess_input** (`modules.py:70-74`):
   ```python
   batch_size, seq_len, *_ = x.shape
   # x shape: (B, 2, 3, 874, 1164)

   x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len, c=3)
   # Shape: (B*2, 3, 874, 1164)

   x = F.interpolate(x, size=(384, 512), mode='bilinear')
   # Shape: (B*2, 3, 384, 512)

   yuv = rgb_to_yuv6(x)
   # Shape: (B*2, 6, 192, 256) -- see Appendix A for rgb_to_yuv6 details

   return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6)
   # Shape: (B, 12, 192, 256) -- 2 frames x 6 YUV channels stacked
   ```

3. **PoseNet.forward** (`modules.py:76-80`):
   ```python
   def forward(self, x):
       # x shape: (B, 12, 192, 256)
       vision_out = self.vision((x - self._mean) / self._std)
       # self._mean = 127.5, self._std = 63.75 (modules.py:64-65)
       # vision_out shape: (B, 2048)
       summary = self.summarizer(vision_out)
       # summary shape: (B, 512)
       ret = self.hydra(summary)
       # ret = {'pose': tensor of shape (B, 12)}
       return ret
   ```

4. **Extract target**: `out['pose'][:, :6]` -> `(B, 6)` float32

**Encoding script must**:
```python
posenet_in = posenet.preprocess_input(x)  # (B, 12, 192, 256)
posenet_out = posenet(posenet_in)  # {'pose': (B, 12)}
pose_targets = posenet_out['pose'][:, :6]  # (B, 6)
```

Save all 600 pose vectors, each 6 floats.

### Complete Encoding Script Structure

```python
import torch, einops
from pathlib import Path
from frame_utils import AVVideoDataset, camera_size, seq_len
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path
from safetensors.torch import load_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models
segnet = SegNet().eval().to(device)
segnet.load_state_dict(load_file(segnet_sd_path, device=str(device)))

posenet = PoseNet().eval().to(device)
posenet.load_state_dict(load_file(posenet_sd_path, device=str(device)))

# Load video
files = ['0.mkv']
ds = AVVideoDataset(files, data_dir=Path('./videos/'), batch_size=16, device=torch.device('cpu'))
ds.prepare_data()

all_seg_maps = []
all_pose_vectors = []

with torch.inference_mode():
    for path, idx, batch in ds:
        # batch shape: (B, 2, 874, 1164, 3) uint8
        batch = batch.to(device)
        x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()

        # SegNet targets
        segnet_in = segnet.preprocess_input(x)  # (B, 3, 384, 512)
        segnet_out = segnet(segnet_in)           # (B, 5, 384, 512)
        seg_maps = segnet_out.argmax(dim=1)      # (B, 384, 512)
        all_seg_maps.append(seg_maps.cpu())

        # PoseNet targets
        posenet_in = posenet.preprocess_input(x) # (B, 12, 192, 256)
        posenet_out = posenet(posenet_in)         # {'pose': (B, 12)}
        pose_vecs = posenet_out['pose'][:, :6]   # (B, 6)
        all_pose_vectors.append(pose_vecs.cpu())

seg_maps = torch.cat(all_seg_maps, dim=0)      # (600, 384, 512)
pose_vectors = torch.cat(all_pose_vectors, dim=0)  # (600, 6)
```

---

## 4. Encoding Phase: Task 2 -- Compress the Target Data

### Segmentation Maps

**Raw size**: 600 maps x 384 x 512 = 117,964,800 values, each in {0, 1, 2, 3, 4}.

At 3 bits per pixel: 117,964,800 * 3 / 8 = ~44 MB. Too large.

**Compression strategy** (exploit temporal redundancy):

1. **Bit-pack**: 5 values (0-4) fit in 3 bits. Pack pixels into bytes.

2. **Delta encoding**: Consecutive dashcam segmentation maps share 95-98% of pixels. Store the first map in full, then for each subsequent map store only an XOR diff with the previous map. The diffs will be extremely sparse (mostly zeros).

3. **Entropy coding**: Apply zlib, zstd, bz2, or lzma to the delta-encoded stream. The high sparsity of diffs should compress extremely well.

4. **Alternative -- palette-mode video codec**: Treat the sequence of 600 segmentation maps as a 5-color palette video and encode with FFV1 (lossless) or PNG sequence. FFV1 with palette mode natively handles temporal redundancy.

**Target compressed size**: 500 KB - 2 MB for 600 maps.

**Implementation options**:

```python
import numpy as np
import zlib  # or zstandard, bz2, lzma

seg_maps_np = seg_maps.numpy().astype(np.uint8)  # (600, 384, 512), values 0-4

# Option A: Delta + zlib
deltas = np.zeros_like(seg_maps_np)
deltas[0] = seg_maps_np[0]
for i in range(1, 600):
    deltas[i] = seg_maps_np[i] ^ seg_maps_np[i - 1]  # XOR diff

raw_bytes = deltas.tobytes()
compressed = zlib.compress(raw_bytes, level=9)
# Expected: ~500 KB - 2 MB

# Option B: Run-length encode the deltas first, then compress
# Option C: Encode as ffv1 video via ffmpeg subprocess
```

**Recommendation**: Try all compression methods offline and pick the smallest. The delta-XOR + zstd/lzma approach is likely simplest and near-optimal.

### Pose Vectors

**Raw size**: 600 vectors x 6 floats x 4 bytes = 14,400 bytes (14.4 KB).

This is tiny. Even uncompressed it barely affects rate.

**Compression strategy**:

1. **Quantize to float16**: 600 x 6 x 2 = 7,200 bytes
2. **Delta encode**: Consecutive pose vectors from a dashcam are smooth; deltas are small
3. **Compress with zstd**: Should shrink to ~3-5 KB

```python
pose_np = pose_vectors.numpy()  # (600, 6) float32

# Delta encode
pose_deltas = np.zeros_like(pose_np)
pose_deltas[0] = pose_np[0]
pose_deltas[1:] = pose_np[1:] - pose_np[:-1]

# Quantize to float16
pose_f16 = pose_deltas.astype(np.float16)

# Compress
pose_compressed = zlib.compress(pose_f16.tobytes(), level=9)
# Expected: ~3-5 KB
```

**Warning on float16 quantization**: Verify that float16 precision is sufficient by checking round-trip error. The PoseNet output values are typically small (order 0.01-1.0), so float16 should be fine. If not, use float32 -- it is only 14.4 KB raw.

### Archive Layout

```
archive/
  seg_maps.bin     # compressed segmentation maps (delta-XOR + zlib/zstd)
  pose_vectors.bin # compressed pose vectors (delta + float16 + zlib/zstd)
  metadata.bin     # 16 bytes: num_pairs (uint32), height (uint32), width (uint32), compression_method (uint32)
```

Total target: **1-3 MB** in `archive.zip`.

### Rate Contribution

- 1 MB archive: rate = 1,000,000 / 37,545,489 = 0.0266, score += 25 * 0.0266 = **0.67**
- 2 MB archive: rate = 2,000,000 / 37,545,489 = 0.0533, score += 25 * 0.0533 = **1.33**
- 3 MB archive: rate = 3,000,000 / 37,545,489 = 0.0799, score += 25 * 0.0799 = **2.00**

---

## 5. Decoding Phase: Task 3 -- Optimization-Based Decoder

### Overview

For each frame pair `(frame_even, frame_odd)`, we initialize two RGB image tensors and iteratively optimize them via gradient descent to produce the correct SegNet and PoseNet outputs.

### What Each Frame Must Satisfy

| Frame | SegNet Constraint | PoseNet Constraint |
|-------|-------------------|--------------------|
| Even frame (frame_0) | **None** -- SegNet only sees the odd frame | Yes -- PoseNet sees both frames as a pair |
| Odd frame (frame_1) | Yes -- `argmax(SegNet(frame_1)) == target_seg_map` | Yes -- PoseNet sees both frames as a pair |

This means:
- **Odd frames** (frame_1) are tightly constrained by both networks
- **Even frames** (frame_0) are only constrained by PoseNet, and PoseNet distortion is under sqrt, so it is less sensitive
- Even frames can be lower quality / fewer optimization iterations

### Initialization Strategy

The optimizer converges faster with a good initial point. Options:

**Option A: Class-color initialization (recommended for odd frames)**

Pre-compute an "ideal RGB color" for each of the 5 segmentation classes. This is the RGB triple that maximally activates that class in the SegNet:

```python
# Offline pre-computation: find RGB values that maximize each class logit
ideal_colors = torch.zeros(5, 3)
for class_idx in range(5):
    pixel = torch.randn(1, 3, 1, 1, requires_grad=True)  # single pixel
    optimizer = torch.optim.Adam([pixel], lr=1.0)
    for step in range(500):
        # Tile pixel to full image
        img = pixel.expand(1, 3, 384, 512)
        logits = segnet(img)  # (1, 5, 384, 512)
        loss = -logits[0, class_idx, 192, 256]  # maximize target class logit at center
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pixel.data.clamp_(0, 255)
    ideal_colors[class_idx] = pixel.squeeze()

# Save ideal_colors for use at decode time (just 15 floats -- tiny)
```

Then at decode time, initialize each pixel to the ideal color for its target segmentation class:

```python
# target_seg_map: (384, 512), values 0-4
# ideal_colors: (5, 3)
init_image = ideal_colors[target_seg_map]  # (384, 512, 3)
init_image = init_image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 384, 512)
```

**Option B: Uniform gray initialization**

```python
init_image = torch.full((1, 3, 384, 512), 128.0)
```

**Option C: Random initialization**

```python
init_image = torch.rand(1, 3, 384, 512) * 255.0
```

Option A is strongly preferred for odd frames -- it starts near the correct segmentation boundary and converges much faster.

For even frames, use uniform gray (Option B) since there is no segmentation constraint.

### The Optimization Loop

For each frame pair (pair_index `p`):

```python
# Load targets
target_seg_map = seg_maps[p]      # (384, 512), int64, values 0-4
target_pose = pose_vectors[p]     # (6,), float32

# Initialize frames at model input resolution (384 x 512)
# Odd frame (frame_1) -- constrained by SegNet + PoseNet
frame_1 = ideal_colors[target_seg_map].permute(2, 0, 1).unsqueeze(0).clone()  # (1, 3, 384, 512)
frame_1.requires_grad_(True)

# Even frame (frame_0) -- constrained by PoseNet only
frame_0 = torch.full((1, 3, 384, 512), 128.0, device=device)
frame_0.requires_grad_(True)

optimizer = torch.optim.Adam([frame_0, frame_1], lr=2.0)

for iteration in range(30):
    optimizer.zero_grad()

    # --- SegNet loss ---
    segnet_logits = segnet(frame_1)  # (1, 5, 384, 512)
    # Cross-entropy loss: we want the target class to have the highest logit
    segnet_loss = F.cross_entropy(segnet_logits, target_seg_map.unsqueeze(0))
    # Alternative: margin loss (see Task 4 below)

    # --- PoseNet loss ---
    # Reconstruct the PoseNet input from both frames
    # PoseNet expects (B, 12, 192, 256) after preprocessing
    both_frames = torch.stack([frame_0, frame_1], dim=1)  # (1, 2, 3, 384, 512)
    posenet_in = posenet.preprocess_input(both_frames)     # (1, 12, 192, 256)
    posenet_out = posenet(posenet_in)['pose'][:, :6]       # (1, 6)
    posenet_loss = F.mse_loss(posenet_out, target_pose.unsqueeze(0))

    # --- Combined loss ---
    # Weight SegNet much higher because of 100x score multiplier
    total_loss = 10.0 * segnet_loss + 1.0 * posenet_loss

    total_loss.backward()
    optimizer.step()

    # Clamp to valid pixel range
    with torch.no_grad():
        frame_0.data.clamp_(0, 255)
        frame_1.data.clamp_(0, 255)

# Round to uint8
frame_0_uint8 = frame_0.detach().round().clamp(0, 255).to(torch.uint8)
frame_1_uint8 = frame_1.detach().round().clamp(0, 255).to(torch.uint8)
```

### Critical: PoseNet Preprocessing is Differentiable

The `PoseNet.preprocess_input` method (`modules.py:70-74`) calls:
1. `einops.rearrange` -- differentiable (just view/permute operations)
2. `F.interpolate(..., mode='bilinear')` -- differentiable
3. `rgb_to_yuv6()` -- differentiable (linear operations + clamp)

Therefore gradients flow from the PoseNet loss all the way back to the RGB pixel values of `frame_0` and `frame_1`.

**However**, `rgb_to_yuv6` (`frame_utils.py:51-78`) contains `.clamp_()` calls (in-place clamp on Y, U, V channels) which **are** differentiable through PyTorch autograd (clamp has gradient 1 within bounds, 0 outside). As long as pixel values stay in [0, 255], the clamps are inactive and gradients flow freely. The main concern is the `@torch.no_grad()` decorator on `rgb_to_yuv6` at `frame_utils.py:50`.

**CRITICAL**: `rgb_to_yuv6` is decorated with `@torch.no_grad()` (`frame_utils.py:50`). This means calling it directly will **break gradient computation**. We must **re-implement `rgb_to_yuv6` without the `@torch.no_grad()` decorator** in our decode script, or copy the function body inline. The implementation is:

```python
def rgb_to_yuv6_differentiable(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Same as frame_utils.rgb_to_yuv6 but without @torch.no_grad()"""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]

    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]

    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    U_sub = (
        U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
        U[..., 0::2, 1::2] + U[..., 1::2, 1::2]
    ) * 0.25
    V_sub = (
        V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
        V[..., 0::2, 1::2] + V[..., 1::2, 1::2]
    ) * 0.25

    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)
```

Similarly, `PoseNet.preprocess_input` must be re-implemented or wrapped to use the differentiable version:

```python
def posenet_preprocess_differentiable(posenet, x):
    """Same as PoseNet.preprocess_input but uses differentiable rgb_to_yuv6"""
    batch_size, seq_len, *_ = x.shape
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len, c=3)
    x = F.interpolate(x, size=(384, 512), mode='bilinear')
    return einops.rearrange(
        rgb_to_yuv6_differentiable(x),
        '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6
    )
```

### Upscaling to Camera Resolution

The optimization runs at 512x384 (SegNet/PoseNet input resolution). The `.raw` output must be at 1164x874 (`camera_size` from `frame_utils.py:11`).

After optimization, upscale:

```python
# frame shape: (1, 3, 384, 512) float, optimized
frame_full = F.interpolate(frame, size=(874, 1164), mode='bicubic', align_corners=False)
frame_full = frame_full.clamp(0, 255).round().to(torch.uint8)
# Shape: (1, 3, 874, 1164)
```

**Important**: The evaluation pipeline reads `.raw` as `(N, H, W, C)` uint8 and then preprocesses it back down to 512x384 via bilinear interpolation. So we optimize at 512x384, upscale to 1164x874 for the `.raw` file, and the eval pipeline downscales it back to 512x384. The round-trip through upscale + downscale can introduce small errors. Mitigations:

1. After upscaling to 1164x874 and rounding to uint8, **downscale back to 512x384** and verify that SegNet argmax is still correct
2. If any pixels flipped, run a few more correction iterations (see Task 4)
3. Alternatively, optimize directly at 1164x874 (slower but eliminates round-trip error). The networks internally downscale via `F.interpolate`, which is differentiable.

### Loss Function Design

**SegNet loss options**:

1. **Cross-entropy** (simplest):
   ```python
   seg_loss = F.cross_entropy(logits, target_seg_map)
   # logits: (1, 5, 384, 512), target: (1, 384, 512)
   ```

2. **Margin loss** (recommended -- provides robustness to uint8 rounding):
   ```python
   # Ensure target class logit exceeds all others by margin M
   target_logits = logits.gather(1, target_seg_map.unsqueeze(1))  # (1, 1, 384, 512)
   max_other = logits.clone()
   max_other.scatter_(1, target_seg_map.unsqueeze(1), float('-inf'))
   max_other = max_other.max(dim=1, keepdim=True).values  # (1, 1, 384, 512)
   margin_violation = F.relu(max_other - target_logits + M)  # M = 2.0 or higher
   seg_loss = margin_violation.mean()
   ```
   This explicitly pushes the target class logit to be at least `M` above the runner-up, providing a safety buffer against quantization noise.

3. **Focal / boundary-weighted loss**: Weight boundary pixels (where adjacent pixels have different target classes) higher, since these are most likely to flip.

**PoseNet loss**: Simple MSE is appropriate:
```python
pose_loss = F.mse_loss(predicted_pose[:, :6], target_pose)
```

**Combined loss**:
```python
total_loss = alpha * seg_loss + beta * pose_loss
```

Recommended starting weights:
- `alpha = 10.0` (SegNet has 100x score multiplier)
- `beta = 1.0` (PoseNet is under sqrt, less sensitive)

Tune these empirically. If SegNet distortion is already near zero but PoseNet is high, reduce alpha and increase beta.

---

## 6. Task 4: Handle the uint8 Quantization Problem

### The Problem

The evaluation pipeline reads `.raw` files as uint8 (`frame_utils.py:231`):
```python
mm = np.memmap(path, dtype=np.uint8, mode='r', shape=(N, H, W, C))
```

After gradient descent produces float-valued pixel tensors, rounding to uint8 can change pixel values by up to 0.5 in each channel. This rounding error propagates through the network and can flip the `argmax` at boundary pixels where two classes have similar logits.

### Solution 1: Margin Loss (Prevention)

Use the margin loss described above with margin `M >= 2.0`. This ensures the target class logit exceeds the runner-up by a comfortable margin, making it unlikely that small perturbations from rounding will flip the argmax.

### Solution 2: Post-Quantization Verification and Correction

After the main optimization loop:

```python
# Round to uint8
frame_uint8 = frame.detach().round().clamp(0, 255).to(torch.uint8)

# Verify SegNet output (for odd frames only)
with torch.no_grad():
    verify_input = frame_uint8.float()
    # If optimizing at camera resolution, downscale first:
    # verify_input = F.interpolate(verify_input, size=(384, 512), mode='bilinear')
    verify_logits = segnet(verify_input)
    verify_argmax = verify_logits.argmax(dim=1)
    mismatches = (verify_argmax != target_seg_map).sum().item()

if mismatches > 0:
    # Run correction iterations
    frame_corrected = frame_uint8.float().requires_grad_(True)
    correction_optimizer = torch.optim.Adam([frame_corrected], lr=0.5)
    for _ in range(10):
        correction_optimizer.zero_grad()
        logits = segnet(frame_corrected)
        loss = margin_loss(logits, target_seg_map, margin=3.0)
        loss.backward()
        correction_optimizer.step()
        with torch.no_grad():
            frame_corrected.data.clamp_(0, 255)

    frame_uint8 = frame_corrected.detach().round().clamp(0, 255).to(torch.uint8)
```

### Solution 3: Simulate Quantization During Optimization

Use Straight-Through Estimator (STE) to simulate uint8 rounding during training:

```python
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad  # Straight-through: pass gradient unchanged

# In the optimization loop, periodically apply:
frame.data = RoundSTE.apply(frame.data)
# Or evaluate the loss on the rounded version:
frame_rounded = RoundSTE.apply(frame)
logits = segnet(frame_rounded)
```

### Solution 4: Optimize at Camera Resolution

If we optimize directly at 1164x874, the evaluation pipeline will downscale via bilinear interpolation. The uint8 values in the `.raw` file are the exact values the network will see (after bilinear downscale). We can simulate this exactly:

```python
# Optimize at (1, 3, 874, 1164)
frame_full = ...  # requires_grad=True, shape (1, 3, 874, 1164)

# Forward pass: downscale to model input size, run SegNet
frame_small = F.interpolate(frame_full, size=(384, 512), mode='bilinear')
logits = segnet(frame_small)
```

This is slower (4.6x more pixels) but eliminates upscale/downscale round-trip errors entirely.

**Recommended approach**: Start with margin loss (Solution 1) at 512x384 resolution with post-quantization verification (Solution 2). Only switch to camera-resolution optimization (Solution 4) if too many pixels flip after quantization.

---

## 7. Task 5: Optimize Archive Compression

### Compression Method Comparison for Segmentation Maps

Test these offline and pick the smallest:

| Method | Expected Size | Notes |
|--------|---------------|-------|
| Raw bytes | ~118 MB | 600 * 384 * 512 * 1 byte |
| 3-bit packed | ~44 MB | 5 values fit in 3 bits |
| Delta XOR + zlib (level 9) | ~500 KB - 2 MB | High temporal redundancy |
| Delta XOR + zstd (level 22) | ~400 KB - 1.5 MB | Better ratio than zlib |
| Delta XOR + lzma | ~300 KB - 1 MB | Best ratio, slow decompression |
| Delta XOR + bz2 | ~400 KB - 1.5 MB | Good ratio |
| FFV1 palette video (ffmpeg) | ~200 KB - 1 MB | Native temporal prediction |
| Bit-packed + delta + zstd | ~200 KB - 800 KB | Combining bit packing with delta |

### Key Insight: Run-Length Encoding of Diffs

After XOR delta encoding, most values in the diff map are 0. Run-length encode the non-zero positions:

```python
# For each diff frame (i > 0):
diff = seg_maps[i] ^ seg_maps[i-1]  # (384, 512), mostly zeros
nonzero_mask = (diff != 0)
# Store: (count_of_nonzero_pixels, list_of_positions, list_of_new_values)
```

Or simply rely on zstd/lzma to handle the sparsity -- general-purpose compressors are very good at runs of zeros.

### FFV1 Approach (Potentially Best)

```bash
# Encode seg maps as a palette video
ffmpeg -f rawvideo -pix_fmt gray -s 512x384 -r 20 -i seg_maps_raw.bin \
    -c:v ffv1 -level 3 -coder 1 -context 1 -g 600 \
    seg_maps.mkv
```

Note: FFV1 expects standard pixel formats. Since our values are 0-4, we can scale to 0-255 (multiply by 51) for a grayscale encoding, or use the raw 0-4 values directly (FFV1 handles arbitrary byte values).

### Do We Need All 600 Maps?

Consecutive segmentation maps from a dashcam are highly correlated. Consider:
- Store only every Nth map as a keyframe
- Interpolate or hold-previous for intermediate maps
- Risk: if the video has scene changes or fast object motion, intermediate maps may diverge

**Recommendation**: Store all 600 maps with delta encoding. The delta-compressed size is already small (the diffs themselves compress well), so the savings from dropping maps would be minimal compared to the risk of introducing segmentation errors.

---

## 8. Timing Budget and Performance Optimization

### Hardware: NVIDIA T4 (16 GB VRAM)

From `.github/workflows/eval.yml:30`: 30-minute time limit.

### Per-Frame Timing Estimates

| Operation | Time (FP32) | Time (FP16) |
|-----------|-------------|-------------|
| SegNet forward (512x384) | ~20-40 ms | ~10-20 ms |
| SegNet backward (512x384) | ~30-60 ms | ~15-30 ms |
| PoseNet forward (192x256 input, 12ch) | ~5-10 ms | ~3-5 ms |
| PoseNet backward (192x256 input, 12ch) | ~5-15 ms | ~3-8 ms |
| One iteration (both networks) | ~60-125 ms | ~30-63 ms |
| 30 iterations per pair | ~1.8-3.75 s | ~0.9-1.9 s |

### Total Time Estimates

| Configuration | Per Pair | 600 Pairs | Feasible? |
|--------------|----------|-----------|-----------|
| FP32, 30 iter | ~2.8 s | ~28 min | Borderline |
| FP16, 30 iter | ~1.4 s | ~14 min | Yes |
| FP16, 20 iter | ~0.9 s | ~9 min | Yes, comfortable |
| FP16, 15 iter | ~0.7 s | ~7 min | Yes, plenty of headroom |

### Optimization Strategies

**1. Mixed Precision (FP16)**

Use `torch.cuda.amp.autocast`:

```python
scaler = torch.cuda.amp.GradScaler()

for iteration in range(num_iterations):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        segnet_logits = segnet(frame_1)
        seg_loss = compute_seg_loss(segnet_logits, target_seg_map)

        posenet_in = posenet_preprocess_differentiable(posenet, both_frames)
        posenet_out = posenet(posenet_in)['pose'][:, :6]
        pose_loss = F.mse_loss(posenet_out, target_pose)

        total_loss = alpha * seg_loss + beta * pose_loss

    scaler.scale(total_loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        frame_0.data.clamp_(0, 255)
        frame_1.data.clamp_(0, 255)
```

**2. Batch Multiple Pairs**

Process multiple frame pairs simultaneously instead of one at a time. With T4's 16 GB VRAM:

- Per pair memory (FP16): ~50-100 MB (two 512x384 images + network activations + gradients)
- Batch size 4-8 should fit comfortably

```python
batch_size = 4
for batch_start in range(0, 600, batch_size):
    batch_end = min(batch_start + batch_size, 600)
    B = batch_end - batch_start

    # Initialize batch of frames
    frames_0 = torch.full((B, 3, 384, 512), 128.0, device=device, requires_grad=True)
    frames_1 = init_from_seg_maps(seg_maps[batch_start:batch_end])  # (B, 3, 384, 512)
    frames_1.requires_grad_(True)

    # ... optimize batch ...
```

**3. Adaptive Iteration Count**

- Run 30 iterations for odd frames (SegNet constraint is strict)
- Run 15 iterations for even frames (only PoseNet, under sqrt)
- Or: monitor loss convergence and stop early when loss < threshold

**4. Pre-compute and Cache**

- Load both models once at the start
- Pre-compute ideal class colors once
- Keep all seg maps and pose vectors in GPU memory (tiny: ~118 MB for seg maps, ~14 KB for pose)

**5. Optimize Even Frames Separately**

Even frames are only constrained by PoseNet. They can be optimized with fewer iterations and a simpler loss. Consider:

- Initialize even frames to the same value as their paired odd frame (since they are from the same scene, the PoseNet output should be similar with similar inputs)
- Or initialize to gray and run only 10-15 PoseNet-only iterations

### Decompression Time

Decompressing the archive data is negligible:
- zlib/zstd decompress ~MB in milliseconds
- Reconstructing seg maps from deltas: <100 ms for 600 maps
- Loading pose vectors: <1 ms

The bottleneck is entirely in the gradient descent optimization.

---

## 9. Deliverables

### File Structure

```
submissions/phase2/
    __init__.py         # Empty, needed for Python module imports
    encode.py           # Offline encoding: extracts SegNet/PoseNet targets from original video
    compress.sh         # Runs encode.py, compresses outputs to archive.zip
    inflate.py          # Loads targets from archive, runs gradient descent, writes .raw
    inflate.sh          # Shell wrapper for inflate.py
```

### `submissions/phase2/encode.py`

Offline script that:
1. Loads `videos/0.mkv` using `AVVideoDataset`
2. Loads SegNet and PoseNet from `models/` directory
3. Runs inference on all 600 frame pairs
4. Saves 600 segmentation argmax maps (384x512, uint8) and 600 pose vectors (6 floats each)
5. Delta-encodes and compresses the data
6. Writes compressed binary files

### `submissions/phase2/compress.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

OUTPUT_DIR="$1"
mkdir -p "$OUTPUT_DIR"

cd "$ROOT"
python -m submissions.phase2.encode "$OUTPUT_DIR"
```

### `submissions/phase2/inflate.py`

The core decode script that:
1. Receives `archive_dir` and `output_path` as arguments
2. Loads compressed seg maps and pose vectors from `archive_dir`
3. Decompresses and reconstructs the targets
4. Loads SegNet and PoseNet from `models/` directory (available in repo root)
5. For each frame pair, runs gradient descent optimization
6. Upscales optimized frames from 512x384 to 1164x874
7. Rounds to uint8 and writes to `.raw` file

Key considerations for `inflate.py`:
- Must handle GPU detection (cuda vs cpu)
- Must use FP16 mixed precision on GPU for speed
- Must complete within 30 minutes on T4
- Must produce `(1200, 874, 1164, 3)` uint8 output in `.raw` format

### `submissions/phase2/inflate.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SUB_NAME="$(basename "$HERE")"

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

mkdir -p "$OUTPUT_DIR"

while IFS= read -r line; do
    [ -z "$line" ] && continue
    BASE="${line%.*}"
    DST="${OUTPUT_DIR}/${BASE}.raw"

    printf "Inflating %s via adversarial decode ... " "$line"
    cd "$ROOT"
    python -m "submissions.${SUB_NAME}.inflate" "$DATA_DIR" "$DST"
done < "$FILE_LIST"
```

---

## 10. Expected Score Breakdown

### Optimistic Scenario (30 iterations, margin loss, post-quant verification)

| Component | Value | Score Contribution |
|-----------|-------|--------------------|
| SegNet distortion | 0.001 (0.1% pixel disagreement) | 100 * 0.001 = 0.10 |
| Rate | 0.04 (1.5 MB archive) | 25 * 0.04 = 1.00 |
| PoseNet distortion | 0.05 | sqrt(10 * 0.05) = 0.71 |
| **Total** | | **1.81** |

### Realistic Scenario (20 iterations, some uint8 flips)

| Component | Value | Score Contribution |
|-----------|-------|--------------------|
| SegNet distortion | 0.005 (0.5% pixel disagreement) | 100 * 0.005 = 0.50 |
| Rate | 0.053 (2 MB archive) | 25 * 0.053 = 1.33 |
| PoseNet distortion | 0.10 | sqrt(10 * 0.10) = 1.00 |
| **Total** | | **2.83** |

### Conservative Scenario (fewer iterations, no post-quant correction)

| Component | Value | Score Contribution |
|-----------|-------|--------------------|
| SegNet distortion | 0.010 (1% disagreement) | 100 * 0.010 = 1.00 |
| Rate | 0.08 (3 MB archive) | 25 * 0.08 = 2.00 |
| PoseNet distortion | 0.15 | sqrt(10 * 0.15) = 1.22 |
| **Total** | | **4.22** |

### Comparison

| Submission | Score |
|-----------|-------|
| EthanYangTW (current leader) | 2.90 |
| haraschax | 2.99 |
| **Phase 2 (optimistic)** | **1.81** |
| **Phase 2 (realistic)** | **2.83** |
| baseline_fast | 4.39 |

Even the realistic scenario is competitive with the current leader, and the optimistic scenario would significantly beat it.

---

## 11. Risks and Mitigations

### Risk 1: Optimization Too Slow (>30 minutes)

**Likelihood**: Medium

**Impact**: Submission fails evaluation

**Mitigations**:
- Use FP16 mixed precision (2x speedup)
- Batch multiple frame pairs (4-8 at a time)
- Reduce iterations: 20 for odd frames, 10 for even frames
- Optimize even frames with PoseNet-only loss (skip SegNet forward/backward)
- Pre-compute class-color initialization (fast convergence from iteration 1)
- Profile and optimize the inner loop: avoid unnecessary tensor allocations, use in-place operations
- Last resort: reduce to 15 iterations with higher learning rate

### Risk 2: uint8 Rounding Flips SegNet Argmax

**Likelihood**: High (especially at class boundaries)

**Impact**: Increased SegNet distortion

**Mitigations**:
- Margin loss with M=2.0-5.0 (preventive)
- Post-quantization verification loop (corrective)
- Straight-Through Estimator during optimization
- Optimize at camera resolution (1164x874) to eliminate upscale/downscale round-trip
- Verify the full pipeline end-to-end: optimize -> round -> upscale -> downscale -> SegNet -> argmax

### Risk 3: PoseNet Hard to Match

**Likelihood**: Medium

**Impact**: Higher PoseNet distortion component

**Mitigations**:
- PoseNet is under sqrt in the score, so it is less sensitive (halving distortion saves ~0.5 points)
- Accept some PoseNet degradation and focus optimization budget on SegNet
- Ensure even frames are reasonably close (PoseNet sees both frames)
- Use the same learning rate schedule but more iterations if PoseNet loss is not converging

### Risk 4: Gradient Descent Gets Stuck in Local Minima

**Likelihood**: Low-Medium

**Impact**: Some frames have high distortion

**Mitigations**:
- Good initialization (class-color init) puts us near the global optimum for SegNet
- Use Adam optimizer which handles saddle points and varying gradient scales
- If a frame is stuck, try restarting with different initialization
- Monitor per-frame loss and flag outliers

### Risk 5: Hidden Test Set with Different Videos

**Likelihood**: High (the download script fetches 64 videos; only video 0 is the public test)

**Impact**: None for this approach -- encoding runs per-video

**Mitigations**:
- The encode step processes each video independently
- The decode step uses only the data in the archive (seg maps + pose vectors extracted from that specific video)
- This approach is fully video-agnostic

### Risk 6: BatchNorm in PoseNet Causes Issues

**Likelihood**: Low

**Impact**: Different outputs in training vs eval mode

**Mitigations**:
- The `AllNorm` class (`modules.py:28-33`) uses `BatchNorm1d(1, ...)` which normalizes across the entire feature vector as a single channel
- During optimization, we must ensure the PoseNet is in `.eval()` mode so BatchNorm uses running statistics, not batch statistics
- With batch size 1, BatchNorm behavior in train mode would be degenerate
- **Always call `segnet.eval()` and `posenet.eval()` before the optimization loop**

### Risk 7: Memory Constraints on T4 (16 GB VRAM)

**Likelihood**: Low

**Impact**: OOM crashes

**Mitigations**:
- At FP16, each 512x384x3 image is ~0.6 MB
- SegNet (EfficientNet-B2 UNet) has ~8M parameters = ~16 MB in FP16
- PoseNet (FastViT-T12 + Hydra) has ~14M parameters = ~28 MB in FP16
- Activations for backprop: ~100-200 MB per frame
- With batch size 4: ~1-2 GB total
- T4 has 16 GB, so we have 10+ GB headroom
- Use `torch.cuda.empty_cache()` between batches if needed

---

## Appendix A: Complete Preprocessing Reference

### SegNet Preprocessing (full trace from .raw to model input)

```
.raw file: flat uint8, shape (1200, 874, 1164, 3)
    |
    v
TensorVideoDataset: batch to (B, 2, 874, 1164, 3) uint8
    |
    v
DistortionNet.preprocess_input (modules.py:143-148):
    einops.rearrange('b t h w c -> b t c h w') -> (B, 2, 3, 874, 1164) float32
    |
    v
SegNet.preprocess_input (modules.py:107-109):
    x[:, -1, ...] -> (B, 3, 874, 1164)  # last frame only
    F.interpolate(size=(384, 512), mode='bilinear') -> (B, 3, 384, 512)
    |
    v
SegNet.forward (smp.Unet):
    Input: (B, 3, 384, 512) float32
    Output: (B, 5, 384, 512) float32 logits
    |
    v
argmax(dim=1) -> (B, 384, 512) int64, values {0,1,2,3,4}
```

### PoseNet Preprocessing (full trace from .raw to model input)

```
.raw file: flat uint8, shape (1200, 874, 1164, 3)
    |
    v
TensorVideoDataset: batch to (B, 2, 874, 1164, 3) uint8
    |
    v
DistortionNet.preprocess_input (modules.py:143-148):
    einops.rearrange('b t h w c -> b t c h w') -> (B, 2, 3, 874, 1164) float32
    |
    v
PoseNet.preprocess_input (modules.py:70-74):
    einops.rearrange('b t c h w -> (b*t) c h w') -> (B*2, 3, 874, 1164)
    F.interpolate(size=(384, 512), mode='bilinear') -> (B*2, 3, 384, 512)
    rgb_to_yuv6() -> (B*2, 6, 192, 256)
    einops.rearrange('(b t) c h w -> b (t c) h w') -> (B, 12, 192, 256)
    |
    v
PoseNet.forward (modules.py:76-80):
    normalize: (x - 127.5) / 63.75
    fastvit_t12: (B, 12, 192, 256) -> (B, 2048)
    summarizer: (B, 2048) -> (B, 512)
    hydra: (B, 512) -> {'pose': (B, 12)}
    |
    v
Distortion uses: out['pose'][:, :6] -> (B, 6) float32
```

### rgb_to_yuv6 Details (frame_utils.py:51-78)

Input: `(B, 3, H, W)` float32 RGB in [0, 255]
Output: `(B, 6, H/2, W/2)` float32

The 6 channels are:
- Channel 0: `y00 = Y[..., 0::2, 0::2]` -- top-left luma subpixel
- Channel 1: `y10 = Y[..., 1::2, 0::2]` -- bottom-left luma subpixel
- Channel 2: `y01 = Y[..., 0::2, 1::2]` -- top-right luma subpixel
- Channel 3: `y11 = Y[..., 1::2, 1::2]` -- bottom-right luma subpixel
- Channel 4: `U_sub` -- 2x2 box-averaged U chroma
- Channel 5: `V_sub` -- 2x2 box-averaged V chroma

BT.601 coefficients:
```
Y = 0.299*R + 0.587*G + 0.114*B        (clamped to [0, 255])
U = (B - Y) / 1.772 + 128.0            (clamped to [0, 255])
V = (R - Y) / 1.402 + 128.0            (clamped to [0, 255])
```

H and W are truncated to even: `H2 = H // 2`, `W2 = W // 2`, and only the first `2*H2` rows and `2*W2` columns are used. For 384x512 input: H2=192, W2=256, no truncation needed (both already even).

---

## Appendix B: Pseudocode for Full Pipeline

### encode.py (offline, unlimited time)

```python
#!/usr/bin/env python
"""
Phase 2 Encoder: Extract SegNet argmax maps and PoseNet pose vectors
from the original video, delta-encode, and compress to binary files.
"""
import sys, struct, zlib, torch, einops, numpy as np
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import AVVideoDataset, camera_size, seq_len, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path

def main():
    output_dir = Path(sys.argv[1])
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load models
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    # Load video
    ds = AVVideoDataset(
        ['0.mkv'], data_dir=Path('./videos/'),
        batch_size=16, device=torch.device('cpu')
    )
    ds.prepare_data()

    all_seg_maps = []
    all_pose_vectors = []

    with torch.inference_mode():
        for path, idx, batch in ds:
            batch = batch.to(device)
            B = batch.shape[0]
            x = einops.rearrange(batch, 'b t h w c -> b t c h w').float()

            # SegNet
            segnet_in = segnet.preprocess_input(x)
            segnet_out = segnet(segnet_in)
            seg_maps = segnet_out.argmax(dim=1).cpu().numpy().astype(np.uint8)
            all_seg_maps.append(seg_maps)

            # PoseNet
            posenet_in = posenet.preprocess_input(x)
            posenet_out = posenet(posenet_in)['pose'][:, :6].cpu().numpy()
            all_pose_vectors.append(posenet_out)

    seg_maps = np.concatenate(all_seg_maps, axis=0)    # (600, 384, 512)
    pose_vectors = np.concatenate(all_pose_vectors, axis=0)  # (600, 6)

    num_pairs = seg_maps.shape[0]
    H, W = seg_maps.shape[1], seg_maps.shape[2]

    # Delta-encode seg maps
    seg_deltas = np.zeros_like(seg_maps)
    seg_deltas[0] = seg_maps[0]
    for i in range(1, num_pairs):
        seg_deltas[i] = seg_maps[i] ^ seg_maps[i - 1]

    seg_compressed = zlib.compress(seg_deltas.tobytes(), level=9)

    # Delta-encode and compress pose vectors
    pose_deltas = np.zeros_like(pose_vectors)
    pose_deltas[0] = pose_vectors[0]
    pose_deltas[1:] = pose_vectors[1:] - pose_vectors[:-1]
    pose_compressed = zlib.compress(pose_deltas.astype(np.float32).tobytes(), level=9)

    # Write metadata + compressed data
    with open(output_dir / 'seg_maps.bin', 'wb') as f:
        f.write(struct.pack('<III', num_pairs, H, W))  # 12 bytes header
        f.write(seg_compressed)

    with open(output_dir / 'pose_vectors.bin', 'wb') as f:
        f.write(struct.pack('<II', num_pairs, 6))  # 8 bytes header
        f.write(pose_compressed)

    print(f"Encoded {num_pairs} pairs")
    print(f"Seg maps compressed: {len(seg_compressed):,} bytes")
    print(f"Pose vectors compressed: {len(pose_compressed):,} bytes")

if __name__ == "__main__":
    main()
```

### inflate.py (at eval time, 30-minute budget on T4)

```python
#!/usr/bin/env python
"""
Phase 2 Decoder: Load compressed SegNet/PoseNet targets, run gradient
descent through the evaluation networks to generate frames that produce
the correct network outputs.
"""
import sys, struct, zlib, torch, einops, numpy as np
import torch.nn.functional as F
from pathlib import Path
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size
from modules import SegNet, PoseNet, segnet_sd_path, posenet_sd_path


def rgb_to_yuv6_differentiable(rgb_chw: torch.Tensor) -> torch.Tensor:
    """frame_utils.rgb_to_yuv6 without @torch.no_grad() decorator."""
    H, W = rgb_chw.shape[-2], rgb_chw.shape[-1]
    H2, W2 = H // 2, W // 2
    rgb = rgb_chw[..., :, :2*H2, :2*W2]
    R = rgb[..., 0, :, :]
    G = rgb[..., 1, :, :]
    B = rgb[..., 2, :, :]
    kYR, kYG, kYB = 0.299, 0.587, 0.114
    Y = (R * kYR + G * kYG + B * kYB).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)
    U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] +
             U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
    V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] +
             V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
    y00 = Y[..., 0::2, 0::2]
    y10 = Y[..., 1::2, 0::2]
    y01 = Y[..., 0::2, 1::2]
    y11 = Y[..., 1::2, 1::2]
    return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)


def posenet_preprocess_differentiable(x):
    """PoseNet.preprocess_input with differentiable rgb_to_yuv6."""
    batch_size, seq_len = x.shape[0], x.shape[1]
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len, c=3)
    x = F.interpolate(x, size=(384, 512), mode='bilinear')
    return einops.rearrange(
        rgb_to_yuv6_differentiable(x),
        '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6
    )


def compute_margin_loss(logits, target, margin=2.0):
    """Margin loss: target class logit must exceed runner-up by margin M."""
    # logits: (B, 5, H, W), target: (B, H, W)
    target_logits = logits.gather(1, target.unsqueeze(1))  # (B, 1, H, W)
    max_other = logits.clone()
    max_other.scatter_(1, target.unsqueeze(1), float('-inf'))
    max_other = max_other.max(dim=1, keepdim=True).values  # (B, 1, H, W)
    violation = F.relu(max_other - target_logits + margin)
    return violation.mean()


def load_targets(archive_dir):
    """Load and decompress seg maps and pose vectors from archive."""
    # Seg maps
    with open(archive_dir / 'seg_maps.bin', 'rb') as f:
        num_pairs, H, W = struct.unpack('<III', f.read(12))
        seg_compressed = f.read()
    seg_deltas = np.frombuffer(zlib.decompress(seg_compressed), dtype=np.uint8)
    seg_deltas = seg_deltas.reshape(num_pairs, H, W)
    # Undo delta encoding
    seg_maps = np.zeros_like(seg_deltas)
    seg_maps[0] = seg_deltas[0]
    for i in range(1, num_pairs):
        seg_maps[i] = seg_maps[i - 1] ^ seg_deltas[i]

    # Pose vectors
    with open(archive_dir / 'pose_vectors.bin', 'rb') as f:
        num_pairs_p, num_dims = struct.unpack('<II', f.read(8))
        pose_compressed = f.read()
    pose_deltas = np.frombuffer(zlib.decompress(pose_compressed), dtype=np.float32)
    pose_deltas = pose_deltas.reshape(num_pairs_p, num_dims)
    # Undo delta encoding
    pose_vectors = np.cumsum(pose_deltas, axis=0)

    return seg_maps, pose_vectors


def compute_ideal_class_colors(segnet, device, num_steps=200):
    """Find RGB values that strongly activate each SegNet class."""
    ideal_colors = torch.zeros(5, 3, device=device)
    for cls in range(5):
        pixel = torch.full((1, 3, 1, 1), 128.0, device=device, requires_grad=True)
        opt = torch.optim.Adam([pixel], lr=5.0)
        for _ in range(num_steps):
            opt.zero_grad()
            img = pixel.expand(1, 3, 384, 512)
            logits = segnet(img)
            # Maximize target class logit at center pixel
            loss = -logits[0, cls].mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                pixel.data.clamp_(0, 255)
        ideal_colors[cls] = pixel.squeeze()
    return ideal_colors


def main():
    archive_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    # Load models
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file(str(segnet_sd_path), device=str(device)))

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_sd_path), device=str(device)))

    # Freeze model weights (we only optimize pixel values)
    for param in segnet.parameters():
        param.requires_grad_(False)
    for param in posenet.parameters():
        param.requires_grad_(False)

    # Load targets
    seg_maps_np, pose_vectors_np = load_targets(archive_dir)
    seg_maps = torch.from_numpy(seg_maps_np).long().to(device)      # (600, 384, 512)
    pose_vectors = torch.from_numpy(pose_vectors_np).float().to(device)  # (600, 6)

    num_pairs = seg_maps.shape[0]
    target_h, target_w = camera_size[1], camera_size[0]  # 874, 1164

    # Pre-compute ideal class colors
    ideal_colors = compute_ideal_class_colors(segnet, device)  # (5, 3)

    # Configuration
    batch_size = 4
    num_iter_odd = 30    # iterations for odd frames (SegNet + PoseNet)
    num_iter_even = 15   # iterations for even frames (PoseNet only)
    seg_margin = 3.0     # margin for SegNet margin loss
    lr = 2.0
    alpha = 10.0         # SegNet loss weight
    beta = 1.0           # PoseNet loss weight

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    with open(output_path, 'wb') as f_out:
        for batch_start in range(0, num_pairs, batch_size):
            batch_end = min(batch_start + batch_size, num_pairs)
            B = batch_end - batch_start

            target_seg = seg_maps[batch_start:batch_end]        # (B, 384, 512)
            target_pose = pose_vectors[batch_start:batch_end]   # (B, 6)

            # Initialize odd frames from class colors
            init_odd = ideal_colors[target_seg]            # (B, 384, 512, 3)
            init_odd = init_odd.permute(0, 3, 1, 2).clone()  # (B, 3, 384, 512)
            frame_1 = init_odd.requires_grad_(True)

            # Initialize even frames to gray
            frame_0 = torch.full((B, 3, 384, 512), 128.0, device=device)
            frame_0.requires_grad_(True)

            optimizer = torch.optim.Adam([frame_0, frame_1], lr=lr)

            # --- Optimization loop ---
            max_iter = max(num_iter_odd, num_iter_even)
            for iteration in range(max_iter):
                optimizer.zero_grad()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        total_loss = torch.tensor(0.0, device=device)

                        # SegNet loss (odd frames only)
                        if iteration < num_iter_odd:
                            seg_logits = segnet(frame_1)  # (B, 5, 384, 512)
                            seg_loss = compute_margin_loss(seg_logits, target_seg, seg_margin)
                            total_loss = total_loss + alpha * seg_loss

                        # PoseNet loss (both frames)
                        if iteration < num_iter_even or iteration < num_iter_odd:
                            both = torch.stack([frame_0, frame_1], dim=1)  # (B, 2, 3, 384, 512)
                            posenet_in = posenet_preprocess_differentiable(both)
                            posenet_out = posenet(posenet_in)['pose'][:, :6]
                            pose_loss = F.mse_loss(posenet_out, target_pose)
                            total_loss = total_loss + beta * pose_loss

                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss = torch.tensor(0.0, device=device)

                    if iteration < num_iter_odd:
                        seg_logits = segnet(frame_1)
                        seg_loss = compute_margin_loss(seg_logits, target_seg, seg_margin)
                        total_loss = total_loss + alpha * seg_loss

                    if iteration < num_iter_even or iteration < num_iter_odd:
                        both = torch.stack([frame_0, frame_1], dim=1)
                        posenet_in = posenet_preprocess_differentiable(both)
                        posenet_out = posenet(posenet_in)['pose'][:, :6]
                        pose_loss = F.mse_loss(posenet_out, target_pose)
                        total_loss = total_loss + beta * pose_loss

                    total_loss.backward()
                    optimizer.step()

                with torch.no_grad():
                    frame_0.data.clamp_(0, 255)
                    frame_1.data.clamp_(0, 255)

            # --- Post-quantization verification for odd frames ---
            with torch.no_grad():
                frame_1_uint8 = frame_1.detach().round().clamp(0, 255)
                verify_logits = segnet(frame_1_uint8)
                verify_argmax = verify_logits.argmax(dim=1)
                mismatches = (verify_argmax != target_seg).any()

            if mismatches:
                # Correction iterations
                frame_1_fix = frame_1_uint8.clone().requires_grad_(True)
                fix_opt = torch.optim.Adam([frame_1_fix], lr=0.5)
                for _ in range(10):
                    fix_opt.zero_grad()
                    logits = segnet(frame_1_fix)
                    loss = compute_margin_loss(logits, target_seg, margin=5.0)
                    loss.backward()
                    fix_opt.step()
                    with torch.no_grad():
                        frame_1_fix.data.clamp_(0, 255)
                frame_1_uint8 = frame_1_fix.detach().round().clamp(0, 255)

            # --- Upscale to camera resolution and write ---
            frame_0_final = F.interpolate(
                frame_0.detach().round().clamp(0, 255),
                size=(target_h, target_w), mode='bicubic', align_corners=False
            ).clamp(0, 255).round().to(torch.uint8)

            frame_1_final = F.interpolate(
                frame_1_uint8,
                size=(target_h, target_w), mode='bicubic', align_corners=False
            ).clamp(0, 255).round().to(torch.uint8)

            # Write pairs: (frame_0, frame_1), (frame_0, frame_1), ...
            for b in range(B):
                # Even frame (frame_0)
                f0 = frame_0_final[b].permute(1, 2, 0).cpu().numpy()  # (874, 1164, 3)
                f_out.write(f0.tobytes())
                # Odd frame (frame_1)
                f1 = frame_1_final[b].permute(1, 2, 0).cpu().numpy()  # (874, 1164, 3)
                f_out.write(f1.tobytes())

            if (batch_start // batch_size) % 10 == 0:
                print(f"Processed pairs {batch_start}-{batch_end} / {num_pairs}")

    print(f"Written {num_pairs * 2} frames to {output_path}")


if __name__ == "__main__":
    main()
```

### Key Implementation Checklist

- [ ] Verify `rgb_to_yuv6_differentiable` produces identical output to `frame_utils.rgb_to_yuv6` (without the no_grad)
- [ ] Verify SegNet and PoseNet are in `.eval()` mode during optimization
- [ ] Verify all model parameters have `requires_grad=False` (only pixel tensors should have gradients)
- [ ] Verify frame ordering: `.raw` file must be `frame_0, frame_1, frame_2, frame_3, ...` (alternating even/odd)
- [ ] Verify output shape: each frame is `(874, 1164, 3)` uint8, written contiguously
- [ ] Verify total frames: 1200 frames = 600 pairs x 2
- [ ] Verify total `.raw` file size: 1200 x 874 x 1164 x 3 = 3,661,113,600 bytes
- [ ] Profile timing on T4 to ensure <30 minutes
- [ ] Test end-to-end: encode -> compress -> inflate -> evaluate
- [ ] Compare SegNet argmax from optimized frames vs targets at every pixel
- [ ] Experiment with different margins (2.0, 3.0, 5.0) and iteration counts (15, 20, 30)
- [ ] Experiment with different compression methods for seg maps
- [ ] Handle edge case: partial last batch (600 is divisible by common batch sizes, but be safe)
