# Phase 3: Implicit Neural Representation (INR) / Self-Compressing Network

## Goal

Achieve a competition score of approximately 1.5-2.5 by encoding the entire 1200-frame dashcam video as the compressed weights of a small neural network. The submission `archive.zip` contains ONLY the model weights and a tiny inference script. At decode time, the network generates all 1200 frames via forward passes, which are written as the `.raw` output file. No traditional video codec is involved.

---

## Table of Contents

1. [Background and Precedent](#1-background-and-precedent)
2. [Core Concept](#2-core-concept)
3. [Competition Mechanics Recap](#3-competition-mechanics-recap)
4. [The Killer Feature: Task-Aware Training Loss](#4-the-killer-feature-task-aware-training-loss)
5. [Task 1: Architecture Design](#task-1-architecture-design)
6. [Task 2: Training Pipeline](#task-2-training-pipeline)
7. [Task 3: Network Compression](#task-3-network-compression)
8. [Task 4: Inference / Decode Pipeline](#task-4-inference--decode-pipeline)
9. [Task 5: Archive Format and Submission Structure](#task-5-archive-format-and-submission-structure)
10. [Expected Score Breakdown](#expected-score-breakdown)
11. [Risks and Mitigations](#risks-and-mitigations)
12. [Comparison to Phase 2 (Adversarial Decode)](#comparison-to-phase-2-adversarial-decode)
13. [Implementation Roadmap](#implementation-roadmap)
14. [Appendix A: Key File Paths and Constants](#appendix-a-key-file-paths-and-constants)
15. [Appendix B: Reference Pseudocode](#appendix-b-reference-pseudocode)

---

## 1. Background and Precedent

### The Previous commaVQ Challenge

The previous comma.ai compression challenge (commaVQ) was won by participants who trained their own neural networks to memorize the data:

- Reddit commentary: "every top submission trained their own" model.
- Szabolcs-cs used "Self-Compressing Neural Networks" -- a technique where the network is simultaneously trained to represent the data and to minimize its own compressed size.
- The winning strategy was NOT traditional video coding. It was neural memorization: overfit a compact network to the specific video, then ship the network weights as the compressed file.

### Why This Works

A neural network is a universal function approximator. Given enough capacity, it can memorize any finite dataset. The key insight is that a well-designed network with 1.5-3M parameters (roughly 6-12MB in FP32) can represent 1200 frames of 512x384 video with reasonable fidelity. After aggressive quantization (6-8 bit) and entropy coding, these weights compress to 500KB-1.5MB -- far smaller than any traditional codec output at equivalent perceptual quality.

### Relevant Research

| Framework | Venue | Key Idea | Repository |
|-----------|-------|----------|------------|
| **HiNeRV** | NeurIPS 2023 | Hierarchical Neural Representation for Videos -- state-of-the-art INR for video with frame-specific and shared components | github.com/hmkx/HiNeRV |
| **NVRC** | NeurIPS 2024 | Neural Video Representation Compression -- builds on NeRV family with improved rate-distortion | -- |
| **SIREN** | NeurIPS 2020 | Sinusoidal representation networks using periodic activation functions for continuous signals | vsitzmann/siren |
| **Instant-NGP** | SIGGRAPH 2022 | Multi-resolution hash grid encoding + tiny MLP -- extremely fast training and inference | NVlabs/instant-ngp |
| **NeRV** | NeurIPS 2021 | Neural Representations for Videos -- frame-level INR with convolutional decoder | -- |
| **E-NeRV** | ECCV 2022 | Enhanced NeRV with temporal and spatial embeddings | -- |

---

## 2. Core Concept

An Implicit Neural Representation (INR) encodes a video as a function:

```
f(frame_index, x, y) -> (R, G, B)
```

where `f` is a neural network. The network is **overfitted** to the specific video -- it memorizes every frame. The compressed network weights ARE the compressed file.

### At Compression Time (compress.sh)

1. Decode the input video to get all 1200 frames as RGB tensors.
2. Design and instantiate a compact neural network architecture.
3. Train the network to reconstruct the video frames (Phase A: MSE, Phase B: task-aware loss).
4. Prune, quantize, and entropy-code the trained weights.
5. Package the compressed weights + config + inference script into `archive.zip`.

### At Decompression Time (inflate.py)

1. Load the compressed weights from the archive.
2. Reconstruct the network (decompress weights, dequantize, instantiate architecture).
3. Run forward passes for all 1200 frames.
4. Output uint8 RGB at 1164x874 to the `.raw` file.

No SegNet or PoseNet is needed at decode time. The INR directly outputs pixel values.

---

## 3. Competition Mechanics Recap

### Scoring Formula

From `evaluate.py:92`:

```
score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
```

Where:
- `segnet_dist`: Average fraction of pixels where `argmax(segnet_logits_original) != argmax(segnet_logits_reconstructed)` across 600 odd frames (frames 1, 3, 5, ..., 1199). Range: [0, 1].
- `posenet_dist`: MSE of first 6 of 12 output dimensions between original and reconstructed frame pairs. 600 pairs: (0,1), (2,3), ..., (1198,1199).
- `rate`: `archive.zip size / 37,545,489 bytes`.

### Weight Sensitivity

| Component | Example Value | Score Contribution | Sensitivity |
|-----------|---------------|-------------------|-------------|
| SegNet (100x) | 0.01 | 1.0 | **Dominant** -- 1% pixel disagreement = 1.0 points |
| PoseNet (sqrt(10x)) | 0.38 | 1.95 | Moderate -- halving from 0.38 to 0.19 saves 0.57 |
| Rate (25x) | 0.06 | 1.50 | Low -- halving from 6% to 3% saves 0.75 |
| Rate (25x) | 0.02 | 0.50 | Very good -- 750KB archive |

### Current Leaderboard Context

| Submission | Method | Score |
|-----------|--------|-------|
| EthanYangTW (PR #17) | SVT-AV1, 45% scale, CRF 32, preset 0 | 2.90 |
| haraschax (PR #10) | SVT-AV1, 35% scale, CRF 30, preset 4 | 2.99 |
| baseline_fast | libx265, 45% scale, CRF 30, ultrafast | 4.39 |

To win, we need a score below 2.90.

### Video Properties

- **File:** `videos/0.mkv`
- **Frames:** 1200 (20fps, 60 seconds)
- **Resolution:** 1164x874
- **Codec:** H.265/HEVC, Main profile
- **Size:** 37,545,489 bytes
- **Source:** comma2k19 dashcam dataset, segment `b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc`

### Key Dimensions

From `frame_utils.py`:
```python
seq_len = 2
camera_size = (1164, 874)           # (width, height) -- output resolution
segnet_model_input_size = (512, 384) # (width, height) -- SegNet/PoseNet input
```

The evaluation models operate at 512x384. The INR can work at this reduced resolution and bicubic-upscale to 1164x874 at decode time.

---

## 4. The Killer Feature: Task-Aware Training Loss

### The Problem with MSE Training

Standard INR training minimizes pixel-level MSE:

```
L_MSE = || f(t, x, y) - pixel_gt(t, x, y) ||^2
```

This allocates network capacity uniformly across all pixels. Sky regions, road textures, and vehicle boundaries all receive equal fidelity. But the competition score does NOT care about pixel fidelity -- it cares about:

1. **SegNet class boundaries** (100x weight): Only the argmax of 5-class logits matters per pixel.
2. **PoseNet motion vectors** (sqrt(10x) weight): Only 6 scalar outputs per frame pair matter.
3. **File size** (25x weight): Smaller network = lower rate.

### The Solution: Train with the Exact Competition Loss

Instead of MSE, use a differentiable proxy of the competition scoring function:

```python
loss = alpha * segnet_proxy_loss + beta * posenet_mse_loss
```

This means the network will:
- Allocate maximum capacity to preserve lane markings, vehicle boundaries, and other features that affect segmentation class boundaries.
- Allocate significant capacity to preserve inter-frame motion features that PoseNet uses.
- Allocate minimal capacity to sky regions, road textures, and other areas where the segmentation class is stable and pose features are insensitive.

### Accessing the Evaluation Models

The SegNet and PoseNet are available at:
- `models/segnet.safetensors` (38,502,892 bytes, ~38 MB)
- `models/posenet.safetensors` (55,835,560 bytes, ~56 MB)

These are loaded during **training only**. They do NOT need to be in the archive. At decode time, the INR directly outputs RGB frames -- no evaluation model is needed.

Loading code (from `modules.py`):
```python
from modules import DistortionNet, segnet_sd_path, posenet_sd_path

distortion_net = DistortionNet().eval().to(device)
distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
```

### The Differentiability Challenge

The SegNet distortion metric is the fraction of pixels where `argmax(logits_orig) != argmax(logits_recon)`. The `argmax` operation is NOT differentiable (it is a step function). We need a differentiable proxy.

**Option 1: Cross-Entropy Loss (Recommended)**

Pre-compute the ground-truth argmax maps for all 600 odd frames:
```python
# Pre-compute once
gt_class_maps = {}  # frame_idx -> (384, 512) int64 tensor
for frame_idx in [1, 3, 5, ..., 1199]:
    frame = load_frame(frame_idx)  # (3, 384, 512)
    logits = segnet(frame.unsqueeze(0))  # (1, 5, 384, 512)
    gt_class_maps[frame_idx] = logits.argmax(dim=1).squeeze(0)  # (384, 512)
```

Then during training:
```python
recon_frame = inr_model(frame_idx)  # (3, 384, 512)
recon_logits = segnet(recon_frame.unsqueeze(0))  # (1, 5, 384, 512)
segnet_loss = F.cross_entropy(recon_logits, gt_class_maps[frame_idx].unsqueeze(0))
```

Cross-entropy directly encourages the correct class to have the highest logit, which is exactly what the argmax metric measures. It is smooth and well-behaved for gradient descent.

**Option 2: Margin Loss**

```python
# Encourage target class logit to be at least M above runner-up
target_logits = recon_logits.gather(1, gt_class.unsqueeze(1))  # logit of correct class
recon_logits_masked = recon_logits.clone()
recon_logits_masked.scatter_(1, gt_class.unsqueeze(1), -1e9)  # mask target
runner_up = recon_logits_masked.max(dim=1).values  # highest non-target logit
margin_loss = F.relu(M - (target_logits.squeeze(1) - runner_up)).mean()
```

This is more directly aligned with the argmax metric -- it only penalizes when the margin is insufficient.

**Option 3: Soft Argmax with Temperature**

```python
# Temperature-scaled softmax approximates argmax
soft_classes = F.softmax(recon_logits / temperature, dim=1)  # lower temp -> harder
gt_one_hot = F.one_hot(gt_class, num_classes=5).permute(0, 3, 1, 2).float()
segnet_loss = (1 - (soft_classes * gt_one_hot).sum(dim=1)).mean()
```

### PoseNet Proxy Loss

PoseNet loss is already differentiable (it is MSE of continuous outputs):

```python
# Pre-compute ground-truth pose vectors for all 600 frame pairs
gt_poses = {}  # pair_idx -> (6,) tensor
for i in range(0, 1200, 2):
    pair = load_frame_pair(i, i+1)  # (1, 2, 3, 384, 512)
    posenet_in = posenet.preprocess_input(pair)
    out = posenet(posenet_in)
    gt_poses[i // 2] = out['pose'][0, :6].clone()

# During training
recon_pair = inr_model.generate_pair(i, i+1)  # (1, 2, 3, 384, 512)
posenet_in = posenet.preprocess_input(recon_pair)
recon_out = posenet(posenet_in)
posenet_loss = F.mse_loss(recon_out['pose'][0, :6], gt_poses[i // 2])
```

### Combined Loss Weighting

```python
# Match competition weighting approximately
# score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate
# Rate is fixed after compression, so during training:
total_loss = alpha * segnet_ce_loss + beta * posenet_mse_loss + gamma * mse_pixel_loss
```

Suggested starting weights:
- `alpha = 10.0` (SegNet is the dominant scoring term)
- `beta = 1.0` (PoseNet is secondary)
- `gamma = 0.1` (small pixel MSE to maintain spatial coherence and prevent color drift)

These should be tuned on the validation set (a held-out subset of frame pairs).

---

## Task 1: Architecture Design

Design a compact neural network that can represent 1200 frames at 512x384 resolution. After reconstruction, frames are bicubic-upscaled to 1164x874 for the `.raw` output.

### Option A: HiNeRV-Style Hierarchical Architecture (Recommended)

**Reference:** HiNeRV (NeurIPS 2023) -- github.com/hmkx/HiNeRV

HiNeRV uses a hierarchical structure with:
- **Frame-level embeddings**: A learnable embedding per frame (or per group of frames).
- **Shared convolutional decoder**: A stack of upsampling blocks that progressively increase spatial resolution.
- **Skip connections**: Between embedding levels and decoder layers.

**Proposed architecture:**

```
Input: frame_index t (integer, 0-1199)
  |
  v
Frame Embedding Lookup: E[t] -> (C_embed,) vector
  |  e.g., C_embed = 128, total: 128 * 1200 = 153,600 params
  v
Reshape to spatial: (C_embed, H0, W0) where H0=6, W0=8
  |  or use a learned mapping: Linear(C_embed, C0 * H0 * W0)
  v
Upsample Block 1: ConvTranspose2d or PixelShuffle (6x8 -> 12x16)
  |  Conv2d(C0, C1, 3, padding=1), GELU, Conv2d(C1, C1, 3, padding=1), GELU
  v
Upsample Block 2: (12x16 -> 24x32)
  |  Conv2d(C1, C2, 3, padding=1), GELU, Conv2d(C2, C2, 3, padding=1), GELU
  v
Upsample Block 3: (24x32 -> 48x64)
  v
Upsample Block 4: (48x64 -> 96x128)
  v
Upsample Block 5: (96x128 -> 192x256)
  v
Upsample Block 6: (192x256 -> 384x512)
  v
Output Head: Conv2d(C_last, 3, 1), Sigmoid() * 255
  -> (3, 384, 512) RGB frame
```

**Parameter budget:**

| Component | Channels | Params (approx.) |
|-----------|----------|-------------------|
| Frame embeddings | 128-dim x 1200 | 153K |
| Reshape MLP | 128 -> 128*6*8=6144 | 790K |
| Upsample block 1 | 128->96 | 220K |
| Upsample block 2 | 96->64 | 110K |
| Upsample block 3 | 64->48 | 55K |
| Upsample block 4 | 48->32 | 28K |
| Upsample block 5 | 32->24 | 14K |
| Upsample block 6 | 24->16 | 7K |
| Output head | 16->3 | 51 |
| **Total** | | **~1.4M** |

Scale up to 2-3M by increasing channel widths (e.g., 192-dim embeddings, wider blocks).

**Advantages:**
- Single forward pass per frame -- very fast inference (100-200+ fps on T4).
- Naturally exploits temporal redundancy (shared decoder weights, frame-specific embeddings).
- Well-studied architecture with known rate-distortion tradeoffs.

**Implementation detail:** Use `nn.PixelShuffle` for upsampling (better than transposed convolution for checkerboard artifact avoidance).

### Option B: Hash-Grid + MLP (Instant-NGP Style)

**Reference:** Instant-NGP (SIGGRAPH 2022)

```
Input: (t, x, y) normalized coordinates in [0, 1]^3
  |
  v
Multi-Resolution Hash Encoding:
  - L = 16 resolution levels
  - T = 2^14 to 2^19 hash table entries per level
  - F = 2 features per entry
  - Total: L * T * F parameters in hash tables
  |
  v
Concatenated features: (L * F,) = (32,) vector
  |
  v
MLP: 3 hidden layers, 64 units each, ReLU
  -> (3,) RGB output, Sigmoid * 255
```

**Parameter budget:**
- Hash tables: 16 levels * 2^17 entries * 2 features * 4 bytes (FP32) = ~33M params (too large at FP32)
- With 8-bit hash entries: 16 * 2^17 * 2 * 1 byte = ~4MB raw
- MLP: 3 * 64 * 64 + input/output layers = ~15K params
- After entropy coding: ~2-3MB

**Advantages:**
- Extremely fast training (minutes, not hours).
- Very fast inference.
- Excellent at capturing high-frequency spatial detail.

**Disadvantages:**
- Hash tables are less compressible than neural network weights.
- Temporal coherence is implicit (encoded in the t coordinate), not explicit.
- May struggle with the 3D video volume (1200 * 512 * 384 = ~235M sample points).

**Verdict:** Worth experimenting with, but likely inferior to HiNeRV for video due to poorer temporal modeling.

### Option C: Frame-Embedding + Convolutional Decoder

A simpler variant of Option A:

```
Input: frame_index t
  |
  v
Learned latent code: z[t] of dimension D (e.g., D=64)
  |  Total: 64 * 1200 = 76,800 params
  v
Shared Convolutional Decoder: maps z -> (3, 384, 512)
  |  Inspired by StyleGAN / DCGAN decoders
  |  z -> Linear -> Reshape(C, 4, 4) -> ConvTranspose layers -> RGB
  v
Output: (3, 384, 512) RGB frame
```

**Parameter budget:**
- Latent codes: 64 * 1200 = 77K
- Decoder: 500K-1M params (depending on depth/width)
- Total: ~600K-1.1M

**Advantages:**
- Very compact latent codes per frame.
- Decoder captures spatial patterns shared across all frames.
- Easy to implement.

**Disadvantages:**
- Small latent code may not capture frame-specific details.
- Decoder may need to be large to produce sharp 512x384 output.
- Quality ceiling may be lower than HiNeRV.

### Architecture Selection Recommendation

**Primary: Option A (HiNeRV-style)** at ~2-3M parameters.
- Best rate-distortion tradeoff based on published results.
- Well-suited for video data.
- Single forward pass per frame.

**Fallback: Option C (Frame-embedding + decoder)** if HiNeRV proves too complex to implement/train in time.
- Simpler implementation.
- Still a strong baseline for neural video representation.

**Exploration: Option B (Hash-grid)** for fast initial experiments to validate the pipeline.
- Training is very fast.
- Can serve as a quick proof-of-concept.

### Working Resolution

Train the INR at **512x384** (the SegNet/PoseNet input resolution), NOT at 1164x874. Reasons:
1. The scoring models operate at 512x384. Any detail beyond this resolution is wasted.
2. 512x384 is 4.6x fewer pixels than 1164x874 -- much faster training and inference.
3. Bicubic upscaling from 512x384 to 1164x874 is fast and introduces minimal SegNet/PoseNet distortion (the models apply bilinear downscaling anyway).

At decode time: generate 512x384, then `F.interpolate(frame, size=(874, 1164), mode='bicubic', align_corners=False)`.

---

## Task 2: Training Pipeline

### Prerequisites

1. **Decode the video** to obtain all 1200 frames as uint8 RGB tensors at 1164x874.
2. **Downscale** all frames to 512x384 via bilinear interpolation (matching the SegNet/PoseNet preprocessing).
3. **Pre-compute ground truth references:**
   - SegNet argmax class maps for 600 odd frames (1, 3, 5, ..., 1199).
   - PoseNet pose vectors (first 6 dims) for 600 frame pairs ((0,1), (2,3), ..., (1198,1199)).
4. **Load SegNet and PoseNet** on the training GPU. These stay frozen (eval mode, no gradients).

### Phase A: MSE Pre-Training (Fast Convergence)

**Purpose:** Get the INR to a reasonable reconstruction quality before switching to the expensive task-aware loss.

**Setup:**
```python
inr_model = HiNeRVModel(config).to(device)
optimizer = torch.optim.Adam(inr_model.parameters(), lr=5e-4, weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

**Training loop:**
```python
for epoch in range(50):
    for batch_indices in dataloader:  # batch of frame indices
        frames_gt = all_frames[batch_indices].to(device)  # (B, 3, 384, 512)
        frames_pred = inr_model(batch_indices)             # (B, 3, 384, 512)
        loss = F.mse_loss(frames_pred, frames_gt)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
```

**Hyperparameters:**
- Batch size: 8-16 frames (limited by GPU memory since each frame is 512x384x3).
- Learning rate: 5e-4 with cosine annealing to 1e-6.
- Epochs: 30-50 (until PSNR stabilizes around 28-32 dB).
- Expected duration: 1-3 hours on a single GPU (T4/A100).

**Monitoring:**
- Track PSNR and SSIM on the full video every 5 epochs.
- Visualize a few sample frames to check for artifacts.

### Phase B: Task-Aware Fine-Tuning

**Purpose:** Reallocate network capacity from pixel fidelity to task-relevant features (SegNet boundaries, PoseNet motion).

**Setup:**
```python
# Load frozen evaluation models
segnet = SegNet().eval().to(device)
segnet.load_state_dict(load_file('models/segnet.safetensors', device=str(device)))
for p in segnet.parameters():
    p.requires_grad = False

posenet = PoseNet().eval().to(device)
posenet.load_state_dict(load_file('models/posenet.safetensors', device=str(device)))
for p in posenet.parameters():
    p.requires_grad = False

# Reduce learning rate for fine-tuning
optimizer = torch.optim.Adam(inr_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
```

**Training loop (SegNet component):**
```python
# For each odd frame index (1, 3, 5, ..., 1199):
frame_pred = inr_model(frame_idx)  # (3, 384, 512) float [0, 255]
# SegNet expects (B, 3, 384, 512) float RGB
segnet_input = frame_pred.unsqueeze(0)
segnet_logits = segnet(segnet_input)  # (1, 5, 384, 512)
segnet_loss = F.cross_entropy(segnet_logits, gt_class_maps[frame_idx].unsqueeze(0))
```

**Training loop (PoseNet component):**
```python
# For each frame pair (i, i+1) where i in {0, 2, 4, ..., 1198}:
frame_a = inr_model(i)    # (3, 384, 512)
frame_b = inr_model(i+1)  # (3, 384, 512)
# PoseNet expects (B, 2, 3, H, W) float
pair = torch.stack([frame_a, frame_b], dim=0).unsqueeze(0)  # (1, 2, 3, 384, 512)
posenet_in = posenet.preprocess_input(pair)  # (1, 12, 192, 256)
posenet_out = posenet(posenet_in)  # {'pose': (1, 12)}
posenet_loss = F.mse_loss(posenet_out['pose'][0, :6], gt_poses[i // 2])
```

**Combined loss:**
```python
total_loss = 10.0 * segnet_loss + 1.0 * posenet_loss + 0.1 * pixel_mse_loss
```

**Important notes on PoseNet preprocessing:**
- PoseNet's `preprocess_input` expects `(B, T=2, C=3, H, W)` float RGB.
- It resizes to (384, 512), converts to YUV6 (6 channels per frame, half spatial resolution), then stacks 2 frames to get (B, 12, 192, 256).
- It then normalizes: `(x - 127.5) / 63.75`.
- The INR output must be in [0, 255] float range to match this pipeline.

**Important notes on SegNet preprocessing:**
- SegNet's `preprocess_input` expects `(B, T, C, H, W)` and takes the last frame: `x[:, -1, ...]`.
- It resizes to (384, 512) via bilinear interpolation.
- Input is float RGB in [0, 255] range (no explicit normalization in preprocessing, but the model was trained on this range).

**Hyperparameters for Phase B:**
- Batch size: 4-8 frame pairs (each requires SegNet + PoseNet forward passes, which are expensive).
- Learning rate: 1e-4 with cosine annealing to 1e-6.
- Epochs: 10-20.
- Expected duration: 3-8 hours on a single GPU (each step requires SegNet and PoseNet forward passes through the frozen models).

### Memory Management

The training step involves:
1. INR forward pass: small memory footprint.
2. SegNet forward pass (frozen): ~38M params, (1, 5, 384, 512) output.
3. PoseNet forward pass (frozen): ~56M params, (1, 12) output.

Total GPU memory needed: ~4-6 GB for models + ~2-4 GB for activations = ~6-10 GB. A T4 (16GB) should suffice. If tight, use gradient checkpointing on the INR.

**Gradient flow:**
- Gradients flow from the loss through the frozen SegNet/PoseNet BACK through the INR's generated frames to the INR's weights.
- The frozen models must be in `eval()` mode but their forward pass must NOT be under `torch.no_grad()` -- the INR's output frames need gradients.
- Only the INR parameters are updated; SegNet/PoseNet parameters stay frozen.

```python
# CORRECT: gradients flow through frozen models to INR
segnet.eval()
for p in segnet.parameters():
    p.requires_grad = False  # don't update segnet weights
# But DO allow gradient computation through segnet's forward pass

frame_pred = inr_model(idx)  # has grad
logits = segnet(frame_pred.unsqueeze(0))  # grad flows through segnet to frame_pred
loss = F.cross_entropy(logits, target)
loss.backward()  # updates inr_model.parameters() only
```

### Training Data Augmentation

No augmentation is needed -- we are overfitting to this specific video intentionally. The entire video is the training set.

However, during task-aware training, consider:
- **Random frame sampling:** Instead of iterating sequentially, randomly sample frame pairs per batch. This provides better gradient diversity.
- **Frame jittering:** Small random spatial shifts (1-2 pixels) during SegNet loss computation to improve robustness of class boundaries.

---

## Task 3: Network Compression

After training, the INR weights must be compressed for the archive. The compression pipeline has three stages.

### Step 1: Structured Pruning (15-30% Parameter Reduction)

Remove channels/neurons that contribute least to the output quality.

**Method:**
```python
import torch.nn.utils.prune as prune

# L1-norm based structured pruning on convolutional layers
for name, module in inr_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name='weight', amount=0.2, n=1, dim=0)
        prune.remove(module, 'weight')  # make pruning permanent
```

**After pruning:**
- Remove zero'd-out channels entirely (true structured pruning).
- Retrain for 5-10 epochs with MSE + task-aware loss to recover accuracy.
- Expected: 15-30% parameter reduction with <5% quality degradation.

**Alternative: Magnitude-based pruning with retraining**
- Set a global threshold based on weight magnitude percentile.
- Zero out weights below threshold.
- Retrain with the pruning mask fixed.

### Step 2: Quantization-Aware Training (QAT)

Simulate low-bit quantization during training so the network learns to be robust to quantization error.

**Target bit-widths:**
- **8-bit:** Simple, well-supported, 75% reduction from FP32.
- **6-bit:** More aggressive, 81% reduction, requires careful implementation.
- **4-bit:** Maximum compression, may significantly degrade quality.

**Implementation using PyTorch QAT:**

```python
# Straight-Through Estimator (STE) for quantization
class QuantizeWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, bits=8):
        # Compute scale and zero point
        w_min, w_max = weight.min(), weight.max()
        scale = (w_max - w_min) / (2**bits - 1)
        zero_point = (-w_min / scale).round()
        # Quantize
        w_quant = (weight / scale + zero_point).round().clamp(0, 2**bits - 1)
        # Dequantize
        w_deq = (w_quant - zero_point) * scale
        return w_deq

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradient unchanged
        return grad_output, None
```

**QAT training procedure:**
1. Insert `QuantizeWeight` operations before every weight usage in the forward pass.
2. Continue training for 10-20 epochs with the task-aware loss.
3. The network learns to place weights at quantization-friendly values.
4. After QAT, extract the actual quantized integers (not dequantized floats).

**Per-channel vs. per-tensor quantization:**
- Per-channel: Each output channel has its own scale/zero-point. Better accuracy, slightly larger metadata.
- Per-tensor: Single scale/zero-point per weight tensor. Simpler, smaller metadata.
- Recommendation: Per-channel for convolutional layers, per-tensor for linear layers.

### Step 3: Entropy Coding

After quantization, the weight values are discrete integers. Apply entropy coding to exploit the non-uniform distribution.

**Method 1: zstd compression (simple)**
```python
import zstd

# Pack quantized weights as bytes
weight_bytes = np.array(quantized_weights, dtype=np.uint8).tobytes()
compressed = zstd.compress(weight_bytes, level=22)  # max compression
```

**Method 2: Arithmetic coding (better compression)**
```python
# Use a range coder with learned probability tables
# Each layer gets its own probability distribution
for layer_name, quantized_weights in model_weights.items():
    histogram = np.bincount(quantized_weights.flatten(), minlength=2**bits)
    probabilities = histogram / histogram.sum()
    compressed_layer = arithmetic_encode(quantized_weights.flatten(), probabilities)
```

**Method 3: ANS (Asymmetric Numeral Systems)**
- Used by modern image codecs (JPEG XL, etc.).
- Very fast decoding, near-optimal compression.
- Python libraries: `constriction`, `craystack`.

**Expected compression ratios:**

| Bit-width | Raw size (3M params) | After entropy coding | Ratio |
|-----------|---------------------|---------------------|-------|
| FP32 | 12.0 MB | 8-10 MB | 0.7-0.8x |
| 8-bit | 3.0 MB | 1.5-2.0 MB | 0.50-0.67x |
| 6-bit | 2.25 MB | 1.0-1.5 MB | 0.44-0.67x |
| 4-bit | 1.5 MB | 0.7-1.0 MB | 0.47-0.67x |

### Target Archive Sizes

| Configuration | Raw Params | Quantized | + Entropy | Rate (/ 37.5MB) |
|---------------|-----------|-----------|-----------|------------------|
| 1.5M params @ 6-bit | 1.13 MB | 1.13 MB | 500-700 KB | 0.013-0.019 |
| 2.0M params @ 6-bit | 1.50 MB | 1.50 MB | 700-1000 KB | 0.019-0.027 |
| 3.0M params @ 6-bit | 2.25 MB | 2.25 MB | 1.0-1.4 MB | 0.027-0.037 |
| 3.0M params @ 8-bit | 3.0 MB | 3.0 MB | 1.5-2.0 MB | 0.040-0.053 |
| 5.0M params @ 8-bit | 5.0 MB | 5.0 MB | 2.5-3.5 MB | 0.067-0.093 |

Rate contribution to score: `25 * rate`.
- 500KB -> 25 * 0.013 = 0.33
- 1.0MB -> 25 * 0.027 = 0.67
- 1.5MB -> 25 * 0.040 = 1.00

### Compression Packaging

```python
import json, struct, zstd

def save_compressed_model(model, config, output_path, bits=6):
    """Save INR model in compressed format."""
    # 1. Extract and quantize all weights
    compressed_layers = {}
    metadata = {'bits': bits, 'layers': {}}

    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy()
        w_min, w_max = w.min(), w.max()
        scale = (w_max - w_min) / (2**bits - 1)
        zero_point = int(round(-w_min / scale))

        # Quantize to integers
        w_int = np.round(w / scale + zero_point).clip(0, 2**bits - 1).astype(np.uint8)

        # Pack bits (for 6-bit, pack 4 values into 3 bytes)
        packed = pack_bits(w_int.flatten(), bits)

        # Entropy code
        compressed = zstd.compress(packed, level=22)

        compressed_layers[name] = compressed
        metadata['layers'][name] = {
            'shape': list(w.shape),
            'scale': float(scale),
            'zero_point': zero_point,
            'compressed_size': len(compressed)
        }

    # 2. Save everything to a single binary file
    with open(output_path, 'wb') as f:
        # Write metadata as JSON + newline
        meta_bytes = json.dumps(metadata).encode('utf-8')
        f.write(struct.pack('<I', len(meta_bytes)))
        f.write(meta_bytes)
        # Write compressed layers
        for name in sorted(compressed_layers.keys()):
            data = compressed_layers[name]
            f.write(struct.pack('<I', len(data)))
            f.write(data)

    # 3. Also save config.json
    with open(output_path.replace('.bin', '_config.json'), 'w') as f:
        json.dump(config, f)
```

---

## Task 4: Inference / Decode Pipeline

The `inflate.py` script must generate all 1200 frames from the compressed model weights.

### Timing Budget (30 Minutes on T4)

| Step | Time (estimated) |
|------|-----------------|
| Decompress weights (zstd) | 1-5 seconds |
| Dequantize weights | 1-2 seconds |
| Instantiate model | <1 second |
| Generate 1200 frames @ 512x384 | 6-24 seconds (50-200 fps) |
| Bicubic upscale 512x384 -> 1164x874 | 12-36 seconds |
| Write .raw file (3.6 GB) | 30-60 seconds |
| **Total** | **~1-2 minutes** |

This is well within the 30-minute limit.

### Inference Script Structure

```python
#!/usr/bin/env python
"""inflate.py -- Reconstruct video frames from compressed INR model weights."""
import sys, os, json, struct, torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

# Can import from repo files available at eval time
from frame_utils import camera_size  # (1164, 874)

# ---- INR Model Definition ----
# (Must be self-contained or imported from a file included in the submission)

class HiNeRVBlock(torch.nn.Module):
    """Single upsampling block for HiNeRV decoder."""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels * scale_factor**2,
                                      kernel_size=3, padding=1)
        self.pixel_shuffle = torch.nn.PixelShuffle(scale_factor)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels,
                                      kernel_size=3, padding=1)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.act(self.pixel_shuffle(self.conv1(x)))
        x = self.act(self.conv2(x))
        return x


class HiNeRVModel(torch.nn.Module):
    """Hierarchical Neural Representation for Video."""
    def __init__(self, config):
        super().__init__()
        self.num_frames = config['num_frames']
        embed_dim = config['embed_dim']
        channels = config['channels']  # list of channel widths per block
        init_h, init_w = config['init_size']  # e.g., (6, 8)

        # Frame embeddings
        self.frame_embed = torch.nn.Embedding(self.num_frames, embed_dim)

        # Map embedding to spatial feature map
        self.stem = torch.nn.Linear(embed_dim, channels[0] * init_h * init_w)
        self.init_h = init_h
        self.init_w = init_w

        # Upsampling blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(HiNeRVBlock(channels[i], channels[i+1]))

        # Output head
        self.head = torch.nn.Conv2d(channels[-1], 3, kernel_size=1)

    def forward(self, frame_indices):
        """Generate frames for given indices.

        Args:
            frame_indices: (B,) int tensor of frame indices
        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        embed = self.frame_embed(frame_indices)  # (B, embed_dim)
        x = self.stem(embed)  # (B, C0 * H0 * W0)
        x = x.view(-1, self.blocks[0].conv1.in_channels if self.blocks
                    else 3, self.init_h, self.init_w)

        for block in self.blocks:
            x = block(x)

        x = torch.sigmoid(self.head(x)) * 255.0
        return x


def load_compressed_model(model_path, config_path, device):
    """Load and decompress INR model."""
    import zstd

    with open(config_path, 'r') as f:
        config = json.load(f)

    model = HiNeRVModel(config).to(device)

    with open(model_path, 'rb') as f:
        # Read metadata
        meta_size = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(meta_size).decode('utf-8'))
        bits = metadata['bits']

        # Read and decompress each layer
        state_dict = {}
        for name in sorted(metadata['layers'].keys()):
            layer_info = metadata['layers'][name]
            comp_size = struct.unpack('<I', f.read(4))[0]
            compressed = f.read(comp_size)

            # Decompress
            packed = zstd.decompress(compressed)

            # Unpack bits
            w_int = unpack_bits(packed, bits, np.prod(layer_info['shape']))

            # Dequantize
            scale = layer_info['scale']
            zero_point = layer_info['zero_point']
            w_float = (w_int.astype(np.float32) - zero_point) * scale

            state_dict[name] = torch.tensor(
                w_float.reshape(layer_info['shape']),
                device=device
            )

    model.load_state_dict(state_dict)
    return model


def main():
    archive_dir = Path(sys.argv[1])
    inflated_dir = Path(sys.argv[2])
    video_names_file = Path(sys.argv[3])

    inflated_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_compressed_model(
        archive_dir / 'model.bin',
        archive_dir / 'config.json',
        device
    )
    model.eval()

    with open(video_names_file) as f:
        video_names = [line.strip() for line in f]

    W, H = camera_size  # (1164, 874)

    for video_name in video_names:
        raw_name = Path(video_name).with_suffix('.raw')
        raw_path = inflated_dir / raw_name

        with open(raw_path, 'wb') as out_f:
            with torch.no_grad():
                # Generate frames in batches
                batch_size = 16
                for start in range(0, 1200, batch_size):
                    end = min(start + batch_size, 1200)
                    indices = torch.arange(start, end, device=device)

                    # Generate at 512x384
                    frames = model(indices)  # (B, 3, 384, 512)

                    # Upscale to 1164x874
                    frames = F.interpolate(
                        frames, size=(H, W),
                        mode='bicubic', align_corners=False
                    )

                    # Convert to uint8
                    frames = frames.clamp(0, 255).round().to(torch.uint8)

                    # Rearrange to (B, H, W, 3) and write
                    frames = frames.permute(0, 2, 3, 1).cpu().numpy()
                    out_f.write(frames.tobytes())

        print(f"Generated {raw_path} ({os.path.getsize(raw_path):,} bytes)")


if __name__ == '__main__':
    main()
```

### inflate.sh

```bash
#!/bin/bash
set -e
ARCHIVE_DIR="$1"
INFLATED_DIR="$2"
VIDEO_NAMES="$3"
python submissions/<submission_name>/inflate.py "$ARCHIVE_DIR" "$INFLATED_DIR" "$VIDEO_NAMES"
```

### Environment Constraints

The inflate.py runs in the CI evaluation environment with these available:
- **Python packages** (from `pyproject.toml` / `uv sync`): `torch`, `numpy`, `safetensors`, `av`, `timm`, `einops`, `segmentation_models_pytorch`, `tqdm`, `Pillow`
- **Repo files:** `frame_utils.py`, `modules.py`, `models/segnet.safetensors`, `models/posenet.safetensors`, `videos/0.mkv`
- **Custom files:** Any `.py` files added by the PR (e.g., in the submission directory)
- **NOTE:** For this approach, SegNet/PoseNet are NOT needed at decode time.

If the INR architecture requires custom layers (e.g., SIREN sine activations, hash grid lookups), they must be defined in:
1. The `inflate.py` file itself, OR
2. A separate `.py` file included in the submission directory.

Standard PyTorch layers (`nn.Conv2d`, `nn.Linear`, `nn.GELU`, `nn.PixelShuffle`, etc.) are all available.

For weight decompression, if using `zstd`:
- The `zstd` Python package may need to be available. Check `pyproject.toml` dependencies.
- Alternative: use Python's built-in `lzma` or `gzip` modules (slightly worse compression but zero dependencies).
- Alternative: use `zipfile` (the archive is already a .zip, so layers can be stored as zip entries with DEFLATE compression).

---

## Task 5: Archive Format and Submission Structure

### archive.zip Contents

```
archive.zip
  |-- model.bin          # Compressed, quantized INR weights (~500KB-1.5MB)
  |-- config.json        # Architecture configuration (~200 bytes)
```

**config.json example:**
```json
{
    "num_frames": 1200,
    "embed_dim": 128,
    "channels": [128, 96, 64, 48, 32, 24, 16],
    "init_size": [6, 8],
    "working_resolution": [384, 512],
    "output_resolution": [874, 1164]
}
```

**model.bin format:**
```
[4 bytes: metadata JSON length]
[metadata JSON bytes]
[4 bytes: layer 1 compressed size]
[layer 1 compressed bytes]
[4 bytes: layer 2 compressed size]
[layer 2 compressed bytes]
...
```

The metadata JSON contains per-layer quantization parameters:
```json
{
    "bits": 6,
    "layers": {
        "frame_embed.weight": {
            "shape": [1200, 128],
            "scale": 0.00123,
            "zero_point": 32,
            "compressed_size": 45678
        },
        "stem.weight": {
            "shape": [6144, 128],
            "scale": 0.00456,
            "zero_point": 31,
            "compressed_size": 23456
        }
    }
}
```

### Submission Directory Structure

```
submissions/phase3_inr/
  |-- inflate.sh          # Calls inflate.py
  |-- inflate.py          # Loads model, generates frames, writes .raw
  |-- compress.sh         # (Optional) Trains INR, quantizes, packages archive
  |-- train_inr.py        # (Optional) Full training script
  |-- inr_model.py        # (Optional) Model definition (shared between train and inflate)
```

### Alternative: Use safetensors for Weights

Instead of a custom binary format, consider using `safetensors` (already a dependency):

```python
from safetensors.torch import save_file, load_file

# Save quantized weights as uint8 tensors
quantized_state = {}
for name, param in model.named_parameters():
    q = quantize_to_uint8(param)
    quantized_state[name] = q

save_file(quantized_state, 'model.safetensors')
# Then compress with zip
```

This is simpler but may not achieve the best compression since safetensors doesn't support sub-byte packing. Best used with 8-bit quantization.

---

## Expected Score Breakdown

### With MSE-Only Training

The network reconstructs frames with good pixel fidelity but does not specifically optimize for SegNet/PoseNet metrics.

| Config | SegNet Dist | PoseNet Dist | Rate | Score |
|--------|------------|-------------|------|-------|
| 1.5M @ 6-bit (~600KB) | ~0.015 | ~0.40 | 0.016 | 100*0.015 + sqrt(10*0.40) + 25*0.016 = 1.5 + 2.0 + 0.4 = **3.9** |
| 2.0M @ 6-bit (~900KB) | ~0.012 | ~0.38 | 0.024 | 100*0.012 + sqrt(10*0.38) + 25*0.024 = 1.2 + 1.95 + 0.6 = **3.75** |
| 3.0M @ 6-bit (~1.2MB) | ~0.010 | ~0.35 | 0.032 | 100*0.010 + sqrt(10*0.35) + 25*0.032 = 1.0 + 1.87 + 0.8 = **3.67** |

These scores are worse than the current leader (2.90) because MSE training wastes capacity on pixel details irrelevant to the scoring function.

### With Task-Aware Training (Cross-Entropy + PoseNet MSE)

The network specifically optimizes for SegNet class preservation and PoseNet motion vector accuracy.

| Config | SegNet Dist | PoseNet Dist | Rate | Score |
|--------|------------|-------------|------|-------|
| 1.5M @ 6-bit (~600KB) | ~0.005 | ~0.50 | 0.016 | 100*0.005 + sqrt(10*0.50) + 25*0.016 = 0.5 + 2.24 + 0.4 = **3.14** |
| 2.0M @ 6-bit (~900KB) | ~0.004 | ~0.42 | 0.024 | 100*0.004 + sqrt(10*0.42) + 25*0.024 = 0.4 + 2.05 + 0.6 = **3.05** |
| 3.0M @ 6-bit (~1.2MB) | ~0.003 | ~0.40 | 0.032 | 100*0.003 + sqrt(10*0.40) + 25*0.032 = 0.3 + 2.0 + 0.8 = **3.1** |
| 3.0M @ 4-bit (~800KB) | ~0.005 | ~0.45 | 0.021 | 100*0.005 + sqrt(10*0.45) + 25*0.021 = 0.5 + 2.12 + 0.53 = **3.15** |

### With Aggressive Task-Aware Training + Optimized Architecture

Best-case scenario with architectural innovations (e.g., attention-based frame embeddings, task-specific capacity allocation):

| Config | SegNet Dist | PoseNet Dist | Rate | Score |
|--------|------------|-------------|------|-------|
| 2.5M @ 6-bit (~1.0MB) | ~0.002 | ~0.30 | 0.027 | 100*0.002 + sqrt(10*0.30) + 25*0.027 = 0.2 + 1.73 + 0.67 = **2.6** |
| 3.0M @ 5-bit (~1.0MB) | ~0.002 | ~0.25 | 0.027 | 100*0.002 + sqrt(10*0.25) + 25*0.027 = 0.2 + 1.58 + 0.67 = **2.45** |
| 4.0M @ 5-bit (~1.3MB) | ~0.001 | ~0.20 | 0.035 | 100*0.001 + sqrt(10*0.20) + 25*0.035 = 0.1 + 1.41 + 0.87 = **2.38** |

These optimistic estimates assume the task-aware loss successfully redirects capacity toward SegNet boundaries and PoseNet features. Actual results will depend heavily on architecture choice and training hyperparameters.

### Score Sensitivity to Architecture Choices

The optimal parameter count involves a tradeoff:
- **Too few parameters** (~1M): Poor reconstruction quality, high SegNet/PoseNet distortion.
- **Too many parameters** (~5M+): Good quality but high rate contribution.
- **Sweet spot** (~2-3M @ 5-6 bit): Balanced rate-distortion.

The rate term is `25 * archive_size / 37,545,489`:
- 500KB: 25 * 0.013 = 0.33
- 750KB: 25 * 0.020 = 0.50
- 1.0MB: 25 * 0.027 = 0.67
- 1.5MB: 25 * 0.040 = 1.00
- 2.0MB: 25 * 0.053 = 1.33

---

## Risks and Mitigations

### Risk 1: Training Time

**Problem:** Training an INR to convergence takes 3-8+ hours on GPU. Task-aware fine-tuning adds 5-10x per-step cost due to SegNet/PoseNet forward passes.

**Mitigation:**
- Use MSE pre-training to get a good initialization quickly (1-2 hours).
- Fine-tune with task-aware loss on a subset of frames initially, then all frames.
- Use mixed precision (bfloat16/float16) for the frozen SegNet/PoseNet to reduce memory and speed up their forward passes.
- Gradient accumulation: process 1-2 frame pairs per step, accumulate gradients over 4-8 steps.

### Risk 2: Hidden Test Set

**Problem:** If the final evaluation uses a different video than `0.mkv`, an INR trained on `0.mkv` will produce garbage.

**Mitigation:**
- The `compress.sh` script should train the INR on whatever input video is provided, making it generalizable to any video.
- The training script takes the input video path as an argument and trains from scratch.
- Training must complete within a reasonable time (few hours).
- This is consistent with the challenge rules: "You can use anything for compression."

**Note:** The current leaderboard evaluates on `0.mkv` only (`public_test_video_names.txt` contains `0.mkv`). The hidden test may use the same or different videos.

### Risk 3: Quantization Degradation

**Problem:** Aggressive quantization (4-6 bit) may cause unacceptable quality loss, especially at SegNet boundaries.

**Mitigation:**
- Use quantization-aware training (QAT) -- do not post-hoc quantize.
- Start with 8-bit and progressively reduce to 6-bit or 4-bit while monitoring quality.
- Use per-channel quantization for convolutional layers (finer granularity).
- Some layers are more sensitive to quantization than others. Use mixed precision: 8-bit for sensitive layers (output head, frame embeddings), 4-bit for insensitive layers (middle decoder blocks).

### Risk 4: Architecture Choice

**Problem:** The wrong architecture can lead to a poor rate-distortion tradeoff. Too many architectural experiments waste time.

**Mitigation:**
- Start with HiNeRV (proven architecture for video INR).
- Benchmark on a small subset (100 frames) first to validate the approach.
- Use ablation studies: vary parameter count systematically (1M, 2M, 3M, 5M) and measure quality at each.

### Risk 5: SegNet Proxy Loss Mismatch

**Problem:** Cross-entropy loss is a proxy for argmax disagreement. The proxy may not perfectly correlate with the actual metric.

**Mitigation:**
- Periodically evaluate the actual argmax disagreement metric during training (non-differentiable, but can be computed for monitoring).
- Experiment with multiple proxy losses (cross-entropy, margin loss, soft argmax) and choose the one that best correlates.
- The pixel MSE regularization term (`gamma * mse_loss`) provides a safety net -- even if the proxy loss is imperfect, reasonable pixel quality ensures reasonable task metric quality.

### Risk 6: PoseNet Sensitivity

**Problem:** PoseNet operates on YUV6 features at half resolution (192x256). Subtle color/luminance shifts from the INR may disproportionately affect PoseNet output.

**Mitigation:**
- Include PoseNet loss in training from the start of fine-tuning.
- Monitor PoseNet distortion per frame pair during training.
- Consider training in YUV space rather than RGB space if PoseNet distortion is stubborn.
- Ensure the INR output range exactly matches the expected [0, 255] float range.

---

## Comparison to Phase 2 (Adversarial Decode)

| Aspect | Phase 2 (Adversarial) | Phase 3 (INR) |
|--------|----------------------|---------------|
| **Archive contents** | Seg maps + pose vectors (~1-3MB) | Neural network weights (~0.5-1.5MB) |
| **Decode mechanism** | Gradient descent at decode time | Single forward pass per frame |
| **Requires SegNet/PoseNet at decode?** | YES | NO |
| **Decode time** | 10-25 minutes | 1-2 minutes |
| **SegNet distortion** | Near-zero (exact argmax matching) | Approximate (depends on training quality) |
| **PoseNet distortion** | Moderate (hard to match exactly) | Moderate (similar challenge) |
| **Implementation complexity** | High (gradient descent + reconstruction) | Medium (training pipeline + compression) |
| **Generalization** | Works for any video (just store targets) | Must retrain for each video |
| **Rate efficiency** | Storing targets is larger than INR weights | Very compact representation |
| **Risk profile** | Lower risk for SegNet, higher for PoseNet | Moderate risk for both |

**Key tradeoffs:**
- Phase 2 nearly guarantees perfect SegNet scores but is slow at decode time and requires the evaluation models.
- Phase 3 is faster at decode and does not need evaluation models, but SegNet performance is approximate.
- Phase 3 can achieve much smaller file sizes (500KB vs. 1-3MB), giving a significant rate advantage.
- Both approaches can be combined: use an INR for the base reconstruction + adversarial refinement for SegNet boundaries (hybrid approach).

---

## Implementation Roadmap

### Week 1: Proof of Concept

**Day 1-2:**
- [ ] Set up the training pipeline: decode video, prepare frame tensors at 512x384.
- [ ] Implement a basic NeRV/HiNeRV model (start with ~500K params for fast iteration).
- [ ] Train with MSE loss on a subset of 100 frames.
- [ ] Verify the model can memorize 100 frames at reasonable PSNR (>28 dB).

**Day 3-4:**
- [ ] Scale up to full 1200 frames.
- [ ] Experiment with model sizes: 500K, 1M, 2M, 3M parameters.
- [ ] Implement the decode pipeline: generate frames, upscale, write .raw.
- [ ] Run `evaluate.py` on the MSE-trained output to get baseline scores.

**Day 5-7:**
- [ ] Implement weight quantization (start with 8-bit post-training quantization).
- [ ] Implement zstd compression of quantized weights.
- [ ] Package as archive.zip and verify end-to-end pipeline works.
- [ ] Measure actual file sizes and rate contributions.

### Week 2: Task-Aware Training

**Day 8-9:**
- [ ] Pre-compute SegNet ground truth argmax maps for all 600 odd frames.
- [ ] Pre-compute PoseNet ground truth pose vectors for all 600 frame pairs.
- [ ] Implement cross-entropy SegNet proxy loss.
- [ ] Implement PoseNet MSE loss with proper preprocessing (YUV6 conversion).

**Day 10-12:**
- [ ] Run task-aware fine-tuning on the MSE-pretrained model.
- [ ] Tune loss weights (alpha, beta, gamma).
- [ ] Monitor actual competition metrics during training.
- [ ] Compare with MSE-only baseline.

**Day 13-14:**
- [ ] Implement quantization-aware training (QAT) with 6-bit weights.
- [ ] Implement structured pruning.
- [ ] Optimize the compression pipeline (try ANS coding vs. zstd).
- [ ] Produce final submission candidates.

### Week 3: Optimization and Submission

**Day 15-17:**
- [ ] Architecture search: try different block designs, channel widths, embedding dimensions.
- [ ] Hyperparameter sweep: learning rates, loss weights, quantization bit-widths.
- [ ] Find the Pareto-optimal configuration (best score at each file size).

**Day 18-19:**
- [ ] Final training run with best configuration.
- [ ] Verify decode pipeline on both CPU and GPU environments.
- [ ] Run full evaluation locally to confirm score.

**Day 20-21:**
- [ ] Package final submission.
- [ ] Create PR with archive.zip download link, inflate.sh, report.txt.
- [ ] Write up the approach for the "best write-up" prize.

---

## Appendix A: Key File Paths and Constants

### Repository File Paths

| File | Purpose |
|------|---------|
| `evaluate.py` | Competition evaluation script |
| `evaluate.sh` | Evaluation shell wrapper |
| `modules.py` | SegNet, PoseNet, DistortionNet definitions |
| `frame_utils.py` | Video loading, YUV conversion, dataset classes |
| `models/segnet.safetensors` | SegNet weights (38 MB) |
| `models/posenet.safetensors` | PoseNet weights (56 MB) |
| `videos/0.mkv` | Test video (37.5 MB, 1200 frames) |
| `public_test_video_names.txt` | Contains "0.mkv" |
| `public_test_segments.txt` | Contains source segment ID |
| `pyproject.toml` | Python dependencies |

### Critical Constants

```python
# From frame_utils.py
seq_len = 2                           # Frames per sequence for evaluation
camera_size = (1164, 874)             # (width, height) -- output resolution
segnet_model_input_size = (512, 384)  # (width, height) -- model input resolution

# From modules.py
BN_EPS = 0.001
BN_MOM = 0.01
VISION_FEATURES = 2048
SUMMARY_FEATURES = 512
IN_CHANS = 12                         # 6 YUV channels x 2 frames
HEADS = [Head('pose', 32, 12)]        # pose head: hidden=32, out=12

# Derived
TOTAL_FRAMES = 1200
FRAME_PAIRS = 600                     # 1200 / 2
SEGNET_FRAMES = 600                   # odd frames: 1, 3, 5, ..., 1199
RAW_FRAME_BYTES = 874 * 1164 * 3      # = 3,050,928 bytes per frame
RAW_TOTAL_BYTES = RAW_FRAME_BYTES * 1200  # = 3,661,113,600 bytes (~3.6 GB)
ORIGINAL_FILE_SIZE = 37_545_489       # bytes (for rate computation)
```

### Scoring Formula (Exact)

```python
# From evaluate.py:92
score = 100 * segnet_dist + math.sqrt(posenet_dist * 10) + 25 * rate

# Where:
# segnet_dist = avg fraction of pixels with argmax disagreement (600 odd frames)
# posenet_dist = avg MSE of first 6 pose dims (600 frame pairs)
# rate = archive.zip size / 37,545,489
```

---

## Appendix B: Reference Pseudocode

### End-to-End Training Script (Pseudocode)

```python
#!/usr/bin/env python
"""train_inr.py -- Train an INR to represent a video for the comma compression challenge."""

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from pathlib import Path
from modules import SegNet, PoseNet
from safetensors.torch import load_file
from frame_utils import camera_size, segnet_model_input_size

# ---- Configuration ----
CONFIG = {
    'num_frames': 1200,
    'embed_dim': 128,
    'channels': [128, 96, 64, 48, 32, 24, 16],
    'init_size': (6, 8),
    'working_resolution': (384, 512),  # (H, W)
}

TRAIN_CONFIG = {
    'mse_epochs': 50,
    'task_epochs': 20,
    'batch_size': 8,
    'lr_mse': 5e-4,
    'lr_task': 1e-4,
    'alpha': 10.0,   # segnet loss weight
    'beta': 1.0,     # posenet loss weight
    'gamma': 0.1,    # pixel mse weight during task-aware phase
}

QUANT_CONFIG = {
    'bits': 6,
    'qat_epochs': 10,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- Step 1: Prepare data ----
def load_all_frames(video_path):
    """Decode video and return all frames at working resolution."""
    import av
    from frame_utils import yuv420_to_rgb

    container = av.open(str(video_path))
    frames = []
    for frame in container.decode(container.streams.video[0]):
        rgb = yuv420_to_rgb(frame)  # (H, W, 3) uint8
        # Convert to (3, H, W) float and resize to working resolution
        rgb_chw = rgb.permute(2, 0, 1).float().unsqueeze(0)  # (1, 3, H, W)
        rgb_resized = F.interpolate(
            rgb_chw,
            size=CONFIG['working_resolution'],
            mode='bilinear',
            align_corners=False
        )
        frames.append(rgb_resized.squeeze(0))  # (3, 384, 512)
    container.close()
    return torch.stack(frames)  # (1200, 3, 384, 512)


def precompute_segnet_targets(all_frames, segnet):
    """Compute argmax class maps for all odd frames."""
    targets = {}
    with torch.no_grad():
        for i in range(1, 1200, 2):  # odd frames: 1, 3, 5, ..., 1199
            frame = all_frames[i].unsqueeze(0).to(device)  # (1, 3, 384, 512)
            logits = segnet(frame)  # (1, 5, 384, 512)
            targets[i] = logits.argmax(dim=1).squeeze(0).cpu()  # (384, 512)
    return targets


def precompute_posenet_targets(all_frames, posenet):
    """Compute pose vectors for all frame pairs."""
    targets = {}
    with torch.no_grad():
        for i in range(0, 1200, 2):
            pair = torch.stack([all_frames[i], all_frames[i+1]], dim=0)  # (2, 3, 384, 512)
            pair = pair.unsqueeze(0).to(device)  # (1, 2, 3, 384, 512)
            posenet_in = posenet.preprocess_input(pair)  # (1, 12, 192, 256)
            out = posenet(posenet_in)  # {'pose': (1, 12)}
            targets[i // 2] = out['pose'][0, :6].cpu()  # (6,)
    return targets


# ---- Step 2: Phase A - MSE pre-training ----
def train_mse(model, all_frames, config):
    """Pre-train with pixel MSE loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_mse'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['mse_epochs']
    )

    indices = torch.arange(1200)

    for epoch in range(config['mse_epochs']):
        perm = torch.randperm(1200)
        epoch_loss = 0.0

        for start in range(0, 1200, config['batch_size']):
            batch_idx = perm[start:start + config['batch_size']].to(device)
            batch_gt = all_frames[batch_idx.cpu()].to(device)

            batch_pred = model(batch_idx)
            loss = F.mse_loss(batch_pred, batch_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / (1200 // config['batch_size'])
        psnr = 10 * np.log10(255**2 / avg_loss) if avg_loss > 0 else float('inf')
        print(f"Epoch {epoch+1}/{config['mse_epochs']}: MSE={avg_loss:.2f}, PSNR={psnr:.1f} dB")


# ---- Step 3: Phase B - Task-aware fine-tuning ----
def train_task_aware(model, all_frames, segnet, posenet,
                     segnet_targets, posenet_targets, config):
    """Fine-tune with task-aware loss."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_task'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['task_epochs']
    )

    for epoch in range(config['task_epochs']):
        # Iterate over frame pairs
        pair_indices = list(range(600))
        np.random.shuffle(pair_indices)

        epoch_seg_loss = 0.0
        epoch_pose_loss = 0.0
        epoch_mse_loss = 0.0

        for pair_idx in pair_indices:
            frame_a_idx = pair_idx * 2
            frame_b_idx = pair_idx * 2 + 1

            # Generate frames
            indices = torch.tensor([frame_a_idx, frame_b_idx], device=device)
            frames_pred = model(indices)  # (2, 3, 384, 512)
            frames_gt = all_frames[[frame_a_idx, frame_b_idx]].to(device)

            # Pixel MSE loss (regularization)
            mse_loss = F.mse_loss(frames_pred, frames_gt)

            # SegNet loss (on odd frame = frame_b)
            seg_target = segnet_targets[frame_b_idx].to(device)  # (384, 512)
            seg_logits = segnet(frames_pred[1:2])  # (1, 5, 384, 512)
            seg_loss = F.cross_entropy(seg_logits, seg_target.unsqueeze(0))

            # PoseNet loss
            pair_pred = frames_pred.unsqueeze(0)  # (1, 2, 3, 384, 512)
            posenet_in = posenet.preprocess_input(pair_pred)
            pose_out = posenet(posenet_in)
            pose_target = posenet_targets[pair_idx].to(device)
            pose_loss = F.mse_loss(pose_out['pose'][0, :6], pose_target)

            # Combined loss
            total_loss = (config['alpha'] * seg_loss +
                         config['beta'] * pose_loss +
                         config['gamma'] * mse_loss)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_seg_loss += seg_loss.item()
            epoch_pose_loss += pose_loss.item()
            epoch_mse_loss += mse_loss.item()

        scheduler.step()
        n = len(pair_indices)
        print(f"Epoch {epoch+1}/{config['task_epochs']}: "
              f"SegCE={epoch_seg_loss/n:.4f}, "
              f"PoseMSE={epoch_pose_loss/n:.6f}, "
              f"PixMSE={epoch_mse_loss/n:.2f}")


# ---- Step 4: Quantize and save ----
def quantize_and_save(model, config, output_dir, bits=6):
    """Quantize model weights and save compressed."""
    import zstd, json, struct

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {'bits': bits, 'layers': {}}
    compressed_data = {}

    total_raw = 0
    total_compressed = 0

    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy().astype(np.float32)
        w_min, w_max = w.min(), w.max()

        if w_max - w_min < 1e-10:
            scale = 1.0
        else:
            scale = (w_max - w_min) / (2**bits - 1)

        zero_point = int(round(-w_min / scale))
        w_int = np.round(w / scale + zero_point).clip(0, 2**bits - 1).astype(np.uint8)

        raw_bytes = w_int.tobytes()
        compressed = zstd.compress(raw_bytes, 22)

        total_raw += len(raw_bytes)
        total_compressed += len(compressed)

        compressed_data[name] = compressed
        metadata['layers'][name] = {
            'shape': list(w.shape),
            'scale': float(scale),
            'zero_point': zero_point,
        }

    # Save model.bin
    with open(output_dir / 'model.bin', 'wb') as f:
        meta_bytes = json.dumps(metadata).encode('utf-8')
        f.write(struct.pack('<I', len(meta_bytes)))
        f.write(meta_bytes)
        for name in sorted(compressed_data.keys()):
            data = compressed_data[name]
            f.write(struct.pack('<I', len(data)))
            f.write(data)

    # Save config.json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    model_size = (output_dir / 'model.bin').stat().st_size
    config_size = (output_dir / 'config.json').stat().st_size
    print(f"Raw quantized: {total_raw:,} bytes")
    print(f"Compressed: {total_compressed:,} bytes")
    print(f"model.bin: {model_size:,} bytes")
    print(f"config.json: {config_size:,} bytes")
    print(f"Total archive (approx): {model_size + config_size:,} bytes")
    print(f"Rate: {(model_size + config_size) / 37_545_489:.6f}")


# ---- Main ----
def main():
    # Load video
    print("Loading video frames...")
    all_frames = load_all_frames('videos/0.mkv')  # (1200, 3, 384, 512)

    # Load evaluation models (frozen)
    print("Loading SegNet and PoseNet...")
    segnet = SegNet().eval().to(device)
    segnet.load_state_dict(load_file('models/segnet.safetensors', device=str(device)))
    for p in segnet.parameters():
        p.requires_grad = False

    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file('models/posenet.safetensors', device=str(device)))
    for p in posenet.parameters():
        p.requires_grad = False

    # Pre-compute targets
    print("Pre-computing SegNet targets...")
    segnet_targets = precompute_segnet_targets(all_frames, segnet)

    print("Pre-computing PoseNet targets...")
    posenet_targets = precompute_posenet_targets(all_frames, posenet)

    # Build model
    print("Building INR model...")
    model = HiNeRVModel(CONFIG).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Phase A: MSE pre-training
    print("\n=== Phase A: MSE Pre-Training ===")
    train_mse(model, all_frames, TRAIN_CONFIG)

    # Phase B: Task-aware fine-tuning
    print("\n=== Phase B: Task-Aware Fine-Tuning ===")
    train_task_aware(model, all_frames, segnet, posenet,
                     segnet_targets, posenet_targets, TRAIN_CONFIG)

    # Quantize and save
    print("\n=== Quantizing and Saving ===")
    quantize_and_save(model, CONFIG, 'submissions/phase3_inr/archive/',
                      bits=QUANT_CONFIG['bits'])

    print("\nDone! Package archive/ directory into archive.zip")


if __name__ == '__main__':
    main()
```

### Bit Packing Utilities (For Sub-Byte Quantization)

```python
def pack_bits(values, bits):
    """Pack array of integers (each 0..2^bits-1) into bytes.

    For 6-bit: packs 4 values into 3 bytes.
    For 4-bit: packs 2 values into 1 byte.
    For 8-bit: no packing needed.
    """
    values = np.array(values, dtype=np.uint8)

    if bits == 8:
        return values.tobytes()
    elif bits == 4:
        # Pack pairs of 4-bit values into bytes
        if len(values) % 2:
            values = np.append(values, 0)
        packed = (values[0::2] << 4) | values[1::2]
        return packed.tobytes()
    elif bits == 6:
        # Pack 4 values (24 bits) into 3 bytes
        n = len(values)
        pad = (4 - n % 4) % 4
        if pad:
            values = np.append(values, np.zeros(pad, dtype=np.uint8))
        # Reshape to groups of 4
        groups = values.reshape(-1, 4)
        # Pack: [a(6) b(6) c(6) d(6)] -> [a(6)+b_hi(2)] [b_lo(4)+c_hi(4)] [c_lo(2)+d(6)]
        byte0 = (groups[:, 0] << 2) | (groups[:, 1] >> 4)
        byte1 = ((groups[:, 1] & 0x0F) << 4) | (groups[:, 2] >> 2)
        byte2 = ((groups[:, 2] & 0x03) << 6) | groups[:, 3]
        packed = np.stack([byte0, byte1, byte2], axis=1).flatten()
        return packed.astype(np.uint8).tobytes()
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


def unpack_bits(packed_bytes, bits, num_values):
    """Unpack bytes back to array of integers."""
    data = np.frombuffer(packed_bytes, dtype=np.uint8)

    if bits == 8:
        return data[:num_values]
    elif bits == 4:
        hi = data >> 4
        lo = data & 0x0F
        values = np.stack([hi, lo], axis=1).flatten()
        return values[:num_values]
    elif bits == 6:
        # Unpack 3 bytes -> 4 values of 6 bits each
        groups = data.reshape(-1, 3)
        v0 = groups[:, 0] >> 2
        v1 = ((groups[:, 0] & 0x03) << 4) | (groups[:, 1] >> 4)
        v2 = ((groups[:, 1] & 0x0F) << 2) | (groups[:, 2] >> 6)
        v3 = groups[:, 2] & 0x3F
        values = np.stack([v0, v1, v2, v3], axis=1).flatten()
        return values[:num_values]
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
```

### Quick Evaluation Helper (For Development)

```python
def quick_evaluate(model, all_frames_fullres, segnet, posenet, device):
    """Compute competition score without writing .raw files.

    Args:
        model: trained INR model
        all_frames_fullres: (1200, 3, 874, 1164) uint8 tensor (original resolution)
        segnet: loaded SegNet model (frozen)
        posenet: loaded PoseNet model (frozen)
    """
    import math
    from frame_utils import camera_size

    model.eval()
    segnet_dists = []
    posenet_dists = []

    with torch.no_grad():
        for pair_idx in range(600):
            i = pair_idx * 2

            # Generate frames at working resolution
            indices = torch.tensor([i, i+1], device=device)
            frames_pred = model(indices)  # (2, 3, 384, 512)

            # Upscale to full resolution
            frames_pred_full = F.interpolate(
                frames_pred, size=(874, 1164),
                mode='bicubic', align_corners=False
            ).clamp(0, 255).round()

            # Prepare as evaluation expects: (1, 2, H, W, 3) uint8
            pred_batch = frames_pred_full.permute(0, 2, 3, 1).unsqueeze(0).to(torch.uint8)
            gt_batch = all_frames_fullres[[i, i+1]].unsqueeze(0).to(device)

            # Compute distortion using DistortionNet
            # (simplified -- uses modules.py DistortionNet interface)
            posenet_in_gt, segnet_in_gt = preprocess(gt_batch)
            posenet_in_pred, segnet_in_pred = preprocess(pred_batch)

            seg_out_gt = segnet(segnet_in_gt)
            seg_out_pred = segnet(segnet_in_pred)
            seg_dist = segnet.compute_distortion(seg_out_gt, seg_out_pred)

            pose_out_gt = posenet(posenet_in_gt)
            pose_out_pred = posenet(posenet_in_pred)
            pose_dist = posenet.compute_distortion(pose_out_gt, pose_out_pred)

            segnet_dists.append(seg_dist.item())
            posenet_dists.append(pose_dist.item())

    avg_seg = np.mean(segnet_dists)
    avg_pose = np.mean(posenet_dists)

    print(f"SegNet distortion: {avg_seg:.8f}")
    print(f"PoseNet distortion: {avg_pose:.8f}")
    print(f"Score (excl. rate): {100 * avg_seg + math.sqrt(10 * avg_pose):.4f}")

    return avg_seg, avg_pose
```

---

## Summary

Phase 3 (Implicit Neural Representation) encodes the entire video as a compact neural network. The network is overfitted to the specific video during compression, then its quantized and entropy-coded weights serve as the compressed file. At decode time, simple forward passes regenerate all 1200 frames.

The critical innovation is **task-aware training**: instead of optimizing for pixel MSE, the network is trained with the exact competition scoring function (SegNet cross-entropy + PoseNet MSE). This reallocates network capacity from pixel-level fidelity to semantically meaningful features -- class boundaries and motion vectors -- that the competition actually measures.

Target score: **2.0-2.5** with a well-tuned 2-3M parameter HiNeRV at 5-6 bit quantization + entropy coding, yielding a ~1MB archive.

Key dependencies on getting right:
1. **Architecture choice** -- HiNeRV-style hierarchical decoder is the recommended starting point.
2. **Task-aware loss** -- Cross-entropy for SegNet, MSE for PoseNet, with proper gradient flow through frozen evaluation models.
3. **Quantization pipeline** -- QAT at 6-bit with per-channel scales and entropy coding.
4. **The rate-quality tradeoff** -- finding the parameter count sweet spot where further parameters do not improve the score due to increased rate.
