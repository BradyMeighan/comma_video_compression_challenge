# mask2mask: Complete Architectural Breakdown of the 0.60 Submission

## Background

The **mask2mask** submission to the comma.ai video compression challenge achieves a score of **0.60**, dramatically beating all prior approaches (previous best ~1.20). This document is a complete reverse-engineering of the approach based on the published submission code.

**The challenge:** Compress a 1200-frame, 1164×874, 20fps dashcam video (`0.mkv`, ~37.5 MB) such that the decompressed output produces minimal disagreement on two evaluation networks:
- **SegNet** (smp.Unet with EfficientNet-B2 backbone, 5-class semantic segmentation, evaluated at 384×512 on the LAST frame of each pair only)
- **PoseNet** (FastViT-T12 backbone, 6-DOF pose vector from YUV6 input at 192×256, evaluated on BOTH frames of each pair)

The score formula: `100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * (archive_size / 37,545,489)`

## Top-Level Strategy: Don't Store the Video

Conventional approaches encode the actual RGB video (with AV1, H.265, etc.). Even with extreme compression and ROI preprocessing, this hits a hard rate floor around 25*830,000/37,545,489 = **0.55** rate contribution.

**mask2mask completely abandons video storage.** Instead, the archive contains three brotli-compressed files totaling **386,192 bytes** (rate contribution = 0.257):

| File | Size | Purpose |
|------|------|---------|
| `mask.mp4.br` | 209,341 B | Brotli-compressed AV1-encoded video of SegNet's class predictions on the original frames (5 classes encoded as grayscale 0-255) |
| `model.pt.br` | 171,628 B | Brotli-compressed FP4-quantized neural network ("AsymmetricPairGenerator") that converts mask pairs back into RGB frames |
| `arch.br` | 4,790 B | The model architecture itself, marshal-serialized + brotli-compressed Python bytecode |

At decode time:
1. Decompress and play `mask.mp4.br` → recover 1200 frames of class indices at 384×512
2. Load the obfuscated architecture from `arch.br` via `marshal.loads(brotli.decompress(...))` and `exec()`
3. Load the FP4-quantized weights into the architecture
4. For each consecutive frame pair (mask1, mask2):
   - Generator hallucinates RGB frame pair (fake1, fake2) from the mask pair
   - Upscale to 1164×874 with bilinear interpolation
5. Write raw RGB output

The hallucinated frames look NOTHING like the original video — they only need to fool SegNet and PoseNet into producing the same outputs as the original frames.

## The AsymmetricPairGenerator

This is the heart of the submission. It takes two consecutive class masks `(mask1, mask2)` at 384×512 with 5 possible class values, and outputs two RGB frames `(fake1, fake2)` at 384×512.

```python
class AsymmetricPairGenerator(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.frame2 = TinyFrame2Renderer(num_classes=5, emb_dim=6, c1=36, c2=60)
        self.motion = TinyMotionFromMasks(num_classes=5, emb_dim=6, c1=32, c2=48)
    
    def forward(self, mask1, mask2):
        b = mask2.shape[0]
        coords = make_coord_grid(b, 384, 512, mask2.device, torch.float32)
        
        # Render frame 2 from scratch (this is the SegNet target frame)
        fake2 = self.frame2(mask2, coords)
        
        # Predict optical flow from frame 1 → frame 2 + per-pixel gate + residual correction
        flow_px, gate, residual = self.motion(mask1, mask2, coords)
        
        # Frame 1 = warped(frame 2) * gate + residual
        warped2 = warp_with_flow(fake2, flow_px)
        fake1 = (warped2 * gate + residual).clamp(0.0, 255.0)
        
        return fake1, fake2  # both at (B, 3, 384, 512), values in [0, 255]
```

**Critical insight: the asymmetric design.** The two frames in each pair serve fundamentally different evaluation roles:
- Frame 2 (the odd index) is what SegNet sees. It must produce the exact correct class predictions when run through SegNet.
- Frame 1 (the even index) is only used by PoseNet alongside frame 2. PoseNet computes a 6-D pose vector from the relative geometry of the two frames. As long as the relative motion looks correct, the absolute appearance of frame 1 barely matters.

So they fully render frame 2 with a dedicated network (`TinyFrame2Renderer`), and for frame 1 they just predict how to **warp** frame 2 backward in time using a tiny optical-flow estimator (`TinyMotionFromMasks`). This avoids needing a second full renderer, halving the parameter cost for the frame 1 path.

## TinyFrame2Renderer (frame 2 rendering)

Renders a single 3-channel RGB frame from a single class mask plus a coordinate grid.

```python
class TinyFrame2Renderer(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=36, c2=60):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)  # FP16
        # Input to stem: 6 (embedding) + 2 (coord grid) = 8 channels
        self.stem = nn.Sequential(
            ConvGNAct(8, c1, k=3, stride=1, quantize_weight=True),  # 8→36
            ResBlock(c1, quantize_weight=True),                     # 36→36
        )
        self.down = nn.Sequential(
            ConvGNAct(c1, c2, k=3, stride=2, quantize_weight=True), # 36→60, /2
            ResBlock(c2, quantize_weight=True),                     # 60→60
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvGNAct(c2, c1, k=3, stride=1, quantize_weight=True), # 60→36, x2
        )
        # Skip-connection concat: c1 (from stem) + c1 (from up) = 72 channels
        self.dec = nn.Sequential(
            ConvGNAct(2*c1, c1, k=3, stride=1, quantize_weight=True), # 72→36
            ResBlock(c1, quantize_weight=True),                       # 36→36
        )
        self.head = QConv2d(c1, 3, kernel_size=1, quantize_weight=False)  # FP16
    
    def forward(self, mask2, coords):
        # mask2: (B, H, W) long, coords: (B, 2, H, W) float
        emb = self.embedding(mask2.long()).permute(0, 3, 1, 2)  # (B, 6, H, W)
        x = torch.cat([emb, coords], dim=1)                     # (B, 8, H, W)
        s = self.stem(x)                                        # (B, 36, H, W)
        d = self.down(s)                                        # (B, 60, H/2, W/2)
        u = self.up(d)                                          # (B, 36, H, W)
        # Resize u to match s if there's a rounding mismatch
        if u.shape[-2:] != s.shape[-2:]:
            u = F.interpolate(u, size=s.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([s, u], dim=1)                            # (B, 72, H, W)
        x = self.dec(x)                                         # (B, 36, H, W)
        return torch.sigmoid(self.head(x)) * 255.0              # (B, 3, H, W)
```

This is a tiny U-Net: stem → downsample → upsample → decoder with a skip connection. Only **two resolution levels**, channel widths 36 and 60. The output passes through sigmoid and is scaled to [0, 255].

## TinyMotionFromMasks (frame 1 prediction)

Same encoder structure but takes BOTH masks (mask1 and mask2) and outputs three things: optical flow, a per-pixel gate, and a per-pixel residual.

```python
class TinyMotionFromMasks(nn.Module):
    def __init__(self, num_classes=5, emb_dim=6, c1=32, c2=48):
        super().__init__()
        self.embedding = QEmbedding(num_classes, emb_dim, quantize_weight=False)  # FP16
        # Input: 2*emb (12 from concatenated embeddings) + 2 (abs diff) + 2 (coords) = 16? wait
        # Actual: emb(mask1) + emb(mask2) + |emb(mask1) - emb(mask2)| + coords
        # From bytecode: 'abs' is in names list — they use abs(emb1 - emb2) as a feature
        # Total stem input channels: 6+6+6+2 = 20 (matches the model weights: motion.stem.0.conv shape [32, 20, 3, 3])
        self.stem = nn.Sequential(
            ConvGNAct(20, c1, k=3, stride=1, quantize_weight=True),
            ResBlock(c1, quantize_weight=True),
        )
        self.down = nn.Sequential(
            ConvGNAct(c1, c2, k=3, stride=2, quantize_weight=True),
            ResBlock(c2, quantize_weight=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvGNAct(c2, c1, k=3, stride=1, quantize_weight=True),
        )
        self.dec = nn.Sequential(
            ConvGNAct(2*c1, c1, k=3, stride=1, quantize_weight=True),
            ResBlock(c1, quantize_weight=True),
        )
        # Three FP16 1x1 conv heads
        self.flow_head = QConv2d(c1, 2, 1, quantize_weight=False)      # 2 channels: dx, dy
        self.gate_head = QConv2d(c1, 1, 1, quantize_weight=False)      # 1 channel: blend mask
        self.residual_head = QConv2d(c1, 3, 1, quantize_weight=False)  # 3 channels: RGB residual
    
    def forward(self, mask1, mask2, coords):
        emb1 = self.embedding(mask1.long()).permute(0, 3, 1, 2)
        emb2 = self.embedding(mask2.long()).permute(0, 3, 1, 2)
        diff = (emb1 - emb2).abs()
        x = torch.cat([emb1, emb2, diff, coords], dim=1)  # (B, 20, H, W)
        s = self.stem(x)
        d = self.down(s)
        u = self.up(d)
        if u.shape[-2:] != s.shape[-2:]:
            u = F.interpolate(u, size=s.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([s, u], dim=1)
        x = self.dec(x)
        # Three outputs with calibrated activation ranges:
        flow_px = torch.tanh(self.flow_head(x)) * 12.0      # ±12 pixel motion
        gate = torch.sigmoid(self.gate_head(x))             # [0, 1]
        residual = torch.tanh(self.residual_head(x)) * 20.0 # ±20 RGB units
        return flow_px, gate, residual
```

## Helper Components

### `ConvGNAct` block
```python
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, quantize_weight=False):
        super().__init__()
        self.conv = QConv2d(in_ch, out_ch, k, stride=stride, padding=k//2, bias=False, quantize_weight=quantize_weight)
        self.norm = nn.GroupNorm(num_groups=int(...), num_channels=out_ch)  # likely 8 groups
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
```

### `ResBlock`
```python
class ResBlock(nn.Module):
    def __init__(self, ch, quantize_weight=False):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(ch, ch, k=3, stride=1, quantize_weight=quantize_weight),
            QConv2d(ch, ch, kernel_size=3, padding=1, bias=False, quantize_weight=quantize_weight),
            nn.GroupNorm(num_groups=2, num_channels=ch),  # exactly 2 groups based on bytecode
        )
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.block(x))
```

### `make_coord_grid`
```python
def make_coord_grid(batch, height, width, device, dtype):
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack([xx, yy], dim=0).unsqueeze(0)  # (1, 2, H, W)
    return grid.expand(batch, -1, -1, -1).contiguous()
```

Note: this returns absolute pixel coordinates (not normalized), giving the network explicit positional awareness.

### `warp_with_flow` (backward warping with bilinear sampling)
```python
def warp_with_flow(img, flow_px):
    B, C, H, W = img.shape
    base = make_coord_grid(B, H, W, img.device, img.dtype)  # (B, 2, H, W) absolute pixels
    new_coords = base + flow_px                             # apply pixel-space offset
    # Normalize to [-1, 1] for grid_sample
    norm_x = 2.0 * new_coords[:, 0] / (W - 1) - 1.0
    norm_y = 2.0 * new_coords[:, 1] / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)            # (B, H, W, 2)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
```

### `QConv2d` and `QEmbedding`
These are quantization-aware wrappers around nn.Conv2d and nn.Embedding. They have a `quantize_weight` flag that, when True, applies fake-quantization to weights during forward pass (presumably FP4 or similar low-bit). The `qat_enabled` and `qat_act_enabled` flags toggle quantization-aware training mode.

The forward pass simply calls `F.conv2d`/`F.embedding` with whatever the current weights are (quantized during training, or fully quantized at inference). The weight quantization is the FP4 codebook described below.

## FP4 Quantization Scheme

Weights are quantized to **4 bits per parameter** using a custom codebook:

```python
class FP4Codebook:
    pos_levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    # 8 positive levels + sign bit = 16 total levels (4 bits)
```

Each weight tensor is divided into blocks (block size determined by `__block_size__` in the saved file). Each block has its own FP16 scale factor. Within a block, each weight is encoded as:
- 1 sign bit
- 3 magnitude bits (index into `pos_levels`)
- Total: 4 bits per weight, packed two per byte (nibbles)

Dequantization:
```python
def dequantize(nibbles, scales, orig_shape):
    block_size = nibbles.numel() // scales.numel()
    nibbles = nibbles.view(-1, block_size)
    signs = (nibbles >> 3).to(torch.int64)
    mag_idx = (nibbles & 0x7).to(torch.int64)
    levels = FP4Codebook.pos_levels[mag_idx]
    q = torch.where(signs.bool(), -levels, levels)
    return (q * scales[:, None]).view(orig_shape)
```

This gives **8x compression vs FP32** and **2x vs INT8**. For a network with ~300K parameters, FP4 = ~150KB raw, which compresses with brotli to ~170KB (the actual `model.pt.br` size, which includes some FP16 layers and metadata).

## What's Quantized vs FP16

**FP4-quantized layers** (the "expensive" weights):
- All ConvGNAct convolutions
- All ResBlock convolutions
- All down/up convolutions
- Total: 26 layers across both networks

**Kept as FP16** (the "sensitive" weights):
- Class embeddings (5×6 — tiny, doesn't matter)
- Output heads (1×1 convs producing RGB / flow / gate / residual — preserving precision matters most at the output)
- All GroupNorm parameters (γ, β — these are tiny)

Total parameter count: **308,392** (most quantized to 4-bit, output heads and norms in FP16).

## Model Weight Sizes Breakdown

| Component | Layers | Params | Quantized |
|-----------|--------|--------|-----------|
| frame2.embedding | 1 | 30 | FP16 |
| frame2.stem (2 layers) | 2 | ~14k | FP4 |
| frame2.down (2 layers) | 2 | ~52k | FP4 |
| frame2.up (1 layer) | 1 | ~19k | FP4 |
| frame2.dec (2 layers) | 2 | ~36k | FP4 |
| frame2.head | 1 | 108 | FP16 |
| motion.embedding | 1 | 30 | FP16 |
| motion.stem (2 layers) | 2 | ~15k | FP4 |
| motion.down (2 layers) | 2 | ~36k | FP4 |
| motion.up (1 layer) | 1 | ~14k | FP4 |
| motion.dec (2 layers) | 2 | ~28k | FP4 |
| motion.flow_head | 1 | 64 | FP16 |
| motion.gate_head | 1 | 32 | FP16 |
| motion.residual_head | 1 | 96 | FP16 |
| GroupNorm params | many | ~1k | FP16 |
| **Total** | | **~308k** | mixed |

After FP4 quantization: ~150 KB raw weight data + FP16 heads/norms → ~165 KB after brotli → 171 KB on disk.

## The Mask Video

The "input" to the generator at decode time is a 1200-frame video at 384×512 with 5-class segmentation predictions. They encode this as an AV1 video with grayscale pixel values mapped from class indices:

**Encoding:** `pixel_value = round(class * 63.75)` (so classes 0,1,2,3,4 map to roughly 0, 64, 128, 191, 255)

**Decoding:** `class = round(pixel_value / 63.0).clip(0, 4)`

Why this works: AV1 is incredibly efficient at compressing nearly-constant regions, and class maps have huge contiguous areas. The actual decoded pixel values are noisy (many intermediate values like 5, 27, 84, etc., probably from AV1 quantization), but the round-to-nearest-class step recovers the exact integer class. The mask video is at 384×512 (model resolution) and uses libdav1d for decoding.

**Mask video size: 209 KB compressed** (vs ~830 KB for the actual video at the same resolution).

The training pipeline must have:
1. Run SegNet on each original odd frame to get the golden 384×512 class predictions
2. Encoded these class maps as a low-bitrate AV1 video
3. Verified that the round-trip decode preserves class identities exactly

## Training Objective (Inferred)

The submission doesn't include the training script, but the loss can be reverse-engineered from the architecture and the score formula:

```python
def loss(generator, mask1, mask2, gt_frame1, gt_frame2, segnet, posenet):
    # Generate fake frames from masks
    fake1, fake2 = generator(mask1, mask2)
    
    # Upscale to 1164×874 for the evaluation pipeline
    fake1_up = F.interpolate(fake1, size=(874, 1164), mode='bilinear', align_corners=False)
    fake2_up = F.interpolate(fake2, size=(874, 1164), mode='bilinear', align_corners=False)
    
    # Get the original SegNet/PoseNet outputs as targets
    with torch.no_grad():
        gt_seg = segnet(prep_seg(gt_frame2_up))
        gt_pose = posenet(prep_pose(stack(gt_frame1_up, gt_frame2_up)))
    
    # SegNet loss: KL divergence or cross-entropy on logits
    # (only frame2 matters for SegNet)
    fake_seg = segnet(prep_seg(fake2_up))
    seg_loss = F.kl_div(F.log_softmax(fake_seg), F.softmax(gt_seg))
    
    # PoseNet loss: MSE on the 6-D pose vector
    # (uses both frames)
    fake_pose = posenet(prep_pose(stack(fake1_up, fake2_up)))
    pose_loss = F.mse_loss(fake_pose[:, :6], gt_pose[:, :6])
    
    return 100 * seg_loss + 3.16 * pose_loss  # Match score formula weights
```

The training would be **pure memorization** on the 600 fixed pairs (no augmentation, no validation split). The author hints that "<0.50 is pretty easily possible as a slightly different architecture gets a score at least 10% better" — suggesting their architecture is a reasonable but not optimal point in the design space.

Likely training details (inferred from architecture choices):
- Adam or AdamW optimizer
- Long training (thousands of epochs) on the fixed dataset
- QAT (Quantization-Aware Training) — the architecture has explicit `quantize_weight=True` flags and the QMixin class with `qat_enabled` toggle, indicating they trained with simulated FP4 quantization to ensure the final quantized model performs as expected
- Two-stage training would make sense: first train at FP32 to convergence, then enable QAT for fine-tuning

## Why This Approach Works So Well

### 1. Massive rate savings via mask-only storage
Storing class indices instead of RGB values cuts the rate contribution from ~0.55 to ~0.26 — a savings of **0.29 points** that's essentially impossible to match with any conventional codec approach.

### 2. SegNet distortion is ~zero by construction
The mask video stores the exact SegNet predictions on the original frames. As long as the generator produces frame 2 RGB values that, when fed back into SegNet, produce the same argmax, the SegNet distortion approaches zero. The generator doesn't have to "look like" the original frame at all — it just needs to fool SegNet at 384×512.

### 3. Asymmetric frame 1 generation halves the parameter cost
Frame 1 doesn't need a full renderer. A tiny optical-flow predictor + warping + small residual is enough to satisfy PoseNet. This is the key insight that makes the model fit in <200KB.

### 4. The coord grid + class embedding is enough conditioning
The generator receives essentially three inputs: which class each pixel belongs to (5 levels), where it is spatially (xy coords), and what's around it (via convolutions). That's enough information to hallucinate plausible RGB values that activate the right SegNet/PoseNet features.

### 5. FP4 quantization is rate-optimal for fixed-overfit models
For a model that's pure memorization on a fixed dataset, you don't need 32-bit precision. The optimization process naturally finds weight values that survive 4-bit quantization (especially with QAT). FP4 gives 8x compression vs FP32 with minimal loss.

## Score Breakdown

The submission's actual scores:
- **SegNet distortion: 0.00264** → contributes `100 * 0.00264 = 0.264`
- **PoseNet distortion: 0.000656** → contributes `sqrt(10 * 0.000656) = 0.081`
- **Rate: 386,192 bytes / 37,545,489 = 0.01029** → contributes `25 * 0.01029 = 0.257`
- **Total: 0.264 + 0.081 + 0.257 = 0.602**

## Improvements They Hint At

The author writes: *"<0.50 is pretty easily possible as a slightly different architecture gets a score at least 10% better."*

Possible architectural improvements:
1. **Smaller mask resolution** — currently 384×512. Going to 192×256 would halve the mask video size (~104 KB vs 209 KB), saving ~0.07 rate, IF the generator can still produce SegNet-satisfying upscales.
2. **Stronger generator** — the current generator only has 308K params. A larger model could fit better in the 5-class manifold.
3. **Better mask compression** — use a more efficient codec for the masks (e.g., a custom run-length scheme) instead of AV1.
4. **Use temporal info** — currently each pair is independent. A model that conditions on a window of multiple masks could exploit redundancy.
5. **Skip the asymmetry** and use a single network that takes both masks at once and produces both frames jointly.
6. **Lower precision** — try ternary or 2-bit weights with more aggressive QAT.
7. **Split the mask into multiple channels** — store class boundaries separately from the class itself for higher precision in critical regions.

## Compliance with Challenge Rules

The rules state: *"External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size."*

mask2mask is fully compliant:
- The generator network IS in the archive (counts toward 386 KB)
- The architecture is in the archive (4.8 KB of bytecode)
- SegNet and PoseNet (the evaluation models) are NOT used at decode time
- All decode operations use only `torch`, `numpy`, `av`, `brotli`, `einops` — standard libraries

## Implementation Roadmap for a Competing Submission

To build something equivalent or better:

### Phase 1: Generate the mask video
1. Run SegNet on every odd frame of `videos/0.mkv` at 384×512 to get golden class predictions
2. Encode these as an AV1 video with grayscale pixel values = `class * 63.75`
3. Tune AV1 settings for absolute minimum size while preserving round-trip class accuracy
4. Verify that decoding+rounding recovers the exact original predictions

### Phase 2: Build the generator architecture
1. Implement the AsymmetricPairGenerator above (reusing the exact channel widths is a starting point)
2. Add QConv2d/QEmbedding wrappers with FP4 fake-quantization support
3. Add QMixin for QAT mode toggling

### Phase 3: Training loop
1. Load 600 pairs of (compressed mask, original RGB frames)
2. Forward pass: generate fake frames from masks
3. Compute SegNet KL loss on frame 2 (target = SegNet outputs on the original frame 2)
4. Compute PoseNet MSE loss on the pair (target = PoseNet outputs on the original pair)
5. Backward through everything (the SegNet/PoseNet are frozen, gradients flow through)
6. Train for 5000+ epochs with cosine annealing
7. Enable QAT after initial convergence
8. Continue training to refine weights for the FP4 codebook

### Phase 4: Quantize, pack, submit
1. Apply final FP4 quantization with per-block scales
2. Save to dict with `quantized` (FP4 layers) and `dense_fp16` (heads, norms, embeddings)
3. Brotli-compress the model state dict
4. Brotli-compress the mask video
5. Marshal + brotli the architecture file
6. Pack all three into archive.zip

### Phase 5: Inference (for verification)
Use the inflate.py from the published submission (we have it) — modify only the architecture loading to point at our own model.

## Key Files in Our Codebase

- `submissions/mask2mask/inflate.py` — Available in PR #53, can be adapted
- `submissions/mask2mask/inflate.sh` — Trivial wrapper
- `modules.py` — SegNet and PoseNet definitions for training
- `frame_utils.py` — `camera_size = (1164, 874)`, `segnet_model_input_size = (512, 384)`, YUV/RGB helpers

## Open Questions for Deeper Research

1. **What's the optimal mask resolution?** Current is 384×512. Could 192×256 work? 256×256?
2. **Can the mask video be smaller than AV1 manages?** Could a custom entropy coder beat AV1 for class index sequences?
3. **Is FP4 the right precision?** Could FP3 or ternary work with more capacity?
4. **What's the right architecture size?** 308K params hits 0.60. What does 500K or 1M params do?
5. **Does pre-training help?** Could you bootstrap from a pretrained image generation model?
6. **Could a transformer attention layer help?** Even a single tiny self-attention block could capture global context that helps SegNet boundaries.
7. **Is there a way to encode JUST the class boundaries instead of the full mask?** Boundaries are where SegNet errors come from anyway.
8. **Could we use a learned mask codec instead of AV1?** A small autoencoder might compress class maps better than a generic video codec.
9. **What if frame1 and frame2 share more than just the warp?** The asymmetric design might be too aggressive — maybe a small frame1 renderer would be worth its cost.
10. **Can we exploit the fact that the camera's motion is mostly forward translation?** A forward-motion-specific flow model could be much smaller than a generic optical flow predictor.
