# Current Status: mask2mask Pipeline — Research Brief

## The Challenge

The comma.ai video compression challenge requires compressing a single 1200-frame dashcam video (1164×874, 20fps, ~37.5MB) such that the decompressed output minimizes disagreement on two evaluation neural networks:

**Score formula:** `100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * (archive_bytes / 37,545,489)`

- **SegNet** (smp.Unet, EfficientNet-B2 backbone, 5-class semantic segmentation): Evaluates ONLY the odd frame (frame 1) of each consecutive pair at 384×512 resolution. Metric: fraction of pixels where argmax class disagrees between original and compressed.
- **PoseNet** (FastViT-T12, YUV6 input at 192×256): Evaluates BOTH frames of each pair. Metric: MSE of 6-dim pose vectors between original and compressed pairs.
- **Rate**: archive.zip bytes / 37,545,489 bytes.

The evaluation is on a single fixed video (`videos/0.mkv`). This is a pure memorization/overfitting task — generalization doesn't matter.

## The mask2mask Paradigm

Instead of storing the actual video (which hits a hard rate floor of ~0.55), we store:
1. **Segmentation masks** as a compressed AV1 video (~190KB after brotli) — the SegNet class predictions on each frame at 384×512
2. **A generator neural network** (~165KB after FP4 quantization + brotli) — an `AsymmetricPairGenerator` that hallucinates RGB frames from mask pairs
3. **Architecture bytecode** (~7KB) — the model definition

Total archive: ~362KB. Rate contribution: **0.242** (vs leader's 0.257 — we're smaller).

At decode time, the generator takes mask pairs and outputs fake RGB frame pairs. The frames look nothing like the original video — they're abstract patterns that just happen to fool SegNet and PoseNet.

## Our Architecture (matches the leader's PR #53 exactly)

```
AsymmetricPairGenerator (308,401 params)
├── TinyFrame2Renderer (frame 2 — what SegNet sees)
│   ├── QEmbedding(5, 6)  [FP16]
│   ├── stem: ConvGNAct(8→36) + ResBlock(36)
│   ├── down: ConvGNAct(36→60, stride=2) + ResBlock(60)
│   ├── up: Upsample(2x) + ConvGNAct(60→36)
│   ├── dec: ConvGNAct(72→36, skip connection) + ResBlock(36)
│   └── head: Conv2d(36→3, 1×1)  [FP16, sigmoid * 255]
│
└── TinyMotionFromMasks (frame 1 — warped from frame 2 for PoseNet)
    ├── QEmbedding(5, 6)  [FP16]
    ├── stem: ConvGNAct(20→32) + ResBlock(32)  [input: emb1+emb2+|diff|+coords]
    ├── down: ConvGNAct(32→48, stride=2) + ResBlock(48)
    ├── up: Upsample(2x) + ConvGNAct(48→32)
    ├── dec: ConvGNAct(64→32, skip connection) + ResBlock(32)
    ├── flow_head: Conv2d(32→2, 1×1)  [FP16, tanh * 12]
    ├── gate_head: Conv2d(32→1, 1×1)  [FP16, sigmoid]
    └── residual_head: Conv2d(32→3, 1×1)  [FP16, tanh * 20]

Forward pass:
  fake2 = frame2_renderer(mask2, coord_grid)
  flow, gate, residual = motion_network(mask1, mask2, coord_grid)
  fake1 = warp(fake2, flow) * gate + residual
```

Key design: Frame 2 is fully rendered (SegNet needs it). Frame 1 is just a warped+corrected version of frame 2 (PoseNet only needs relative geometry).

ConvGNAct = Conv2d + GroupNorm + SiLU. ResBlock = ConvGNAct → Conv → GroupNorm + skip + SiLU.

## FP4 Quantization

Weights are quantized to 4-bit with a custom non-linear codebook:
```
positive levels: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
+ sign bit = 16 total levels per weight
```

Per-block scaling with block_size=32 and FP16 scale factors. The conv body layers use FP4. The output heads (flow, gate, residual, RGB) and embeddings use FP16 for precision. GroupNorm params are FP16.

After FP4 packing + brotli compression: ~165KB for 308K parameters.

## Training Setup

**Loss function:**
```python
loss = 100.0 * cross_entropy(segnet(fake2_resized), gt_seg_argmax) + 2.0 * sqrt(10 * mse(posenet(pair), gt_pose))
```

This directly mirrors the score formula (with 2x pose boost to prioritize motion quality). The sqrt makes pose gradient grow as pose gets small, naturally rebalancing.

**QAT:** FP4 FakeQuantization on conv weights during forward pass with straight-through estimator for gradients. Enabled from epoch 1 of the current run.

**uint8 STE:** We simulate the uint8 round-trip (float→round→clamp→float) during training with a straight-through estimator, so the model learns to produce values that survive the integer rounding in the actual eval pipeline.

**Golden targets:** SegNet argmax and PoseNet 6-dim pose vectors computed on the original video, preprocessed through the same Lanczos→unsharp→bilinear pipeline that the real evaluator uses.

**Data:** 600 consecutive frame pairs from the fixed video. No augmentation, no regularization (pure memorization).

**Optimizer:** Adam, cosine LR schedule from 1e-4 to 1e-5 over 2000 epochs. Batch size 4.

## Current Scores

| Metric | Our v2 | Leader (PR #53) | Gap |
|--------|--------|-----------------|-----|
| **Total score** | **0.70** | **0.61** | **0.09** |
| SegNet (100×) | 0.284 | 0.267 | 0.017 |
| PoseNet (sqrt) | 0.172 | 0.090 | **0.082** |
| Rate (25×) | 0.242 | 0.257 | -0.015 (we're better) |

**The gap is 91% PoseNet.** Our SegNet and rate are competitive.

## What We've Tried

### Training approaches (chronological)
1. **v1 loss (KL+CE for seg, raw MSE for pose):** Converged to 2.34 in 3 epochs but pose dominated by seg loss. Plateaued at 1.4 (no rate) after 275 epochs.
2. **v2 loss (CE for seg, sqrt(10*MSE) for pose):** Matched score formula. Rapid convergence. FP32 reached 0.34 (no rate) by epoch 1250.
3. **QAT fine-tuning (200 epochs from FP32 best):** Recovered from FP4 shock. QAT score 0.36. But real eval showed 0.70 — large gap.
4. **QAT from start with high LR warm restart:** Best approach. 1000 epochs, reached QAT score 0.35. Real eval still 0.70.
5. **2x pose weight + QAT warm restart:** Current run. QAT score converging to ~0.37. Real eval 0.70.
6. **uint8 STE simulation in training:** Added round().clamp() with STE to simulate the eval pipeline's uint8 quantization. Score now includes this loss. Converging to ~0.37.

### Quantization experiments
7. **FP4 with block_size=64:** Original attempt. High quantization noise.
8. **FP4 with block_size=32:** Matches leader. Better but still significant pose degradation.
9. **Mixed precision (FP4 frame2, INT8 motion):** Made things WORSE (0.75) because model was trained for FP4 codebook, not INT8.

### Upscale experiments
10. **Optimized upscale (iterative refinement to minimize bilinear round-trip):** Didn't help — the round-trip error isn't what hurts.

### Even-frame sidecars on top of mask2mask
11. **dxyr corrections for generated even frames:** Marginal or negative — the generator already optimized the frame pair relationship. Corrections fight the learned output.

## The Core Problem

Our FP32 QAT training achieves:
- seg = 0.00280 (close to leader's 0.00267)
- pose = 0.00050 (BETTER than leader's 0.00081)

But after FP4 quantization → inflate → real eval:
- seg = 0.00284 (minimal degradation ✓)
- pose = 0.00297 (6x degradation from 0.00050 ✗)

**FP4 quantization degrades pose by 6x but barely affects seg.** The motion network's optical flow predictions are much more sensitive to weight precision than the frame renderer's RGB output.

The leader somehow achieves pose=0.00081 after FP4 quantization. They use the same architecture, same FP4 codebook, same block_size=32. So their FP4-quantized motion weights must be in a DIFFERENT local minimum that happens to be more robust to 4-bit precision.

## Key Files

- `submissions/mask2mask_v2/architecture.py` — model definition
- `submissions/mask2mask_v2/train.py` — training script with QAT + uint8 STE
- `submissions/mask2mask_v2/inflate.py` — decode pipeline (brotli decompress → load model → generate frames)
- `submissions/mask2mask_v2/quantize_pack.py` — FP4 quantization + archive packing
- `submissions/mask2mask_v2/extract_masks.py` — SegNet mask extraction from original video
- `modules.py` — SegNet and PoseNet definitions (evaluation models)
- `frame_utils.py` — camera_size=(1164,874), segnet_model_input_size=(512,384), YUV conversion
- `evaluate.py` — official evaluation script

## The Evaluation Pipeline (exact data flow)

### Compression (offline):
1. Run SegNet on all 1200 original frames → get 384×512 class predictions
2. Encode class predictions as AV1 grayscale video (class × 63.75)
3. Brotli-compress mask video
4. Train AsymmetricPairGenerator with QAT + uint8 STE
5. FP4-quantize model weights, brotli-compress
6. Marshal + brotli the architecture Python code
7. Pack all three into archive.zip (currently 362KB)

### Inflation (decode time):
1. Brotli-decompress all three files
2. Load architecture via `exec(marshal.loads(...))`
3. FP4-dequantize model weights and load into generator
4. Decode mask video from AV1 → round to nearest class (0-4)
5. For each of 600 mask pairs:
   - Generator outputs fake1, fake2 at 384×512
   - Bilinear upscale both to 1164×874
   - Round to uint8 and write to .raw file

### Evaluation:
1. Load original video and compressed .raw file
2. Group into pairs of 2 consecutive frames
3. For each pair:
   - Resize to 384×512 with bilinear (both original and compressed)
   - Run SegNet on frame 2 only → compare argmax
   - Convert to YUV6 at 192×256 → run PoseNet on both frames → compare pose vectors
4. Average seg disagreement and pose MSE across 600 pairs
5. Compute `100*seg + sqrt(10*pose) + 25*rate`

### The round-trip for generated frames:
```
Generator output (384×512 float) 
  → bilinear upscale to 1164×874 
  → round to uint8 
  → write to disk 
  → read from disk 
  → bilinear downscale to 384×512 (SegNet)
  → bilinear downscale + YUV6 at 192×256 (PoseNet)
```

The upscale→uint8→downscale round-trip introduces ~1-2 pixel mean error for the generator's intended output. We simulate this in training with STE.

## What the Leader Likely Did Differently

Based on their score (0.61) and our analysis:
1. **Trained for much longer** — possibly 5000-10000+ epochs. Our best was 1000-2000 QAT epochs.
2. **Better QAT schedule** — they may have used curriculum: FP32 warmup → gradual FP4 introduction → full QAT. We jump-started QAT which may find a worse local minimum.
3. **Different loss weights** — they may have tuned the seg/pose balance differently.
4. **Possibly a custom QAT implementation** — their QConv2d forward does `F.conv2d(x, self.weight, ...)` which suggests they might apply fake quantization OUTSIDE the forward method (e.g., in a training hook), giving more control over when/how quantization noise is applied.
5. **Brotli quality tuning** — they may use different brotli quality levels for different files.

## Open Questions for Research

### 1. Why does FP4 quantization destroy pose but not seg?
Both networks have similar architectures (tiny U-Net with skip connections). Both are FP4-quantized. But pose degrades 6x while seg barely moves. Is this because:
- The flow prediction (tanh × 12) amplifies small weight errors into pixel-level shifts?
- The warping operation (grid_sample) is highly sensitive to sub-pixel flow errors?
- The PoseNet evaluation at 192×256 with YUV6 conversion amplifies errors differently than SegNet at 384×512?

### 2. How can we find a more quantization-robust minimum for the motion network?
The FP4 codebook has only 16 values. The motion network needs to predict flow with sub-pixel accuracy (typical flow ±2 pixels). With FP4 weights, the effective flow resolution is limited. Can we:
- Use a different activation scaling (e.g., tanh × 6 instead of × 12) to reduce the dynamic range the quantized weights need to cover?
- Add a learnable per-channel scale after each FP4 conv (stored in FP16) to compensate for quantization error?
- Regularize the weight distribution during training to be more "FP4-friendly" (e.g., encourage weights near codebook levels)?

### 3. Can we reduce the mask video size?
Currently 190KB after brotli (CRF 60 AV1). Options:
- Lower resolution masks (256×340 or 192×256) — the generator's embedding+interpolate supports this natively
- Higher CRF (63 = 97KB) — loses 0.56% of pixels, but the generator might learn to compensate
- Different codec (custom entropy coding for 5-class sequences)
- Temporal delta coding (store only changed pixels between frames)

### 4. Is there a way to make the generator output closer to what the eval expects WITHOUT changing the architecture?
- Train with the exact eval preprocessing in the loss (we're doing this now with uint8 STE)
- Apply a cheap post-processing filter in inflate.py that adjusts the generated frames
- Use a different upscale method in inflate.py (nearest, bicubic, area-based)

### 5. Can we exploit the PoseNet evaluation more?
PoseNet uses YUV6 at 192×256. The rgb_to_yuv6 conversion subsamples chroma (U,V) by 2×2 averaging. This means:
- Individual pixel chroma variations within 2×2 blocks are invisible to PoseNet
- We could potentially use this to store additional information without affecting PoseNet
- But more importantly: the flow prediction only needs to be accurate at 192×256 resolution, not 384×512

### 6. Could a post-training weight perturbation search help?
Instead of retraining, could we:
- Take the best FP4-quantized weights
- Do a local search: for each weight, try the neighboring FP4 codebook levels
- Accept changes that improve the real eval score
- This is essentially "quantization-aware fine-tuning" but done directly on the discrete weights

### 7. Are there training tricks from the knowledge distillation / model compression literature?
- Progressive quantization: start at FP16, gradually reduce to FP8, FP6, FP4
- Learned step size quantization (LSQ): learn the quantization scale factors
- GPTQ-style: quantize one layer at a time, compensating for error in subsequent layers
- AWQ (Activation-Aware Quantization): scale weights based on activation magnitudes before quantizing

### 8. Can we improve score via the mask video encoding?
The masks are encoded as grayscale (class × 63.75). After AV1 compression, some pixels are misclassified on round-trip (0.27% at CRF 60). These misclassified mask pixels create an irreducible seg error floor. Could we:
- Use a different class-to-intensity mapping that's more robust to AV1 quantization?
- Encode the mask at a higher quality but at lower resolution?
- Use error-correction codes in the mask encoding?

### 9. What if we train two separate models?
- A frame2-only model (optimized for SegNet) with FP4
- A motion-only model (optimized for PoseNet) at higher precision (FP8 or INT8)
- The motion model is smaller, so higher precision costs less in rate
- But this requires changing the architecture slightly (separate forward passes)

### 10. Is there a fundamentally different approach to frame 1 generation?
Currently: warp(frame2, flow) × gate + residual. This requires accurate flow prediction.
Alternative: directly predict frame 1 from the mask pair WITHOUT warping. A second small renderer (like frame2 but smaller) that directly outputs frame 1. No flow = no flow quantization error. But more parameters.
