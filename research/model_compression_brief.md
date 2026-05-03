# Deep Research Brief: Extreme Neural Network Weight Compression for Adversarial Video Codec

## The Challenge

comma.ai video compression challenge. Compress a 37.5 MB dashcam video. Score formula (lower is better):

```
score = 100 * segnet_distortion + sqrt(10 * posenet_distortion) + 25 * compression_rate
```

- **SegNet distortion**: average pixel-wise argmax class disagreement (5 classes) between segmentation predictions on original vs reconstructed frames. Discrete metric.
- **PoseNet distortion**: MSE of 6-dim pose vectors predicted from pairs of consecutive frames (original vs reconstructed). Continuous metric.
- **Compression rate**: archive.zip size / original video size (37,545,489 bytes)

Current baseline score: 4.4. Our approach without model weights achieves ~0.3.

## Our Approach: Adversarial Decode

Instead of compressing video frames, we:
1. **Encode**: Run SegNet + PoseNet on original video, extract their outputs (600 segmentation maps + 600 pose vectors). Compress these to ~314 KB.
2. **Decode**: Use gradient descent to *invert* the evaluation networks — generate frames whose SegNet argmax and PoseNet pose outputs match the originals. Frames look nothing like the original video; they only need to produce identical network outputs.

This achieves near-zero distortion on both metrics with an archive of only 314 KB.

## The Problem

Competition rules state: *"External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. This applies to the PoseNet and SegNet."*

Our decode requires SegNet + PoseNet at inference time. If their weights must be in the archive, we need to compress them **aggressively** while preserving output fidelity.

## Budget Analysis

For score < 4.4 (beat baseline), assuming ~0 distortion from the adversarial decode itself:
- 25 * rate < ~4.0  →  archive must be < **6.0 MB**
- Data payload: 314 KB  →  **model budget: ~5.7 MB**
- Raw model weights: **94.3 MB** (38.5 MB SegNet + 55.8 MB PoseNet)
- Required compression ratio: **~17x**

Stretch targets:
- Score < 3.0: models < 4.2 MB (23x)
- Score < 2.0: models < 2.7 MB (35x)

## Model Architectures

### SegNet (38.5 MB, 9.6M params)
- Architecture: `segmentation_models_pytorch.Unet` with `tu-efficientnet_b2` encoder
- Output: 5-class pixel-wise segmentation at 512x384 resolution
- Evaluation uses **argmax** of logits (discrete)
- **98.4% of params are conv weights** (125 layers)
- 78 BatchNorm layers (0.7% of params — can be folded into conv)
- 562 total layers

### PoseNet (55.8 MB, 13.9M params)
- Architecture: `timm.fastvit_t12` backbone → linear summarizer → Hydra head
- Input: 12-channel YUV6 representation of 2 consecutive frames at 512x384
- Output: 6-dim pose vector (only first 6 of 12 dims used for scoring)
- Evaluation uses **MSE** of continuous outputs
- **52.8% linear weights** (14 layers), **46.4% conv weights** (78 layers)
- 510 total layers

### Preprocessing Pipeline (must be reproduced exactly)
- SegNet: takes last frame, bilinear resize to 512x384, 3-channel RGB float input
- PoseNet: both frames → bilinear resize to 512x384 → RGB-to-YUV6 conversion → concatenate to 12 channels → normalize (mean=127.5, std=63.75)

## Weight Statistics

### SegNet Conv Weights (9.46M params)
- Range: [-2.47, 3.15]
- Mean: -0.0017, Std: 0.088
- 77.4% of weights have |w| < 0.1
- 10.0% of weights have |w| < 0.01
- p50=0.054, p75=0.095, p90=0.140, p95=0.172, p99=0.248

### PoseNet Conv Weights (6.47M params)
- Range: [-2.40, 2.22]
- Mean: -0.0004, Std: 0.094
- 74.5% within |w| < 0.1
- p50=0.057, p75=0.101, p90=0.153, p95=0.189, p99=0.269

### PoseNet Linear Weights (7.36M params) — MUCH SPARSER
- Range: [-7.63, 3.22]
- Mean: -0.004, Std: 0.049
- **95.7% within |w| < 0.1**
- **23.3% within |w| < 0.01**
- p50=0.023, p75=0.040, p90=0.067, p95=0.093, p99=0.191

## Compression Results So Far (naive approaches)

| Method | SegNet | PoseNet | Total | 25*rate | Viable? |
|--------|--------|---------|-------|---------|---------|
| Raw f32 | 38.5 MB | 55.8 MB | 94.3 MB | 62.8 | No |
| f32 bz2 | 36.4 MB | 53.1 MB | 89.5 MB | 59.7 | No |
| f16 byte-plane bz2 | 16.8 MB | 24.5 MB | 41.3 MB | 27.7 | No |
| INT8/layer bz2 | 8.0 MB | 10.4 MB | 18.4 MB | 12.5 | No |
| **INT4/layer bz2** | **3.05 MB** | **3.28 MB** | **6.33 MB** | **4.42** | Barely no |
| INT4 lzma | 2.88 MB | 3.02 MB | 5.90 MB | 4.14 | Close! |
| **INT3/layer bz2** | **2.01 MB** | **1.94 MB** | **3.95 MB** | **2.84** | **Yes** |
| INT2/layer bz2 | 1.12 MB | 1.24 MB | 2.36 MB | 1.78 | Yes (quality?) |
| INT4+50% prune bz2 | 2.87 MB | 3.09 MB | 5.96 MB | 4.18 | Close |
| INT4+70% prune bz2 | 2.24 MB | 2.63 MB | 4.87 MB | 3.45 | Maybe |
| INT4+90% prune bz2 | 1.01 MB | 1.31 MB | 2.31 MB | 1.75 | Yes (quality?) |

## SVD Analysis (Low-Rank Decomposition)

Most conv layers are **not very low-rank** — retaining 90% energy typically requires 70-85% of full rank. However:

### PoseNet has outlier layers near the output:
- `hydra.resblock.block_a.3.weight` (512, 1024): **rank 7 of 512 for 90% energy** (0.02x!)
- `summarizer.0.weight` (512, 2048): rank 214 for 90% energy (0.52x)
- `vision.head.fc.weight` (2048, 1024): rank 394 for 90% energy (0.58x)

### SegNet layers don't SVD well:
- Decoder conv layers need 70-80% of full rank for 90% energy

## Key Constraints

1. **30-minute wall time** for inflation on T4 GPU (16 GB VRAM, 26 GB RAM). Our adversarial decode currently uses ~26 min. Model decompression must be fast.
2. **The compressed model must reproduce the original model's outputs closely enough.** Adversarial frames optimized against the compressed model are evaluated by the ORIGINAL model. This is an adversarial transferability problem.
3. **SegNet is more forgiving** — argmax is discrete, small weight perturbations often don't change class predictions.
4. **PoseNet is sensitive** — continuous MSE means any output drift directly hurts the score.
5. Models load from safetensors format. The inflate script can reconstruct weights from any compressed format at runtime.

## What We Need Researched

**Primary question**: What are the most effective techniques to compress neural network weights to extreme ratios (17-35x from f32) while preserving functional equivalence of outputs?

**Specific techniques to investigate**:

1. **Quantization below INT4**: Is INT3 or mixed INT3/INT4 viable? What are best practices for per-channel vs per-group vs per-layer quantization at extreme bit-widths? Can we use non-uniform quantization (learned codebooks) to get better quality at the same bit rate?

2. **Structured pruning + quantization combos**: Can we prune channels/filters from conv layers (reducing actual computation) and then quantize the remaining weights? This changes the model architecture but preserves exact output for retained structure. What are the practical limits?

3. **Knowledge distillation to a smaller architecture**: Could we train a tiny student model (e.g., MobileNet-tiny or custom) that mimics the original SegNet/PoseNet outputs? If the student has 1M params instead of 10M, even float16 would be 2 MB. The student would need to match outputs closely enough for adversarial transferability.

4. **Weight sharing / codebook quantization**: Replace each weight with an index into a small codebook (e.g., 16 or 32 centroids per layer). The codebook + indices might compress better than raw quantized values.

5. **Low-rank factorization + quantization**: For layers where SVD works (PoseNet head layers), decompose W = UV, then quantize U and V separately. Combined with other techniques for non-low-rank layers.

6. **Entropy coding of quantized weights**: After quantization, what's the best entropy coder? bz2, lzma, ANS, arithmetic coding? Are there neural-network-specific entropy models?

7. **BatchNorm folding**: Folding BN into adjacent conv/linear layers eliminates ~1% of params and removes BN running stats. This is lossless.

8. **Task-specific compression**: We don't need the models to be good at everything — we only need SegNet's argmax on adversarially-generated frames to match, and PoseNet's 6-dim output to be close. Can we identify and discard weights/channels that don't affect these specific outputs?

9. **Self-compression / neural weight compression**: Are there methods where you train a tiny network to predict/generate the weights of the larger network? This is extreme but could theoretically get very high ratios.

10. **Adversarial transferability mitigation**: Even with perfect compression, frames optimized against the compressed model may not fool the original. Are there techniques to improve transferability? (e.g., optimizing against an ensemble of quantization levels, adding noise during optimization, using input-space smoothing)

## Available Infrastructure
- Compression can take as long as needed (offline, no time limit)
- Decompression must complete within 30 min on T4 (currently uses 26 min for adversarial decode)
- PyTorch, numpy, scipy available at decode time
- Can use any Python library installable via pip (doesn't count toward archive size)
- Models defined in `modules.py` using `timm` and `segmentation_models_pytorch`

## Success Criteria
- Combined model archive size < 5.7 MB (score < 4.4, beat baseline)
- Ideally < 4.2 MB (score < 3.0)
- Model output fidelity: SegNet argmax agreement > 99%, PoseNet output MSE < 0.1
- Decompression overhead < 60 seconds (to stay within wall time budget)
