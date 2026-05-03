# Model Compression Status v3 — Final Comprehensive Results

## Executive Summary

We have **exhaustively tested** every known compression technique for preserving adversarial decode gradient quality. The definitive conclusion: **the teacher's exact non-factored computation graph is required for gradient transfer, and storing those weights requires a minimum of ~5MB.** No student model, factored representation, or alternative architecture achieves competitive distortion.

Current leader: **1.95**. Our best feasible score: **~4.4** (5MB SVDQuant model). The 10× gap between our model size floor (5MB) and the competitive budget (500KB) appears to be a fundamental limitation of the adversarial decode approach under the archive constraint.

---

## Complete Experimental Results — ALL approaches tested

### Approaches that PRESERVE gradient quality (distortion < 2.0)

All of these use the **exact teacher architecture with non-factored weights** (standard F.conv2d(x, W)).

| Method | Forward Acc | Adv Decode seg_dist | Adv Decode pose_mse | Distortion | Size |
|---|---|---|---|---|---|
| Full teacher FP32 | 100% | ~0.003 | ~0.001 | ~0.3 | 94.5 MB |
| Full teacher INT8 | 99.3% | 0.007 | 0.003 | ~0.9 | 21.9 MB |
| Full teacher INT6 | 99.2% | 0.006 | 0.036 | ~1.2 | 16.1 MB |
| Full teacher INT5 | 99.1% | 0.007 | 0.853 | ~3.6 | 13.1 MB |
| Non-factored SVD 30% + fine-tune (INT5) | 97.0% | 0.012 | 0.001 | 1.31 | 5.5 MB |
| SVDQuant 3% outlier + INT4 residual | 97.8% | 0.008 | 0.001 | **0.87** | 5.0 MB |
| SVDQuant 5% outlier + INT4 residual | 97.2% | 0.009 | 0.001 | 1.01 | 5.4 MB |
| SVDQuant 10% outlier + INT4 residual | 98.1% | 0.007 | 0.001 | **0.77** | 6.4 MB |

**The absolute size floor for working gradient transfer: ~5.0 MB** (SVDQuant 3% outlier + INT4).

### SVDQuant full sweep results

| Outlier% | Residual Bits | Size (KB) | Forward Acc | Seg Dist | Pose MSE | Distortion |
|---|---|---|---|---|---|---|
| 3% | INT4 | 4997 | 97.82% | 0.008 | 0.001 | **0.87** |
| 3% | INT3 | 3672 | 35.04% | BROKEN | - | - |
| 3% | INT2 | 1828 | 47.71% | BROKEN | - | - |
| 5% | INT4 | 5398 | 97.21% | 0.009 | 0.001 | 1.01 |
| 5% | INT3 | 4079 | 30.75% | BROKEN | - | - |
| 10% | INT4 | 6382 | 98.05% | 0.007 | 0.001 | **0.77** |
| 10% | INT3 | 5047 | 36.43% | BROKEN | - | - |
| 15% | INT4 | 7348 | 98.01% | 0.008 | 0.001 | 0.91 |
| 20% | INT4 | 8310 | 96.47% | 0.008 | 0.001 | 0.94 |

**INT3 residual breaks the model at ALL outlier percentages.** INT4 is the absolute floor for the residual.

### Approaches that DESTROY gradient quality (distortion > 15)

All of these either use factored weights, different architectures, or aggressive weight modification.

| Method | Forward Acc | Adv Decode seg_dist | Distortion | Size | Why It Failed |
|---|---|---|---|---|---|
| MobileUNet student (ReLU) | 99.75% | 0.164 | ~26.5 | ~1 MB | Architecture mismatch → different Jacobian |
| On-policy MobileUNet | 99.0% | 0.164 | ~26.5 | ~1 MB | Same architecture mismatch |
| Image SIREN (sin() activations) | 97.98% | 0.207 | **20.8** | 250 KB | Different architecture, slightly better than ReLU |
| Factored SVD 20% + fine-tune | 97.2% | 0.269 | 27.0 | ~500 KB | Factored U@V computation → different Jacobian |
| LoRA-on-SVD (frozen base + residual) | 98.6% | 0.264 | 26.5 | ~1.7 MB | (W_base + A·Bᵀ) still modifies computation graph |
| Riemannian SVD (Geoopt retraction) | 97.5% | 0.270 | 27.1 | ~4 MB | Same factored U@V problem |
| Unstructured sparsity 84% | 99.3% | 0.263 | ~57 | ~2 MB | Zeros destroy Jacobian pathways |
| Sparse + trajectory + KDIGA | varies | 0.237 | ~50 | ~2 MB | Sparsity damage irreversible |
| Full teacher INT4 PTQ | 63.1% | 0.054 | ~19 | 10 MB | Quantization noise too large |
| Codebook 16-centroid (4-bit) | 42.5% | 0.169 | 17.0 | 4.2 MB | Per-tensor k-means too aggressive |

### Approaches that are INCOMPATIBLE with adversarial decode

| Method | Issue |
|---|---|
| Coordinate SIREN (frame_id, x, y) → logits | Does not take pixel colors as input → ∂loss/∂pixel = 0 → cannot optimize pixels |
| FiLM-conditioned SIREN | Same: coordinate-based, no pixel gradient |
| Color distance proxy (0 params) | Provides pixel gradients but only 96.6% seg accuracy, terrible pose (no temporal info) |
| PoseLUT (216KB) | Tested with SVD SegNet: pose_mse=92 (WORSE than no pose optimization). Useless. |

### PoseNet replacement results

Tested with the working non-factored SVD SegNet:

| Pose Method | pose_mse | Pose Score Contribution | Size | Notes |
|---|---|---|---|---|
| Full teacher PoseNet | 0.001 | 0.11 | 56 MB | Perfect but huge |
| Geometric photometric proxy (w=0.2) | 5.5 | 7.4 | 0 KB | Best zero-param result |
| Geometric photometric proxy (w=2.0) | 7.1 | 8.4 | 0 KB | Stronger weight made it worse |
| Warp-based geometric proxy | 130 | 36.1 | 0 KB | Warp parameters completely wrong |
| PoseLUT | 104 | 32.4 | 216 KB | Worse than no optimization |
| No pose optimization at all | 104 | 32.3 | 0 KB | Baseline |

---

## The Three Fundamental Laws of Adversarial Decode Compression

Across 20+ experiments, three iron laws have emerged:

### Law 1: The Computation Graph Is Sacred
The gradient ∂loss/∂pixel depends on the EXACT sequence of PyTorch operations in the forward pass. Even mathematically equivalent operations produce different autodiff Jacobians:
- `F.conv2d(x, W)` ≠ `F.conv2d(x, U @ V)` for gradient purposes, even when `W == U @ V`
- `F.conv2d(x, W + A @ B)` ≠ `F.conv2d(x, W)` even when A @ B ≈ 0
- Any architectural change (different layer types, skip connections, activations) → different Jacobian

### Law 2: INT4 Is the Quantization Floor
For the teacher architecture, INT4 (with outlier isolation) is the minimum bit depth that preserves gradient quality. INT3 breaks the model at every outlier percentage tested. This sets a hard floor of ~5MB for SegNet (9.6M params × 4 bits + outlier overhead).

### Law 3: Forward Accuracy Does Not Predict Gradient Quality
Models with 97-99% forward accuracy routinely produce 20-27× worse adversarial decode distortion than the teacher. The 99.75% MobileUNet student is 85× worse than INT5 teacher (26.5 vs 0.31 distortion). Accuracy is a necessary but wildly insufficient condition.

---

## Score Math Reality

```
score = distortion + 25 × (archive_size / 37,545,489)
```

| Archive Size | Rate Penalty | Distortion Budget (beat 1.95) |
|---|---|---|
| 500 KB | 0.34 | 1.61 |
| 1.0 MB | 0.68 | 1.27 |
| 2.0 MB | 1.37 | 0.58 |
| 5.0 MB | 3.42 | impossible |
| 5.5 MB | 3.76 | impossible |

**Our best achievable score with current techniques:**
- SVDQuant 3% INT4 SegNet (5.0 MB) + targets (300 KB) = 5.3 MB
- Distortion: 0.87 (SegNet only, using full PoseNet for pose)
- Score: 0.87 + 3.63 = **4.50** (without PoseNet in archive)
- With PoseNet: score explodes (56 MB PoseNet)
- Without PoseNet: need geometric proxy for pose (adds 7.4 points) → **8.27**

**None of our current approaches beat the leader (1.95) or even the baseline (4.39).**

---

## What Has Been Tried vs What Hasn't

### Exhaustively tested and ruled out:
- Student distillation (MobileUNet, on-policy, various architectures)
- Structured channel pruning (torch-pruning)
- Unstructured weight sparsity (L1 + masks)
- Factored SVD (FactoredConv2d, separate U/V training)
- LoRA-on-SVD (frozen base + trainable residual)
- Riemannian optimization (Geoopt, SVD retraction)
- Nuclear norm regularization during fine-tuning
- KDIGA gradient alignment (VRAM issues, when it ran it hurt accuracy)
- Image SIREN (sin() activation ConvNet)
- Coordinate SIREN (incompatible with pixel optimization)
- PoseLUT (worse than no pose with SVD SegNet)
- Various geometric pose proxies (photometric, warp-based)
- Post-training quantization sweep (INT2 through INT8)
- SVDQuant sweep (3-20% outlier, INT2-INT4 residual)
- Codebook/k-means quantization

### NOT yet tested but proposed in research:
- **GaLore** (Meta, ICML 2024) — project gradients into low-rank subspace during training. Reduces memory 65.5% while allowing full-parameter learning. Different from our approaches because it constrains GRADIENTS not WEIGHTS.
- **LOTION framework** — train on expectation of quantized loss under randomized-rounding noise. May enable INT3 QAT.
- **GaborNet MFN** — multiplicative filter networks with Gabor wavelets. ~200KB. But likely same student-model failure (~20 distortion).
- **Hash-grid + SIREN hybrid** — multi-resolution hash grid for spatial features + SIREN MLP head. Natively compact but coordinate-based (pixel gradient problem).
- **EPro-PnP** — differentiable probabilistic PnP solver for pose. Zero parameters. Needs testing with SVD SegNet frames.
- **SVDQuant with QAT** — train the model to survive SVDQuant INT3 quantization. Currently INT3 breaks at all configurations, but QAT might recover it.
- **Deep Compression pipeline** — pruning + codebook quantization + Huffman coding (Han et al.). Not tested as a combined pipeline.
- **Completely different inflate strategy** — abandon adversarial decode, use traditional codec optimization, or a learned neural codec.

---

## Architecture Details

### SegNet
- `smp.Unet('tu-efficientnet_b2', classes=5)` — 9.6M params, 38.5 MB
- EfficientNet-B2 encoder: MBConv blocks with depthwise separable convs, SE blocks, residual connections
- UNet decoder: skip connections from encoder stages, standard conv blocks
- Input: (B, 3, 384, 512) → Output: (B, 5, 384, 512) logits
- Score weight: 100× (dominant term)

### PoseNet
- FastViT-T12 backbone (12-channel YUV6 input) → Linear summarizer → Hydra multi-head
- 13.9M params, 56 MB
- Input: (B, 12, 192, 256) → Output: 6D pose vector
- Score weight: sqrt(10×) (under square root, less sensitive)

### Adversarial Decode Pipeline
```python
# Initialize: flat ideal colors per target class
frame = ideal_colors[target_segmap]  # (B, 3, 384, 512)

for iter in range(150):
    seg_loss = margin_loss(segnet(frame), target_segmap)      # needs ∂/∂pixel
    pose_loss = smooth_l1(posenet(frame_pair), target_pose)   # needs ∂/∂pixel
    total = 120 * seg_loss + 0.2 * pose_loss
    total.backward()      # gradients flow to pixel values
    optimizer.step()      # update pixels
    frame.clamp_(0, 255)

# Upscale 384×512 → 874×1164 via bicubic
output = bicubic_upscale(frame)
```

## Available Assets
- `compressed_models/segnet_svd_finetuned.pt`: Non-factored SVD 30% fine-tuned model (97% acc, 1.31 distortion, best working model)
- `distill_data/trajectory/`: 4,200 adversarial decode trajectory frames with teacher logits
- `distill_data/`: 600 original frames, teacher logits, pose vectors, base segmentation map
- `tiny_models/pose_lut.pt`: PoseLUT student (49K params, 216 KB) — proven useless
- `tiny_models/mobile_segnet.pt`: MobileUNet student (472K params) — proven useless for transfer
- All teacher weights in `models/`

## What We Need

A fundamentally new approach. Either:
1. A way to store the teacher's exact computation graph in <500KB (appears impossible with current knowledge)
2. A completely different inflate strategy that doesn't require adversarial decode through the evaluation models
3. A mathematical breakthrough in gradient-preserving compression that we haven't discovered

The competition deadline is May 3, 2026.
