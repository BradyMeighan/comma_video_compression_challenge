# Model Compression Status v2 — Updated with all experimental results

## The Breakthrough + The Wall

We PROVED that SVD-based compression preserves the adversarial decode gradient landscape. A 30% rank SVD of the teacher SegNet (no fine-tuning) gives seg_dist=0.077 in adversarial decode — 3.4× better than sparsity (0.263) at similar forward accuracy levels. After fine-tuning, the non-factored SVD model achieves **1.31 distortion** (seg_dist=0.012, pose_mse=0.001), which crushes the current leader at 2.90.

**The wall:** The fine-tuned model can only be stored as full reconstructed weights. At INT5, that's **5.5 MB for SegNet alone**. PoseNet hasn't been compressed at all yet (56 MB). We need both models combined at ~1.5 MB.

## Complete Experimental Results

### What Works for Gradient Preservation

| Method | Forward Acc | Adv Decode seg_dist | Size | Verdict |
|---|---|---|---|---|
| Full teacher FP32 | 100% | ~0.003 | 38.5 MB | Perfect but huge |
| Full teacher INT8 | 99.3% | 0.007 | 9.1 MB | Great transfer |
| Full teacher INT6 | 99.2% | 0.006 | 6.7 MB | Great transfer |
| Full teacher INT5 | 99.1% | 0.007 | 5.5 MB | Good transfer |
| **Non-factored SVD 30% + fine-tune** | **97.0%** | **0.012** | **5.5 MB (INT5)** | **Best compressed result** |
| Non-factored SVD 30% (no fine-tune) | 23.2% | 0.077 | ~500 KB (factors) | Gradients OK, accuracy bad |
| Non-factored SVD 20% (no fine-tune) | 18.3% | 0.152 | ~350 KB (factors) | Gradients decent |
| Non-factored SVD 10% (no fine-tune) | 19.5% | 0.224 | ~200 KB (factors) | Gradients marginal |

### What Fails for Gradient Preservation

| Method | Forward Acc | Adv Decode seg_dist | Why It Failed |
|---|---|---|---|
| Full teacher INT4 (PTQ) | 63.1% | 0.054 | Quantization noise too large |
| Unstructured sparsity 84% | 99.3% | 0.263 | Zeros in Jacobian destroy gradient pathways |
| MobileUNet student | 99.75% | 0.164 | Architecture mismatch = Jacobian mismatch |
| On-policy MobileUNet | 99.0% | 0.164 | Same architecture mismatch |
| **Factored SVD 20% + fine-tune** | **97.2%** | **0.269** | **Factored training creates gradient shortcuts** |
| Color distance proxy (0 params) | 96.6% | 0.034 | No temporal info for pose (pose_mse=44) |

### The Factored SVD Trap

We attempted to store SVD factors (U, V) directly instead of full reconstructed weights. Two approaches both failed:

1. **Factor then fine-tune**: Replace Conv2d with FactoredConv2d (V spatial → U pointwise), fine-tune U and V directly. Gets 97.2% forward accuracy but 26.9 distortion. Fine-tuning U, V independently gives the optimizer freedom to find shortcuts that match outputs but produce wrong gradients on OOD inputs.

2. **Fine-tune then re-factor**: Fine-tune the non-factored SVD model (works great, 1.31 distortion), then re-decompose via SVD for storage. Gives 49.5% accuracy at ALL rank levels — fine-tuning moved the weights completely off the low-rank manifold.

## Latest Experimental Results (Round 3)

### LoRA-on-SVD — FAILED
- Froze SVD-compressed weights (30% rank), trained only a low-rank residual ΔW = A·Bᵀ (rank 8, 648K LoRA params = 1265KB FP16)
- Forward accuracy: 98.6% after 15 epochs
- Adversarial decode: seg_dist=0.264, distortion=26.53 — same failure as factored SVD
- Why: even a small additive LoRA correction changes the computation graph. Gradient flows through (W_base + A·Bᵀ), producing different Jacobians than W_base alone
- Conclusion: ANY modification to the forward pass that adds trainable parameters on top of SVD base destroys gradient quality

### Riemannian Fine-tuning (Geoopt) — FAILED
- Parameterized weights as U @ V with SVD retraction after each epoch to stay on low-rank manifold
- Forward accuracy: 97.5% after 2 epochs
- Adversarial decode: seg_dist=0.270, distortion=27.07 — identical failure to factored SVD
- Why: backpropagating through the U @ V matrix product produces fundamentally different Jacobians than backpropagating through a single weight matrix W, even when U @ V == W numerically
- Conclusion: the factored computation graph (F.conv2d(x, U @ V)) has different ∂L/∂x than the direct graph (F.conv2d(x, W))

### SVDQuant (Outlier Isolation + INT4) — WORKS
- Applied to the non-factored SVD fine-tuned model (the one with 1.31 distortion)
- Extract top 5% singular values as FP16 "outlier branch", quantize residual to INT4
- Forward accuracy: 97.97%
- Adversarial decode: seg_dist=0.015, pose_mse=0.0007, distortion=1.62 — WORKS!
- Compressed size: 5256 KB — smaller than INT5 (5500KB) with comparable gradient quality
- This proves outlier isolation enables INT4 on the non-factored model

### Codebook Quantization (k-means) — MIXED
- 16 centroids (4-bit): 42.47% accuracy, distortion=17.04 — FAILED (per-tensor k-means too aggressive)
- 32 centroids (5-bit): 97.39% accuracy, 4872 KB — promising but adversarial decode not tested

### SIREN (Coordinate-Based Network) — INCONCLUSIVE
- 150K params (294KB at FP16), maps (frame_id, x, y) → segmentation logits
- Accuracy after 10 epochs: 70% (still learning fast, 48% → 70%)
- Adversarial decode test crashed (forward() incompatible with pixel-based optimization)
- Needs redesigned forward pass and more training epochs to evaluate properly

### Geometric Pose Proxy — PARTIALLY WORKS
- Photometric multi-scale loss between frame pairs, zero parameters
- With SVD SegNet: pose_mse reduced from 104 (no optimization) to 5.5 (20× improvement)
- But sqrt(10 × 5.5) = 7.4 points — still too high for competitive score (need <1 point)
- Higher photometric weights (2.0, 5.0) and warp-based variants all performed worse
- Conclusion: gets us from terrible to mediocre on pose, but not competitive

## Key Findings Summary

The definitive pattern across ALL experiments:
1. **Non-factored weights with full W**: gradient landscape preserved (INT5-INT8, SVDQuant all work)
2. **Any factored/modified computation graph** (U@V, LoRA, different architecture): gradient landscape destroyed
3. **The boundary is the computation graph, not the weight values** — even mathematically equivalent computations (U@V == W) produce different autodiff Jacobians

The remaining viable paths:
1. **SVDQuant pushed harder** — vary outlier % and residual bit depth. Currently 5.2MB, need 500KB. Huge gap.
2. **SIREN** — needs fundamental redesign for adversarial decode compatibility, but 294KB is the right size
3. **Codebook quantization with QAT** — 5-bit codebook showed 97.4% acc, might preserve gradients
4. **GaLore** — project gradients into low-rank subspace during training (untested)

### Key Insight: The Compression-Gradient Paradox

Fine-tuning is NECESSARY to get distortion below ~8 (SVD without fine-tuning gives 7.8). But fine-tuning makes the model INCOMPRESSIBLE:
- Before fine-tuning: weights are low-rank (compressible) but gradients are approximate (7.8 distortion)
- After fine-tuning: weights are full-rank (incompressible) but gradients are excellent (1.31 distortion)

This is the fundamental tension. Nuclear norm regularization during fine-tuning (to keep weights low-rank) didn't work — the penalty was too weak to overcome the task loss, and stronger penalties hurt accuracy.

## What We Need Solved

### Problem 1: SegNet compression (38.5 MB → target ~1 MB)

The non-factored SVD + fine-tune approach gives 1.31 distortion at 5.5 MB (INT5). We need 4× more compression without losing gradient quality.

**Unsolved questions:**
- Can INT4 QAT recover the full-teacher INT4 from 63% to 99%? (INT4 would give ~4.2 MB — still too big but closer)
- Can we fine-tune while constraining weights to stay on the low-rank manifold? (nuclear norm failed, need stronger approach)
- Can the factored model be saved from its gradient problem? (adding adversarial decode quality as a direct training objective?)
- Is there a way to store the fine-tuned weights more compactly than INT5 full weights? (codebook quantization, mixed precision, delta coding?)

### Problem 2: PoseNet compression (56 MB → target ~0.5 MB or elimination)

PoseNet hasn't been seriously attempted yet. Options:
- **PoseLUT** (216 KB): Stores exact teacher pose vectors with tiny CNN interpolation. Never tested with a GOOD SegNet — only tested with color proxy (failed because flat frames have no temporal info). With SVD SegNet producing structured frames, PoseLUT might work.
- **SVD PoseNet**: Same approach as SegNet. PoseNet has 13.9M params (FastViT-T12 + linear layers). Would need similar SVD + fine-tune pipeline.
- **Geometric pose proxy** (0 bytes): Differentiable Normal Flow or EPro-PnP solver. Computes pose from optical flow between frames — no model needed. Risk: needs texture in frames to compute flow, which early adversarial decode iterations lack.
- **Skip pose entirely**: Just optimize SegNet, accept pose penalty. From color proxy test: pose_mse ≈ 44 → sqrt(10×44) = 21 points. Way too much.

### Problem 3: Score math

Current leader: **1.95**. To win convincingly we need to be well under that.

```
score = distortion + 25 × (archive_size / 37,545,489)
```

Every 1 MB of archive adds 25 × (1/37.5) = **0.68 points** to the score.

| Archive Size | Rate Penalty | Distortion Budget (to beat 1.95) | Distortion Budget (to hit 1.00) |
|---|---|---|---|
| 500 KB | 0.34 | 1.61 | 0.66 |
| 750 KB | 0.51 | 1.44 | 0.49 |
| 1.0 MB | 0.68 | 1.27 | 0.32 |
| 1.5 MB | 1.02 | 0.93 | impossible |
| 2.0 MB | 1.37 | 0.58 | impossible |
| 3.0 MB | 2.05 | impossible | impossible |
| 5.5 MB | 3.76 | impossible | impossible |

The archive includes targets (~300 KB) + SegNet model + PoseNet model (or proxy). So the model budget is archive minus ~300 KB.

**To realistically win:** We need combined models at **500 KB or less**. At 500 KB models + 300 KB targets = 800 KB archive → rate penalty 0.54. With distortion 1.31, total = 1.85 (beats leader). With distortion 0.5, total = 1.04 (dominant).

At 1 MB models: archive = 1.3 MB → rate 0.89 → need distortion < 1.06 to beat leader.
At 1.5 MB models: archive = 1.8 MB → rate 1.23 → need distortion < 0.72. Very tight.
At 5.5 MB (our current INT5 SegNet alone): rate 3.76 → impossible even with zero distortion.

## Architecture Details (for reference)

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

### The Adversarial Decode Pipeline
- Initialize frames with ideal class colors from stored target segmentation maps
- Gradient descent through SegNet + PoseNet: 150 iterations, AdamW optimizer
- Frames start as flat colored blobs → converge to frames that fool evaluation models
- Must run within 30 minutes on T4 GPU (16GB VRAM)

## Available Assets
- `compressed_models/segnet_svd_finetuned.pt`: Non-factored SVD 30% fine-tuned model (97% acc, 1.31 distortion, 5.5 MB INT5)
- `distill_data/trajectory/`: 4,200 adversarial decode trajectory frames with teacher logits
- `distill_data/`: 600 original frames, teacher logits, pose vectors, base segmentation map
- `tiny_models/pose_lut.pt`: PoseLUT student (49K params, 216 KB) — untested with SVD SegNet
- All teacher weights in `models/`

## What Research Should Focus On

1. **How to fine-tune while constraining low-rank structure** — the nuclear norm approach was too weak. Is there a way to parameterize the optimization so that weights CANNOT leave the low-rank manifold? (e.g., Riemannian optimization on the Grassmann manifold?)

2. **How to make factored fine-tuning preserve gradients** — the FactoredConv2d approach gives small models but broken gradients. Can we add a loss term during factored training that directly penalizes gradient landscape divergence, without the VRAM explosion of full KDIGA? (torch.func.jvp doesn't work with timm's BatchNorm due to in-place operations)

3. **Alternative compact differentiable architectures** — hash-grid + MLP, SIREN, or other implicit representations that are natively compact and designed for memorization. These avoid the compression problem entirely.

4. **Better quantization** — codebook (k-means) quantization, mixed precision (HAWQ), or learned step sizes (LSQ). Can these close the gap between INT5 (works, 5.5 MB) and INT4 (broken, 4.2 MB)?

5. **PoseNet elimination** — is the differentiable geometric proxy (Normal Flow / EPro-PnP) viable? Or can PoseLUT work when paired with an SVD SegNet that produces structured frames?
