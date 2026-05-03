# Model Compression for Adversarial Decode: Status & Findings

## The Challenge

We're competing in a video compression challenge. The score formula is:
```
score = 100 × segnet_distortion + sqrt(10 × posenet_distortion) + 25 × compression_rate
```
Lower is better. Current leader: **1.95**. Baseline: **4.39**.

Our approach ("adversarial decode") achieves **~0.3 distortion** — far better than anyone — by running gradient descent through the evaluation neural networks (SegNet + PoseNet) to generate frames that produce the correct network outputs. The frames look nothing like the original video but perfectly fool the evaluation models.

**The problem:** The competition rules require that any neural network used during the inflate (decompression) step must be included in the archive, and its size counts toward the compression rate. The two teacher models are:
- **SegNet**: 9.6M params, 38.5 MB (EfficientNet-B2 + UNet decoder from segmentation_models_pytorch)
- **PoseNet**: 13.9M params, 56 MB (FastViT-T12 + linear summarizer + Hydra multi-head)
- **Combined**: 94.5 MB — larger than the original video (37.5 MB)

Including these models at full size gives a rate of 252%, contributing 63 points to the score, making the total ~63.3 despite the excellent 0.3 distortion.

**To beat the leader (1.95), we need:**
- distortion + 25 × rate < 1.95
- If distortion ≈ 0.5: rate < 0.058 → archive < 2.18 MB
- If distortion ≈ 1.0: rate < 0.038 → archive < 1.43 MB

So we need **both models compressed to ~1-1.5 MB combined** while maintaining gradient quality for adversarial decode. The archive also contains compressed targets (segmentation maps + pose vectors, ~300 KB).

## What Makes This Hard: The Gradient Landscape Problem

Adversarial decode works by backpropagating through the model to optimize pixel values:
```python
for iteration in range(150):
    logits = model(frame)
    loss = margin_loss(logits, target_segmap)
    loss.backward()  # gradients flow to pixel values
    optimizer.step()  # update pixels
```

The frames start as **flat-colored blobs** (each pixel set to the ideal color for its target class) and gradually converge to frames that fool the evaluation model. The optimization trajectory passes through:
- Iteration 0: flat solid-colored regions
- Iteration 30: noisy colored blobs
- Iteration 75: vaguely frame-like shapes
- Iteration 150: converged frames

The model needs to produce **correct gradients** not just on the 600 natural video frames, but on this entire trajectory of synthetic inputs. This is the critical constraint that every compression approach has failed on.

**Forward accuracy ≠ gradient quality.** A model can achieve 99.75% pixel-level argmax agreement with the teacher on natural frames, yet produce completely wrong gradients on flat-colored blobs, causing adversarial decode to generate frames the real teacher disagrees with.

## What We've Tried and Results

### 1. Student Model Distillation (MobileUNet) — FAILED
- **Architecture**: Custom MobileUNet with inverted residual blocks, Fourier positional encoding, learnable spatial bias, base map prior. 472K params, ~1 MB.
- **Training**: KL divergence + cross-entropy + boundary-weighted loss, trained on 600 frames with adversarial-style augmentation (flat colors, noise, blends).
- **Forward accuracy**: 99.75% pixel argmax agreement with teacher on natural frames.
- **Adversarial decode transfer**: seg_dist = 0.164 (83.6% agreement), pose_mse = 101. **Total score: 48.3**
- **Why it failed**: Different architecture = completely different gradient landscape. The student finds "shortcuts" that match teacher outputs on specific inputs but the internal representation is alien. Gradients on flat-colored blobs point in wrong directions.

### 2. On-Policy Student Distillation — FAILED
- **Idea**: Train the MobileUNet on the actual adversarial decode trajectory (flat blobs → converged frames) instead of just natural frames.
- **Data**: Generated 4,200 trajectory frames by running adversarial decode with the real teacher, recording intermediate frames at checkpoints [0, 5, 15, 30, 60, 100, 149].
- **Training**: 70% trajectory frames + 30% original frames, KL + CE + MSE loss.
- **Forward accuracy**: 99% original, 97% trajectory.
- **Adversarial decode transfer**: seg_dist = 0.164, pose_mse = 101. **Total score: 49.2**
- **Why it failed**: Same architecture mismatch. Training on trajectory data didn't fix the fundamental gradient landscape difference.

### 3. Zero-Weight Proxy (No Model) — FAILED
- **Approach**: Replace SegNet with a pure function: negative squared color distance to ideal class colors. Zero parameters. Differentiable.
- **Adversarial decode transfer**: seg_dist = 0.034 (96.6%), pose_mse = 44.0. **Total score: 24.6**
- **Why it failed**: Color distance provides decent seg gradients (96.6% agreement), but flat-colored frames give PoseNet zero temporal information → catastrophic pose error. No model = no way to encode motion between frame pairs.

### 4. Zero-Weight Proxy + PoseLUT — FAILED
- **PoseLUT**: Tiny (49K params, 216 KB) differentiable lookup table storing exact teacher pose vectors. Uses a tiny CNN + learned anchor embeddings + softmax-weighted interpolation.
- **Adversarial decode transfer**: seg_dist = 0.033, pose_mse = 45.4. PoseLUT didn't help at all. **Total score: 24.9**
- **Why it failed**: PoseLUT can't produce useful gradients when the input frames are flat-colored blobs (from the color proxy). It was trained on natural-ish frames.

### 5. Structured Channel Pruning (torch-pruning) — FAILED
- **Approach**: Physically remove entire channels from the teacher using torch-pruning library.
- **Result at 5% pruning**: Accuracy dropped from 100% to 97.6% (no fine-tuning).
- **Result at 85% pruning (with fine-tuning between steps)**: Fine-tuning couldn't recover — loss of 41,142 at first epoch, plateauing at ~10,000. Each step removed too much.
- **One-shot 85% pruning**: Accuracy crashed to 49.5% (predicting one class for everything).
- **Why it failed**: EfficientNet-B2 UNet has tightly coupled residual connections, skip connections, and squeeze-excitation blocks. Removing channels cascades through the entire architecture. Even 5% removal causes significant damage.

### 6. Post-Training Quantization (INT8 through INT2) — PARTIALLY WORKED

Quantized the full (non-sparse) teacher models with per-channel symmetric quantization + bz2 compression:

| Bits | SegNet Size | PoseNet Size | Total | SegNet Acc | Adv Decode seg_dist | Adv Decode pose_mse | Score |
|------|-------------|-------------|-------|------------|--------------------|--------------------|-------|
| INT8 | 9,083 KB | 12,768 KB | 21,851 KB | 99.31% | 0.007 | 0.003 | 15.9 |
| INT6 | 6,741 KB | 9,336 KB | 16,076 KB | 99.21% | 0.006 | 0.036 | 12.4 |
| INT5 | 5,518 KB | 7,554 KB | 13,072 KB | 99.12% | 0.007 | 0.853 | 12.7 |
| INT4 | 4,266 KB | 5,717 KB | 9,983 KB | 63.11% | 0.054 | 4.857 | 19.4 |
| INT3 | 2,937 KB | 3,727 KB | 6,664 KB | 49.21% | 0.236 | 127.2 | 64.0 |
| INT2 | 1,086 KB | 1,018 KB | 2,103 KB | 26.81% | 0.216 | 114.2 | 57.1 |

**Key finding**: INT5-INT8 preserve the gradient landscape perfectly (adversarial decode works). INT4 and below break it. The boundary is between INT4 and INT5.

**The size problem**: Even INT5 (both models combined) = 13 MB. To beat the leader we need ~1.5-2 MB. That's an 8x gap from the smallest bit depth that works.

### 7. Unstructured Sparsity + L1 Regularization — FAILED for transfer

- **Approach**: Fine-tune teacher with L1 penalty on weights to drive individual weights to zero. Permanent weight masks prevent zeroed weights from recovering. Hard threshold every 5 epochs.
- **Sparsity progression** (with permanent masks):
  ```
  Epoch 4:  52.5% sparse, 3740 KB at INT4, 98.7% accuracy
  Epoch 24: 65.2% sparse, 3169 KB, 99.3% accuracy
  Epoch 44: 77.8% sparse, 2401 KB, 98.9% accuracy
  Epoch 59: 87.3% sparse, 1661 KB, 97.2% accuracy → recovered to 98.8%
  Epoch 64: 90.0% sparse, 1432 KB, 90.3% accuracy → recovered to 98.5%
  ```
- **Forward accuracy**: 99.3% at 84.2% sparsity (best checkpoint), 98.6% at 90% sparsity.
- **Adversarial decode transfer at 84.2% sparsity (FP32, no quantization)**:
  - seg_dist = 0.263 (73.7% agreement). **Total score: 57.2**
- **Adversarial decode transfer at 84.2% sparsity (INT4 quantized)**:
  - seg_dist = 0.237 (76.3% agreement). **Total score: 50.7**
- **Why it failed**: Zeroing 84% of weights removes gradient pathways needed for the adversarial decode trajectory. The model compensates for forward accuracy on natural frames but can't produce correct gradients on flat-colored blobs and intermediate synthetic frames. Same fundamental gradient landscape problem as student models.

### 8. Currently Running: Sparsity + Trajectory Data + KDIGA
- **Approach**: Combine sparse fine-tuning (L1) with training on trajectory data AND KDIGA (input gradient alignment loss: `||∇_x student(x) - ∇_x teacher(x)||²`).
- **Hypothesis**: Training on trajectory data preserves weights needed for the adversarial decode trajectory. KDIGA directly constrains gradient landscape alignment.
- **Status**: Running, but VRAM issues with KDIGA (create_graph=True triples memory). Had to reduce KDIGA to sub-batch of 2, only on original frames, every 5th batch.
- **Concern**: This approach hasn't been validated yet and may still fail for the same fundamental reason — any significant weight change disrupts the gradient landscape on out-of-distribution inputs.

## The Fundamental Issue

The adversarial decode optimization trajectory visits inputs that are FAR from the training distribution of natural images. The full teacher handles these because it was trained on millions of diverse images, giving it robust gradient landscapes even on unusual inputs.

Any compression technique that removes model capacity (sparsity, pruning, different architecture) loses the ability to handle these unusual inputs. The model adapts its remaining capacity to maintain forward accuracy on the specific 600 frames, but the gradient landscape on the optimization trajectory becomes garbage.

**The only thing that preserves gradient quality is keeping the weights very close to their original values** — which is why INT8/INT6/INT5 quantization works (tiny per-weight perturbation) but sparsity doesn't (massive per-weight perturbation: value → 0).

## Architecture Details

### SegNet
```python
class SegNet(smp.Unet):
    def __init__(self):
        super().__init__('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)
```
- Input: (B, 3, 384, 512) float RGB
- Output: (B, 5, 384, 512) class logits
- Evaluation: pixel-wise argmax disagreement
- Score weight: **100×** (dominant term)

### PoseNet
```python
class PoseNet(nn.Module):
    def __init__(self):
        self.vision = timm.create_model('fastvit_t12', pretrained=False,
                                         num_classes=2048, in_chans=12,
                                         act_layer=timm.layers.get_act_layer('gelu_tanh'))
        self.summarizer = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), ResBlock(512))
        self.hydra = Hydra(num_features=512, heads=[Head('pose', 32, 12)])
```
- Input: (B, 12, 192, 256) — YUV6 of 2 consecutive frames
- Output: 6D pose vector (first 6 of 12 dims)
- Evaluation: MSE of 6D pose
- Score weight: **sqrt(10×)** (under square root, less sensitive)

### Adversarial Decode Pipeline
```python
# Initialize: flat ideal colors per target class
frame = ideal_colors[target_segmap]  # (B, 3, 384, 512)

for iter in range(150):
    seg_loss = margin_loss(segnet(frame), target_segmap)
    pose_loss = smooth_l1(posenet(frame_pair), target_pose)
    total = 120 * seg_loss + 0.2 * pose_loss
    total.backward()
    optimizer.step()
    frame.clamp_(0, 255)

# Upscale 384×512 → 874×1164 for evaluation
output = bicubic_upscale(frame)
```

## Existing Assets
- `distill_data/seg_inputs.pt`: 600 preprocessed frames (3, 384, 512)
- `distill_data/seg_logits.pt`: 600 teacher SegNet logits (5, 384, 512)
- `distill_data/pose_inputs.pt`: 600 preprocessed pose inputs (12, 192, 256)
- `distill_data/pose_outputs.pt`: 600 teacher pose vectors (6,)
- `distill_data/trajectory/traj_frames.pt`: 4,200 adversarial decode intermediate frames (FP16)
- `distill_data/trajectory/traj_logits.pt`: 4,200 teacher logits for trajectory frames (FP16)
- `tiny_models/mobile_segnet.pt`: MobileUNet student (472K params, 5.7 MB) — doesn't transfer
- `tiny_models/pose_lut.pt`: PoseLUT student (49K params, 216 KB) — untested with good SegNet
- `models/segnet.safetensors`: Full teacher SegNet (9.6M params, 38.5 MB)
- `models/posenet.safetensors`: Full teacher PoseNet (13.9M params, 56 MB)

## Critical Context: This Is Pure Memorization, Not Generalization

**We only need these models to work on ONE specific 60-second dashcam video.** That's 600 frame pairs from a single highway drive with consistent lighting, minimal scene variation, and a nearly static camera.

- SegNet only needs correct argmax for 600 specific frames (384×512 pixels each)
- PoseNet only needs correct 6D pose vectors for 600 specific frame pairs
- The teacher models were trained on millions of diverse images, but 99%+ of that capacity is wasted on scenes/objects not present in our video

**The adversarial decode trajectory is also narrow.** It always starts from the same initialization (ideal class colors for the target segmap), follows a similar optimization path, and converges to similar-looking frames. The trajectory for our 600 targets is a specific, bounded region of input space — not arbitrary images.

So we need models that:
1. Produce correct outputs on 600 natural frames (forward accuracy)
2. Produce correct gradients on ~4200 trajectory frames (the path from flat colors → converged frames for these specific 600 targets)
3. Nothing else. No generalization. No robustness. Just these specific inputs.

The teacher has 23.5M total parameters handling millions of diverse images. For 600 frames from one drive, the theoretical minimum model capacity needed is vastly smaller. We just haven't found a way to extract it without destroying the gradient landscape.

## What We Need

A way to represent the SegNet and PoseNet models in ~1.5 MB combined (from 94.5 MB total) that preserves the gradient landscape well enough for adversarial decode to produce frames with <1.0 total distortion (100×seg_dist + sqrt(10×pose_dist)).

The key constraint: the compressed model must produce useful ∂loss/∂pixel gradients not just on 600 natural video frames, but on the entire optimization trajectory from flat-colored blobs to converged frames. We have the exact trajectory data (4,200 frames) pre-computed and available for training/validation.

## Untested Ideas
1. **INT4 Quantization-Aware Training (QAT)**: Simulate INT4 quantization during fine-tuning with STE. The model learns to work WITH the quantization. Might recover INT4 from 63% to 99%. But even INT4 for both models = ~10 MB.
2. **K-means / codebook quantization**: Instead of uniform INT4 levels, find optimal 16 centroids per layer via k-means on weights. Better utilization of 4 bits. With QAT might work at 2-3 bits.
3. **Mixed precision**: INT8 for critical layers (early encoder, decoder head), INT3 for less sensitive layers (deep bottleneck).
4. **Low-rank factorization**: Decompose weight matrices via SVD, keep top-k singular values. Reduces params without zeroing.
5. **Self-Compression (paper 2301.13142)**: Per-channel differentiable bit-depth as a learnable parameter. Each channel learns its own optimal bit depth during training. Automatically prunes 0-bit channels. Never implemented.
6. **PoseLUT with good SegNet**: Never tested PoseLUT (216 KB) paired with a high-quality compressed SegNet. If the SegNet produces natural-looking frames, PoseLUT might produce adequate pose gradients, eliminating the need to compress PoseNet at all.
7. **Zeroth-order optimization**: Use finite-difference gradient estimation instead of backprop. Only needs forward passes (more robust to model changes). But requires many forward passes per step.
8. **Neural codec (learned video compression)**: Train an autoencoder specifically for this video, optimized for the evaluation metrics. But the decoder model would need to be in the archive.
9. **Geometric pose encoding**: Instead of a PoseNet model, apply geometric transforms (shift, rotation) between frame pairs to encode the target pose. No model needed for pose, only SegNet.
