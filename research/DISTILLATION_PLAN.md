# Distillation Plan: Memorize One Video's SegNet/PoseNet Outputs

## The Actual Problem

We do NOT need generalizable models. We need tiny neural networks that:
1. **SegNet student**: Given a 3×384×512 RGB frame, output 5-class logits where the argmax matches the teacher SegNet's argmax on >99.5% of pixels — but ONLY for frames from one specific dashcam video and adversarial-decode-style synthetic frames derived from it.
2. **PoseNet student**: Given a 12×192×256 YUV6 frame pair, output a 6-dim pose vector close to the teacher PoseNet's output — same narrow distribution.

These models are used during adversarial decode (gradient descent to generate frames). They need to be **differentiable** and produce **useful gradients** — but only on the optimization trajectory from flat-colored-region initialization to converged adversarial frames.

## What We Know So Far

- A 422K-param MicroUNet (base_ch=32) plateaus at ~97.6% argmax agreement after 30 epochs of KD training. The curve is flattening. This is not enough — we need 99.5%+.
- The bottleneck is NOT the training data (600 frames is plenty for memorization) — it's that a generic small UNet can't approximate the teacher's decision boundaries precisely enough.
- Training is slow because the teacher must run on augmented samples every other epoch.

## Why Standard KD Is Failing

The teacher SegNet is an EfficientNet-B2 UNet (9.6M params). Its decision boundaries at pixel level are complex — shaped by deep features, skip connections across 5 encoder levels, and channel attention. A 3-level MicroUNet with 32 channels simply can't represent these boundaries, even on one video.

The remaining ~2.4% error is likely at **class boundaries** — pixels where the teacher's prediction depends on subtle context that the tiny model can't capture. Standard KD with soft logits helps but can't close this gap because the student lacks the representational capacity to model those boundary regions.

## Research Questions for Deep Research

### Primary Question
**How do we build a <500K-param model that achieves >99.5% pixel-level argmax agreement with a 9.6M-param segmentation teacher on a fixed set of 600 dashcam frames?**

This is a memorization + function approximation problem, NOT a generalization problem.

### Specific Approaches to Investigate

#### 1. Spatial Coordinate Conditioning
Dashcam segmentation is heavily position-dependent:
- Sky is always in the top ~40% of the image
- Road is always in the bottom ~40%
- The horizon line is roughly fixed
- Cars/objects appear in predictable vertical bands

**Idea**: Add normalized (x, y) coordinate channels as model input. This gives the model an explicit spatial prior, letting it learn position-dependent rules like "if y < 0.3, predict sky" with very few parameters. Could dramatically reduce the capacity needed.

**Variants**:
- CoordConv (Liu et al., 2018): concatenate (x, y) coordinate maps as extra input channels
- Fourier positional encoding: encode (x, y) as sin/cos at multiple frequencies, allowing the model to learn sharp spatial boundaries with few params
- Learnable spatial bias: add a per-pixel learnable bias to the output logits (5×384×512 = 983K params but could be heavily quantized since most pixels are confidently one class)

#### 2. Per-Pixel Learnable Lookup + Small Refinement Network
Since we only have 600 frames from one video with a nearly static scene:
- Store a **base segmentation map** (the most common class per pixel across all 600 frames) — this is essentially free (one 384×512 map of 5 classes ≈ compressed to <1 KB)
- Train a tiny network that outputs **corrections/deltas** from this base map
- The corrections only need to handle the ~5-10% of pixels that change between frames (moving cars, etc.)
- This decomposes the problem: static regions (cheap) + dynamic regions (need a model)

#### 3. Train Big, Self-Compress (from paper 2301.13142)
Instead of starting small and hoping it's enough:
- Start with a LARGE student (base_ch=64 or 128, ~2-4M params)
- Train to 99.5%+ accuracy (should be easy with enough capacity)
- Apply self-compression: add differentiable bit-depth penalty to the loss
- Channels that aren't needed for this specific video get pruned to 0 bits automatically
- The paper achieved 3% of bits remaining on CIFAR-10 with no accuracy loss

**Key advantage**: We separate the accuracy problem from the compression problem. First achieve the accuracy target, then compress. No need to hope a small architecture is sufficient.

**Concern**: Compression might remove capacity needed for the adversarial decode gradient landscape. Need to validate transfer after compression.

#### 4. Feature-Matching / Attention Transfer Distillation
Instead of only matching output logits:
- Extract teacher's intermediate feature maps at multiple resolutions
- Train student to match these features through learned projection layers
- This preserves the teacher's internal representation structure, which is what the adversarial decode backprops through
- Methods: FitNets (Romero et al., 2015), Attention Transfer (Zagoruyko & Komodakis, 2017), CRD (Tian et al., 2020)

#### 5. Boundary-Focused Training
The 2.4% error is concentrated at class boundaries. Specifically target these:
- Weight the loss by distance to the nearest class boundary (pixels at boundaries get 10-100x higher weight)
- Use a boundary-aware loss like Dice loss or focal loss in addition to KD
- Oversample boundary regions during training
- Add an auxiliary boundary detection head

#### 6. Depthwise-Separable + Inverted Residual Blocks (MobileNet-style)
Standard 3×3 convs are parameter-inefficient. MobileNet-style blocks get more receptive field per parameter:
- Depthwise separable conv: 3×3 depthwise + 1×1 pointwise = ~9x fewer params than standard 3×3
- Inverted residual blocks with expansion factor
- Could achieve equivalent effective capacity at 3-5x fewer parameters

#### 7. Neural Architecture Search (NAS) for This Specific Task
Use a simple NAS or hyperparameter sweep to find the optimal tiny architecture:
- Sweep: number of levels, channel widths per level, kernel sizes, skip connection types
- Objective: maximize argmax agreement on the 600 frames at a fixed parameter budget
- Could discover that an asymmetric architecture (wide early layers, narrow deep layers, or vice versa) works much better than uniform channel widths

#### 8. Knowledge Distillation with Adversarial Training Loop
Instead of standard augmentation, specifically train on the failure mode:
1. Train student with standard KD
2. Run adversarial decode using student → produce frames
3. Evaluate those frames with teacher → find pixels where they disagree
4. Create a hard-example dataset from those disagreement regions
5. Retrain student with extra weight on hard examples
6. Repeat until convergence

This iteratively closes the gap specifically where it matters for the adversarial decode pipeline.

## Hard Constraints

- **Budget**: Combined SegNet + PoseNet student must fit in ~200 KB - 1.5 MB after quantization/compression in the archive.zip
- **Accuracy**: SegNet student must achieve >99% argmax agreement with teacher (ideally 99.5%+). PoseNet student must achieve MSE < 0.05.
- **Differentiability**: Both students must support backpropagation for adversarial decode.
- **Inference speed**: Students run on T4 GPU during adversarial decode. Smaller models = faster per-iteration = more iterations in 30 min = better convergence.
- **Training time**: We have unlimited offline compute for training, but faster iteration is better for experimentation.

## Model Details

### Teacher SegNet
- Architecture: `segmentation_models_pytorch.Unet` with `tu-efficientnet_b2` encoder
- 9.6M params, 5 output classes, input 3×384×512
- Evaluation metric: pixel-wise argmax disagreement (discrete)
- Weight statistics: 98.4% conv weights, 77.4% within |w| < 0.1

### Teacher PoseNet
- Architecture: `timm.fastvit_t12` backbone → linear summarizer → Hydra multi-head
- 13.9M params, 6-dim pose output (first 6 of 12), input 12×192×256 (YUV6 of 2 frames)
- Evaluation metric: MSE of 6-dim output (continuous)
- Both frames resized to 384×512, then YUV6 conversion subsamples to 192×256

### Training Data
- 600 frame pairs from a single 1-minute dashcam video (videos/0.mkv)
- Pre-extracted teacher outputs: seg_logits (600, 5, 384, 512), pose_outputs (600, 6)
- Pre-extracted preprocessed inputs: seg_inputs (600, 3, 384, 512), pose_inputs (600, 12, 192, 256)
- Very narrow distribution: one drive, consistent lighting, highway/suburban driving
- Class distribution: dominated by road (class 3) and sky/background (class 0, class 4)

### Current Training Results (baseline to beat)
- MicroUNet base_ch=32 (422K params): 97.6% accuracy at epoch 30/150, plateauing
- Training speed: ~30s/epoch without augmentation, ~55s/epoch with augmentation
- Clear diminishing returns on the current approach

## What We Need From Research

1. **The single most effective technique** to push a small model from 97.6% to 99.5%+ on this specific task
2. **Whether "train big then self-compress" is the safest path** — if a 2M-param model can hit 99.5% easily, then the compression paper's 3% bits result would give us ~60 KB
3. **Concrete architecture recommendations** for memorization-style distillation on a fixed, narrow dataset
4. **Whether spatial coordinate conditioning (CoordConv or Fourier features) can close the boundary accuracy gap** with minimal parameter overhead
