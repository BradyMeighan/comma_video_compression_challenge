# Deep Research Prompt: Winning the comma.ai Video Compression Challenge

## Context

I'm competing in the [comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge). The goal is to compress a 1-minute, 20fps, 1164x874 dashcam video (37.5 MB as H.265 in MKV) as small as possible while preserving **semantic content** and **temporal dynamics** as measured by two specific neural networks. The current best baseline scores 4.39. I want to beat it significantly. Deadline is May 3, 2026.

## Scoring Formula (lower is better)

```
score = 100 * segnet_distortion + 25 * compression_rate + sqrt(10 * posenet_distortion)
```

Where:
- **segnet_distortion**: Average fraction of pixels where a SegNet's argmax class predictions disagree between original and reconstructed frames. The SegNet is an EfficientNet-B2 U-Net with 5 output classes (a road scene segmentation model). It operates on only the **last frame** of each 2-frame sequence, resized to **512x384** via bilinear interpolation. Input is RGB float `(B, 3, 384, 512)`.
- **posenet_distortion**: MSE between PoseNet outputs on original vs. reconstructed frame pairs. The PoseNet is a FastViT-T12 backbone that takes **two consecutive frames** converted to YUV420 (6-channel per frame = 12 input channels), resized to **512x384**. It outputs a 12-dim vector; distortion is MSE on the **first 6 dimensions** only (likely translational/rotational pose). Input is preprocessed as: RGB -> resize to 512x384 -> YUV420 with 4:2:0 subsampling -> stack two frames -> 12 channels, normalized by (x - 127.5) / 63.75.
- **compression_rate**: `archive.zip size / original file size (37,545,489 bytes)`

### Score Component Weights Analysis

The weights are highly asymmetric:
- **SegNet has 100x multiplier** — even 1% pixel disagreement costs 1.0 points. This is the dominant term.
- **Rate has 25x multiplier** — going from 6% to 3% compression rate saves only 0.75 points.
- **PoseNet is under a square root** — `sqrt(10 * 0.38) = 1.95`, so halving PoseNet distortion from 0.38 to 0.19 saves only ~0.57 points.

**Implication: SegNet preservation is overwhelmingly the most important factor. Rate reduction is second. PoseNet is least sensitive to degradation.**

## Baseline Analysis

### baseline_fast (score: 4.39)
- Downscales to 45% resolution (524x394 approx) using Lanczos
- Encodes with libx265 ultrafast, CRF 30, all-intra (keyint=1, no B-frames)
- At inflate time: decodes and bicubic-upscales back to 1164x874
- Archive size: ~2.24 MB (6.0% compression rate)
- SegNet distortion: 0.00947 (contributes ~0.95 to score)
- PoseNet distortion: 0.380 (contributes ~1.95 to score)
- Rate contribution: 25 * 0.060 = ~1.50

### no_compress (score: 25.0)
- Simply copies the original video into the archive
- Zero distortion but rate = 1.0, so 25 * 1.0 = 25.0

## Constraints

- **Inflation time limit: 30 minutes** on either a T4 GPU (16GB VRAM, 26GB RAM) or CPU-only (4 cores, 16GB RAM)
- **External libraries are allowed** freely, but any large artifacts (neural networks, meshes, point clouds) used at inflate time **must be included in archive.zip** and count toward compressed size
- The provided SegNet (~38MB) and PoseNet (~56MB) models are available in the repo and do NOT need to be in the archive
- Output must be `.raw` files: flat binary uint8 RGB frames `(N, 874, 1164, 3)`
- You may use the evaluation models, original video, and any other assets for compression (offline, no time limit)

## Key Technical Details

### SegNet Architecture
- `segmentation_models_pytorch.Unet` with `tu-efficientnet_b2` encoder, 5 output classes, no pretrained encoder weights (custom trained)
- Input: last frame of each 2-frame pair, resized to 512x384, RGB float
- Distortion = fraction of pixels where `argmax(pred_original) != argmax(pred_reconstructed)` averaged over spatial dims
- This means: as long as the dominant class per pixel doesn't flip, there is **zero distortion**. We only need to preserve the ranking of class logits, not their magnitudes.

### PoseNet Architecture
- FastViT-T12 backbone (a fast mobile-oriented vision transformer)
- Takes YUV420 representation: 4 luma sub-pixels + U + V = 6 channels per frame, 12 channels for 2 frames
- Processes at 512x384 resolution
- Outputs 12-dim vector, only first 6 dims matter for distortion (MSE)
- Key: operates in YUV space, so chrominance is already subsampled 4:2:0 — color fidelity matters less than luma structure

### Color Space Pipeline
The evaluation converts RGB to YUV420 using BT.601 coefficients:
- Y = 0.299R + 0.587G + 0.114B (luma — high importance)
- U = (B - Y) / 1.772 + 128 (chroma blue — lower importance, 2x2 averaged)
- V = (R - Y) / 1.402 + 128 (chroma red — lower importance, 2x2 averaged)

This means the PoseNet is **much more sensitive to luminance than chrominance**, and chrominance is already spatially downsampled 2x.

## Research Questions

I need comprehensive research on the following topics. For each, provide specific techniques, relevant papers, implementation details, and how they map to this challenge's unique scoring function.

### 1. Neural Network-Aware Video Compression

Since distortion is measured by specific neural networks (not PSNR/SSIM), what techniques exist for:
- **Perceptual/task-aware compression**: Methods that optimize compressed representations to preserve neural network feature spaces rather than pixel-level fidelity. How can I compress in a way that specifically preserves SegNet class boundaries and PoseNet motion features?
- **Adversarial/feature-matching approaches**: Can I find transformations of frames that look very different pixel-wise but produce identical or near-identical outputs from the SegNet and PoseNet? This would let me "simplify" frames dramatically before encoding.
- **Network distillation for compression**: Using the SegNet/PoseNet as teachers to guide what information to retain during compression.
- **Segmentation-map-based encoding**: Since SegNet only cares about class labels per pixel, could I encode just the segmentation map + a minimal texture that reconstructs frames the PoseNet still processes correctly? What's the theoretical minimum information needed?

### 2. Extreme Low-Bitrate Video Coding Techniques

The baseline achieves 6% rate. What can push much lower?
- **State-of-the-art neural video codecs** (2023-2026): DCVC-FM, DCVC-DC, SSF, CANF-VC, ELF-VC, NVTC, and newer approaches. Which achieve the best rate-distortion at very low bitrates? Which can decode within 30 minutes on a T4?
- **Generative/synthesis-based compression**: Approaches like HiFiC, CDC, GFVC (generative face video compression) adapted for driving scenes. Using GANs or diffusion models at decode time to hallucinate plausible details from very compact representations.
- **Keyframe + motion-only coding**: Encoding one or a few keyframes well, then encoding only optical flow / motion vectors for remaining frames. How compact can motion representations be?
- **Implicit neural representations (INR/NeRF-style)**: Encoding the video as neural network weights (SIREN, instant-NGP, FFN). What are realistic decode times and compression rates for 1-minute video?
- **VQ-VAE / discrete latent approaches**: Vector-quantized variational autoencoders for video. Could a compact codebook + index sequence beat traditional codecs at this bitrate?
- **AV1/VVC/H.266 vs. H.265**: What gains do newer codecs provide at extreme low bitrates for dashcam content specifically? What are the best ffmpeg/encoder parameter settings for this scenario?

### 3. Resolution and Spatial Optimization

The baseline downscales to 45%. Both models evaluate at 512x384.
- **Optimal encode resolution**: Since both networks resize to 512x384, what is the minimum resolution we can encode at without degrading the networks' outputs after bilinear/bicubic upscale? Is encoding directly at 512x384 (or even lower) optimal?
- **Content-adaptive resolution**: Encoding different spatial regions at different resolutions (e.g., road surface at lower res, lane markings and vehicles at higher res).
- **Super-resolution at decode time**: Using lightweight neural super-resolution (Real-ESRGAN, SwinIR, or custom models) during inflation to upscale from very low resolution. Given 30 minutes on a T4, what resolution could we upscale from?
- **Frequency-domain approaches**: Encoding in DCT/wavelet domain with aggressive quantization of high frequencies that don't affect the evaluation networks.

### 4. Temporal Optimization Strategies

PoseNet takes 2 consecutive frames. The video is 20fps, ~1200 frames.
- **Frame dropping/interpolation**: Can we encode fewer frames and interpolate at decode time? The PoseNet needs consecutive frames to match — what does "consecutive" mean in evaluation (every pair? overlapping?)? The dataloader iterates with `seq_len=2`, creating non-overlapping pairs.
- **Temporal prediction models**: Using video prediction/interpolation networks (RIFE, AMT, FILM) at decode time to reconstruct frames from sparse keyframes.
- **Delta encoding strategies**: First frame + residuals, or periodic keyframes with learned interpolation.
- **GOP structure optimization**: Finding the optimal keyframe interval (the baseline uses all-intra which is terrible for rate; using predictive frames should help enormously).

### 5. Exploiting the Evaluation Pipeline

Understanding and exploiting the specific evaluation setup:
- **Pre-computing optimal "target" frames**: Running the SegNet and PoseNet on the original video, extracting the class maps and pose vectors, then at decode time generating frames that reproduce those exact outputs — even if the frames look nothing like the originals. Is this feasible? What generative approach would work?
- **Adversarial frame generation**: Finding minimal perturbations or simplified frame representations that produce identical network outputs. Feature inversion techniques.
- **Quantization robustness**: The evaluation uses uint8 RGB. Understanding the networks' sensitivity to quantization in different regions of the input space.
- **Evaluation batch structure**: The dataloader yields non-overlapping 2-frame sequences. Frame 0-1, 2-3, 4-5, etc. This means odd-numbered frames only need to be correct in PoseNet context with their even-numbered partner and for SegNet (which uses only the last frame of each pair, so frames 1, 3, 5...).

### 6. Practical Implementation Strategy

Given the constraints, what's a realistic winning pipeline?
- **Encoder-side (no time limit, any compute)**: What should the offline compression pipeline look like? Using GPUs, multiple models, optimization loops, etc.
- **Decoder-side (30 min on T4 or CPU)**: What can realistically be decoded in this time? How many frames/sec can a neural decoder or super-resolution model process at 512x384 or 1164x874?
- **Archive format optimization**: Beyond video codecs, what are the best ways to compress the final bitstream? Arithmetic coding, custom entropy coding, etc.
- **Model weight compression**: If including neural network weights in the archive (e.g., for a decoder network), techniques to minimize their size: pruning, quantization (int4/int8), knowledge distillation, architecture search for tiny models.

### 7. Dashcam-Specific Priors

Driving video has strong structural priors:
- **Static scene geometry**: The road, sky, and scene structure are highly predictable. Perspective vanishing points, road plane geometry.
- **Ego-motion model**: The camera is on a moving vehicle. Motion is dominated by ego-motion (forward translation + rotation), which is low-dimensional.
- **Semantic regions**: Sky (very compressible), road surface (low texture), other vehicles (important for segmentation), lane markings (critical thin features).
- **Comma2k19 dataset statistics**: What are the typical scene statistics of this dataset? Lighting conditions, road types, traffic density?

### 8. Hybrid Approaches

What combinations of techniques might yield the best score?
- Example: Very low-res H.265 encode + neural super-resolution + SegNet-aware post-processing
- Example: Keyframe coding with generative interpolation + pose-vector residual correction
- Example: Segmentation map encoding + conditional generation + YUV-space optimization
- Example: Feature-space coding (encode the intermediate features of SegNet/PoseNet rather than pixels)

## Deliverable

For each research area, I need:
1. **Specific named techniques/papers** (with years) that are most relevant
2. **Quantitative benchmarks** where available (rate-distortion numbers, decode speeds)
3. **Feasibility assessment** given constraints (30-min decode on T4, archive size matters)
4. **Concrete implementation recommendations** prioritized by expected score impact
5. **A proposed top-3 most promising approaches** ranked by likely score improvement over the 4.39 baseline, with estimated score breakdowns (segnet_dist, posenet_dist, rate)
