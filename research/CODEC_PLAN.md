# Codec Compression Plan — SVT-AV1 + Evaluation-Aware Optimization

## Goal
Beat the current leader (1.95) using a codec-based approach. No models in the archive — just a compressed video file + inflate script that decodes and upscales.

## Why This Works
The current leader already proves this approach at 896KB / 1.95 score:
- seg_dist ≈ 0.00509 → 0.51 points
- pose_dist ≈ 0.07084 → 0.84 points
- rate ≈ 0.024 (896KB / 37.5MB) → 0.60 points

They use: SVT-AV1 codec + Lanczos downscale to 45% + ROI-aware preprocessing + unsharp mask post-upscale.

We can beat this by using the teacher models OFFLINE to optimize every parameter.

## The Pipeline

### Step 1: Offline Sensitivity Analysis
Use SegNet and PoseNet gradients to compute per-pixel importance maps:

```python
# For each frame, compute: which pixels cause the most seg/pose distortion when perturbed?
frame.requires_grad_(True)
seg_out = segnet(preprocess(frame))
seg_loss = cross_entropy(seg_out, seg_out.argmax(1))  # self-referential
seg_loss.backward()
seg_sensitivity = frame.grad.abs().mean(dim=1)  # per-pixel importance for SegNet

# Same for PoseNet
pose_sensitivity = ...  # per-pixel importance for PoseNet

# Combined importance map (weighted by score formula)
importance = 100 * seg_sensitivity + 3.16 * pose_sensitivity
```

This tells us WHERE bits matter most. Boundary pixels (where SegNet class changes) and motion-informative pixels (where PoseNet is sensitive) get more bits.

### Step 2: ROI Map Generation for SVT-AV1
SVT-AV1 supports ROI (Region of Interest) maps — per-block QP offsets that allocate more bits to important regions.

```bash
# SVT-AV1 ROI format: per-64x64 block QP delta
# Lower QP = more bits = higher quality
# Generate ROI map from sensitivity analysis
python generate_roi_maps.py --input videos/0.mkv --output roi_maps/
```

Key regions to prioritize:
- **Class boundaries** in segmentation (where argmax can flip)
- **Road markings, lane lines** (SegNet edge features)
- **Moving objects** (PoseNet motion cues)
- **Horizon line** (both models sensitive here)

Lower priority (can be aggressively compressed):
- **Sky interior** (uniform, stable class)
- **Road surface interior** (uniform, stable class)
- **Frame edges** (often cropped/irrelevant)

### Step 3: Preprocessing Before Encoding
The leader uses several preprocessing tricks:

1. **Downscale**: Lanczos to ~45% resolution (524×394 from 1164×874). Experiment with:
   - Different scale factors (40%, 45%, 50%, 55%)
   - Different resampling filters (Lanczos, spline, bicubic)
   - The PoseNet is reportedly sensitive to resampling filter choice

2. **ROI-aware denoising**: Blur/denoise regions where SegNet/PoseNet don't care (interior of large uniform regions). This saves bits without hurting scores.
   - Apply bilateral filter or guided filter to low-importance regions
   - Keep edges sharp in high-importance regions

3. **Color optimization**: The evaluation models process in specific color spaces (SegNet: RGB, PoseNet: YUV420). Optimize color quantization for these models specifically.

4. **Film grain synthesis removal**: If the source video has grain, removing it before encoding saves significant bits.

### Step 4: SVT-AV1 Encoding
```bash
# Install SVT-AV1
# Key parameters to sweep:
SvtAv1EncApp -i input.y4m -b output.ivf \
  --preset 0 \          # slowest = best compression
  --crf <sweep 20-40> \ # quality vs size tradeoff
  --film-grain 0 \      # no film grain synthesis
  --roi-map roi.txt \   # per-block QP offsets
  --keyint 1 \          # all-intra for random access (or experiment with inter)
  --tile-rows 0 --tile-columns 0
```

Also test:
- H.265/HEVC via x265 (the baseline uses this)
- AV1 via libaom (slower but sometimes better than SVT-AV1)
- VVC/H.266 via vvenc (newest codec, best compression)

### Step 5: Post-Decode Enhancement
After decoding and upscaling, apply targeted post-processing:

1. **Unsharp mask / edge restoration**: Sharpens class boundaries that got blurred by codec + downscale. Critical for SegNet argmax accuracy.
   ```python
   # Binomial unsharp mask (what the leader uses)
   blurred = gaussian_blur(frame, sigma=1.0)
   sharpened = frame + alpha * (frame - blurred)  # alpha ~0.3-0.5
   ```

2. **Class-boundary refinement**: Use the stored segmentation targets to guide edge restoration:
   - At pixels near class boundaries, push colors toward the ideal class color
   - This prevents argmax flips at boundaries without a neural network

3. **Upscale filter optimization**: Test different upscale methods:
   - Lanczos (sharp but can ring)
   - Bicubic (smooth)
   - Spline (reportedly better for PoseNet)
   - NNEDI3 / waifu2x (neural upscale — but model would need to be in archive)

### Step 6: Grid Search
Sweep all parameters and evaluate with SegNet + PoseNet:

```python
for scale in [0.40, 0.42, 0.45, 0.48, 0.50]:
    for crf in [20, 22, 25, 28, 30, 32, 35]:
        for upscale in ['lanczos', 'bicubic', 'spline']:
            for sharpen_alpha in [0, 0.2, 0.3, 0.5]:
                # Encode → decode → upscale → sharpen → evaluate
                seg_dist, pose_dist, size = evaluate(params)
                score = 100*seg_dist + sqrt(10*pose_dist) + 25*(size/37545489)
```

## Expected Score Components

Based on the leader's approach and our available teacher models for optimization:

| Component | Leader (1.95) | Our Target |
|---|---|---|
| 100 × seg_dist | 0.51 | 0.30-0.45 (ROI should help boundaries) |
| sqrt(10 × pose_dist) | 0.84 | 0.70-0.84 (harder to improve) |
| 25 × rate | 0.60 | 0.50-0.60 (similar codec, similar size) |
| **Total** | **1.95** | **1.50-1.89** |

The biggest opportunity: reducing seg_dist via better ROI allocation and boundary-aware post-processing. The leader's 0.51 seg component suggests their boundaries aren't perfectly optimized.

## Implementation Priority

1. **Baseline**: Reproduce the leader's approach (Lanczos 45% + SVT-AV1 + unsharp)
2. **ROI maps**: Generate from SegNet gradient sensitivity, feed to SVT-AV1
3. **Boundary post-processing**: Use stored seg targets to refine class edges after decode
4. **Parameter sweep**: CRF, scale, upscale filter, sharpen strength
5. **Advanced**: per-frame adaptive ROI, PoseNet-aware temporal encoding

## Files Needed
- `videos/0.mkv` — source video
- `models/segnet.safetensors` — for offline sensitivity analysis
- `models/posenet.safetensors` — for offline sensitivity analysis
- `distill_data/seg_logits.pt` — pre-extracted teacher outputs for boundary detection

## Dependencies
- SVT-AV1 encoder (`SvtAv1EncApp`)
- ffmpeg (for format conversion, scaling, muxing)
- Python: torch, numpy, scipy (for sensitivity maps and post-processing)

## Key Insight
The teacher models are used OFFLINE for analysis only — they never appear in the archive. The archive contains just:
- Compressed video file (~800KB)
- inflate.py script (decodes + upscales + optional post-processing)
- Optionally: ROI maps or boundary masks if they help post-processing (~50-100KB)
