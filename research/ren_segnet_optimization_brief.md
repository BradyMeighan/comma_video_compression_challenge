# REN for SegNet Optimization — Deep Research Brief

## The Problem

We have a video compression pipeline for the comma.ai compression challenge. After compression and decoding, our frames have ~1,100 mismatched pixels per frame when evaluated by SegNet (a semantic segmentation model). These mismatches account for 0.53 points (41%) of our total score of 1.23.

We want to train a tiny neural network (Residual Enhancement Network / REN) that runs at decode time and corrects the decoded frames to minimize SegNet disagreement. The REN weights are included in the archive and count toward file size, so the model must be extremely small (~20-25KB after int8 quantization + bz2 compression).

## Previous REN Attempt (failed for SegNet)

We trained a REN with this architecture:
```python
class REN(nn.Module):
    def __init__(self, features=32):
        super().__init__()
        self.down = nn.PixelUnshuffle(2)  # (B,3,H,W) -> (B,12,H/2,W/2)
        self.body = nn.Sequential(
            nn.Conv2d(12, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(features, 12, 3, padding=1),
        )
        self.up = nn.PixelShuffle(2)  # (B,12,H/2,W/2) -> (B,3,H,W)
    
    def forward(self, x):
        x_norm = x / 255.0
        residual = self.up(self.body(self.down(x_norm)))
        return (x_norm + residual).clamp(0, 1) * 255.0
```

**Result: The REN crushed PoseNet (0.077 → 0.001) but did NOT improve SegNet at all.** SegNet stayed at ~0.00565.

**Why it failed for SegNet:**
1. The training loss used `w_seg = 0.01` (auto-calibrated) — SegNet loss was essentially ignored
2. The SegNet KL divergence loss didn't directly target argmax flips
3. The REN learned to match PoseNet's pose vectors (continuous, easy to optimize via MSE) but couldn't learn the discrete boundary corrections SegNet needs

## What Makes SegNet Hard to Optimize

### The Metric
SegNet distortion = fraction of pixels where `argmax(logits_original) != argmax(logits_compressed)`. This is a **discrete, non-differentiable metric** — you either flip a pixel's class or you don't. There's no gradient signal from pixels that are "almost right."

### The Error Structure  
- 1,100 mismatched pixels per frame (out of 196,608 at model resolution 384×512)
- **85% of errors are at class boundaries** (road↔lane_marking, car↔road, road↔undrivable)
- **94% of errors have logit margins < 0.5** — the correct class is very close to winning
- Errors are scattered as small clusters (57% are single isolated pixels)
- Top class confusions: car→road (17.5%), road→lane_marking (16.5%), road→undrivable (15.5%)

### Why Standard Training Losses Don't Work Well
- **Cross-entropy loss**: Pushes ALL pixels toward their correct class, wasting capacity on the 99.4% of pixels that are already correct
- **KL divergence on logits**: Matches the full logit distribution, but doesn't specifically target the ~1,100 pixels where argmax disagrees
- **Focal loss**: Better at hard examples, but still spreads signal across all pixels
- The REN has only ~25K parameters to correct 1,100 errors scattered across 196,608 pixels per frame — it needs to be extremely targeted

### The Budget Constraint
- ~25K parameters = ~25KB raw
- int8 quantized + bz2 compressed = ~20KB in the archive
- Rate cost: `25 * 20000 / 37545489 = 0.0133` points
- To be worth it, the REN must reduce `100 * seg_dist` by MORE than 0.0133
- That means fixing at least **16 pixels per frame** on average (16 / 196608 * 100 = 0.008, times some safety margin)
- Fixing even 100 pixels per frame = 0.05 seg improvement, which is a clear win

## What We Need Researched

### 1. Loss Function Design
What is the best loss function to train a tiny CNN to minimize argmax disagreement on a fixed evaluation model?

Ideas to explore:
- **Boundary-focused cross-entropy**: Weight the loss by distance to class boundaries, so the REN focuses capacity on boundary pixels
- **Margin-based loss**: For each pixel, compute the gap between the top-1 and top-2 logits. Penalize pixels where the margin is small or negative (wrong class winning). Something like hinge loss on the margin.
- **Straight-through estimator for argmax**: Use a differentiable approximation of the argmax operation so we can directly optimize the disagreement metric
- **Hard example mining**: Each epoch, identify the mismatched pixels (where compressed prediction != original prediction), compute loss ONLY on those pixels. Ignore the 99.4% of correct pixels.
- **Distillation with temperature**: Use high-temperature softmax on the original frame's logits as the target, so the REN learns the full class distribution, not just the argmax
- **Adversarial/targeted attack framing**: Treat each mismatched pixel as an adversarial example to reverse — find the minimal perturbation to flip it back to the correct class

### 2. Training Strategy
- **Overfitting is desirable**: We have exactly 600 frames and will only ever evaluate on these exact frames. The REN should memorize the corrections, not generalize.
- **Should we train at model resolution (384×512) or full resolution (1164×874)?** The REN runs at full resolution in inference but SegNet evaluates at model resolution. Training at model resolution is faster but the correction must survive the bilinear resize at eval time.
- **Two-stage training**: First train with soft loss (KL divergence) for general alignment, then fine-tune with hard loss (boundary-focused cross-entropy) for argmax flips?
- **Curriculum**: Start with the easiest-to-flip pixels (smallest margins) and progressively include harder ones?
- **Quantization-aware training**: Since we int8-quantize the final weights, should we simulate quantization noise during training?

### 3. Architecture Considerations
- Current: PixelUnshuffle(2) → 4 Conv layers (12→32→32→32→12) → PixelShuffle(2) = 25,452 params
- Should we use a different architecture? E.g., wider but shallower? Deeper but narrower?
- Should the REN operate at full resolution or at model resolution with an upscale?
- Would attention mechanisms (even tiny ones) help focus on boundary pixels?
- Could we use depthwise-separable convolutions to get more layers in the same parameter budget?

### 4. Inference Integration
The REN runs at decode time (inflate.py). Currently the pipeline is:
```
Decode AV1 → Lanczos upscale to 1164×874 → unsharp mask → [even-frame sidecars] → write
```
The REN would go after upscale, replacing or supplementing the unsharp mask:
```
Decode AV1 → Lanczos upscale to 1164×874 → REN → [even-frame sidecars] → write
```

Key constraint: SegNet internally resizes the REN output from 1164×874 to 384×512 via bilinear interpolation. So the REN's corrections at full resolution must survive this downscale.

## Key Files

- `modules.py` — SegNet definition (smp.Unet with tu-efficientnet_b2, 5 classes). See `preprocess_input` (line 108-110) which takes only the last frame and resizes to 384×512. See `compute_distortion` (line 112-114) which does argmax disagreement.
- `frame_utils.py` — `camera_size=(1164, 874)`, `segnet_model_input_size=(512, 384)`
- `evaluate.py` — The evaluation loop that computes the score
- `submissions/evenframe_meta_v4_crf34/inflate.py` — Current decode pipeline
- `train_ren.py` — Previous REN training script (needs major loss function changes)

## Concrete Deliverable

We need a training recipe (loss function + training loop + hyperparameters) that produces a 25K-parameter REN which, when applied to our compressed video frames, reduces the SegNet argmax disagreement by at least 100 pixels per frame (ideally 300-500+), within a 20KB weight budget.

The training data is fixed: 600 odd frames (compressed) paired with their ground-truth SegNet predictions (from the original video). This is a pure memorization task on a fixed dataset, not a generalization task.
