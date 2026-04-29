# autoresearch — comma.ai Video Compression Challenge

Automated research loop. You are an autonomous ML researcher. Your goal: find the lowest possible proxy score by modifying `train.py`.

## Setup

1. **Agree on a run tag** with the user (e.g. `apr29`). Branch `autoresearch/<tag>` must not exist yet.
2. **Create branch**: `git checkout -b autoresearch/<tag>`
3. **Read these files for full context**:
   - `program.md` (this file) — your research strategy and instructions
   - `prepare.py` — fixed infrastructure (DO NOT MODIFY): data loading, evaluation, helpers
   - `train.py` — the ONLY file you edit
4. **Run baseline**: `python train.py > run.log 2>&1` — this builds the data cache on first run (~30s extra). Subsequent runs start instantly.
5. **Create results.tsv** with header: `commit\tscore\tseg_term\tpose_term\trate_term\tmodel_bytes\tn_params\tstatus\tdescription`
6. **Log baseline** and begin the loop.

## The Challenge

Compress driving video so two frozen neural networks (SegNet + PoseNet) produce identical outputs on reconstructions vs originals, while keeping file size small.

**Score** (lower = better):
```
score = 100 × avg_segnet_dist + √(10 × avg_posenet_dist) + 25 × rate
```

**The approach**: Train a tiny neural generator: segmentation mask + 6D pose → (frame1, frame2). Model is FP4-quantized. Total archive = mask video (~220KB) + poses (~13KB) + compressed model weights.

**Current leaderboard**: #1 Quantizr 0.33, #2 fp4_mask_gen 0.36, #3 selfcomp 0.38. Our goal: beat 0.33.

## What You Can Edit

Only `train.py`. Everything in it is fair game:
- Architecture (any PyTorch nn.Module)
- Hyperparameters (LR, batch size, loss weights, etc.)
- Training loop (stages, scheduling, curriculum)
- Loss functions (anything differentiable through SegNet/PoseNet)
- Quantization strategy (when to enable QAT, block sizes)

## What You Cannot Do

- Modify `prepare.py`
- Install new packages
- Change the evaluation function

## Output Format

`train.py` prints a parseable summary:
```
---
score: 0.4500
seg_term: 0.3200
pose_term: 0.0800
rate_term: 0.0500
model_bytes: 58000
total_bytes: 290782
n_params: 87836
```

Extract: `grep "^score:\|^seg_term:\|^pose_term:\|^rate_term:\|^model_bytes:\|^n_params:" run.log`

## The Experiment Loop

LOOP FOREVER:

1. Check git state and `results.tsv` to understand progress
2. **Think**: which score component dominates? What worked before? What hasn't been tried?
3. Edit `train.py` with ONE focused change
4. `git add train.py && git commit -m "description"`
5. `python train.py > run.log 2>&1`
6. Extract results: `grep "^score:\|^seg_term:\|^pose_term:\|^rate_term:\|^model_bytes:\|^n_params:" run.log`
7. If empty → crash. Run `tail -n 50 run.log`, attempt fix. If fundamentally broken, skip.
8. Append to `results.tsv` (tab-separated, do NOT commit this file)
9. **If score improved** (lower): keep commit, advance branch
10. **If score worse or equal**: `git reset --hard HEAD~1`
11. GOTO 1

**NEVER STOP.** The human is asleep. Run until manually killed. ~10 experiments/hour, ~100 overnight.

**Be scientific**: one variable per experiment. Only combine when merging two individually-proven wins.

## Score Breakdown — What Matters

| Term | Formula | Typical range | What drives it |
|------|---------|---------------|----------------|
| seg_term | 100 × disagree_rate | 0.15–0.40 | Frame2 must match SegNet argmax classes pixel-by-pixel |
| pose_term | √(10 × pose_mse) | 0.05–0.15 | Frame1+Frame2 pair must produce same PoseNet 6D vector |
| rate_term | 25 × bytes/37.5M | 0.04–0.08 | Model weight size after FP4+brotli |

**seg_term dominates.** Focus 70% of effort here. Even 0.001 reduction in seg_dist = 0.1 score improvement.

## Technical Details You Must Know

- **SegNet**: Unet + efficientnet_b2, 5 classes, uses ONLY last frame (frame2), metric = argmax disagreement
- **PoseNet**: fastvit_t12, 12-ch YUV420 input (both frames), outputs 6D pose, metric = MSE on first 6 dims
- **Resolution**: model works at 384×512, upscaled to 874×1164 for eval
- **FP4**: 4-bit blockwise quantization (block_size=32), 16-bit scales. ~78% brotli compression on quantized weights
- **Rate cost**: every extra KB of model ≈ +0.0007 rate_term. A 10KB model size reduction = ~0.007 score improvement

## Proxy vs Full Training — Why This Scales

The proxy uses 100 pairs (from 600), split 80 train / 20 val. **Evaluation is on the held-out 20 val pairs that the model never trains on.** This is critical:

- We measure **generalization**, not memorization. A trick that overfits 80 pairs will score poorly on the 20 val pairs.
- This mirrors Karpathy's autoresearch design where val_bpb is measured on a separate validation set.
- Improvements that help on held-out val will transfer to the full 600-pair run because they reflect genuine architectural/loss/optimization advantages, not data memorization.

Absolute proxy scores are higher (worse) than full training. What matters is **relative ranking** — if A beats B on proxy val, A will almost certainly beat B in full training.

## ═══════════════════════════════════════════════════
## RESEARCH DIRECTIONS — From Safe to Exotic
## ═══════════════════════════════════════════════════

### Tier 1: Likely Impactful (try first)

1. **Error boost schedule**: Currently 9→49. Try adaptive per-class boosting — weight misclassified classes proportionally to their frequency of error. Or use focal loss instead of boosted CE.

2. **Width/depth sweep**: Current C1=56, C2=64. Try narrower (C1=48) with more res blocks, or wider (C1=64) with fewer. The tradeoff is seg_term improvement vs rate_term cost.

3. **Time allocation**: 55/27/13% may not be optimal. Maybe 70/15/10% (more anchor time) or 40/30/25% (more joint time) works better.

4. **Learning rate**: Try 1e-3 with cosine decay, or warmup+decay. The current flat LR may leave performance on the table.

5. **Dual FiLM**: Add a second FiLM conditioning block in Head1. Quantizr uses this. Tiny param cost, may help pose_term significantly.

### Tier 2: Novel and Promising

6. **Logit-matching loss instead of CE**: Instead of cross-entropy on argmax, directly minimize MSE or cosine distance on SegNet's raw 5-class logit vector. This is softer than hard class matching and gives gradients even when the class is correct but confidence differs.

7. **Boundary-weighted loss**: Compute class boundaries (where adjacent pixels have different GT classes) and weight the loss 5-10× higher there. Boundaries are where most segnet disagreements happen.

8. **Gradient-guided error maps**: Run one eval pass first, identify the exact pixels where SegNet disagrees, then do a second training pass with extreme weighting on those pixels. Like iterative refinement within the training loop.

9. **Asymmetric quantization**: Don't quantize the final output conv (already done) but ALSO keep the class embedding in higher precision. These are the "information bottleneck" layers.

10. **Shared head with frame-index conditioning**: Instead of separate Head1/Head2, use ONE shared head conditioned on a binary "frame index" embedding. This halves the head parameters → lower rate_term, and forces the trunk to learn frame-agnostic features.

### Tier 3: Exotic / Out-of-the-box

11. **Soft mask input**: Instead of hard class indices → embedding, use the SegNet logits (softmax probabilities) as input. This preserves class boundary uncertainty information that hard argmax throws away. The mask compression would need to store probabilities not classes, but the information gain at boundaries could be huge.

12. **SegNet feature distillation**: Instead of just matching the argmax class, train a small adapter that matches the intermediate features of SegNet. If we can match features at layer X, the argmax will automatically match. This attacks the problem at a deeper level.

13. **Pose-conditioned trunk**: FiLM the trunk itself with pose, not just the head. This means the shared features are already pose-aware, which could improve both heads simultaneously. Currently the trunk is pose-blind.

14. **Coordinate frequency bias**: Replace linear coords with fixed sinusoidal frequencies tuned to the typical class boundary scale. If class regions are ~50px wide, encode coords at that frequency. This gives the network a prior on spatial structure.

15. **Reverse distillation from SegNet encoder**: SegNet's efficientnet_b2 encoder extracts powerful features from the GT frame. What if we train our generator to produce frames that, when fed through SegNet's encoder, produce similar intermediate feature maps? This is a more fine-grained signal than just matching the final logits.

16. **Multi-resolution mask trick**: Train the generator on masks at both 384×512 AND a 192×256 downsampled version as auxiliary input. The lower-res mask gives global context, the full-res gives detail. Tiny parameter overhead for a second embedding path.

17. **Stochastic depth during training**: Randomly skip residual blocks during training. This acts as regularization and may improve the model's robustness to FP4 quantization noise (the weights learn to be more redundant/robust).

18. **QAT-aware initialization**: Initialize weights using a distribution that's already close to the FP4 codebook levels [0, 0.5, 1, 1.5, 2, 3, 4, 6]. This reduces the optimization burden of QAT — the weights start closer to their quantized values.

19. **Dynamic error boosting**: Instead of a fixed boost schedule, maintain a running EMA of per-pixel error and use that as the loss weight. Pixels that are consistently wrong get exponentially more weight. Like online hard example mining but smoother.

20. **Frame2-only mode with pose bypass**: For the pose metric, what if we generate ONLY frame2 and reconstruct frame1 as a trivially-derived version (e.g., frame2 + learned offset)? The pose network sees both frames through YUV420 — maybe a simple color shift is enough to match pose without a full separate head.

### Things That Will NOT Work (avoid)
- GAN/adversarial losses (unstable with FP4)
- Very large models (rate term kills you)
- Complex attention (too many params, too slow for 5-min budget)
- Perceptual/VGG losses (not aligned with the actual SegNet/PoseNet metrics)
- Trying to modify prepare.py (read-only, will break things)

## Tips

- Always check which term improved/degraded. A change that helps seg_term but hurts rate_term might still be net positive — do the math.
- When stuck, go back to basics: read the score breakdown, think about what the SegNet actually looks at (class boundaries!), and what PoseNet needs (consistent ego-motion cues).
- If you find a big win, don't immediately move on. Try to squeeze more from the same direction (e.g., if wider model helped, try slightly wider again).
- Track GPU memory. If you're using <20GB on the 3090, there's room to be more aggressive with model size or batch size.
- Remember: we're optimizing for the leaderboard, not for pretty images. A frame that looks terrible to humans but fools SegNet+PoseNet is perfect.
