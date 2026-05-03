# Deep Research Prompt: Sidecar Optimization (UPDATED with overnight findings)

## Background

I'm competing in the [comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge). Score formula (lower is better, scores ROUND DOWN at submission so 0.299 → 0.29):

```
score = 100 × seg_dist + sqrt(10 × pose_dist) + 25 × (archive_bytes / 37,545,489)
```

We trained a **tiny 92K-parameter FP4-quantized generator** that takes (mask, pose) as input and outputs predicted frames. Mask 219,588 bytes, pose 13,194 bytes, model 54,078 bytes (all fixed, in archive).

**Current best score: 0.3010** (delta -0.0308 from model-alone baseline of 0.3318).

We are **0.0010 score above** the next "rounded down" tier of 0.29. The deadline is May 3, 2026.

---

## Current state of the art (UPDATED 2026-05-02)

**Best config: X5 = mask 2x2 block flips top 600 + CMA-ES K=2 single-pixel mask flips top 100 + channel-only RGB 250_K5+250_K2**
- Mask blocks (X2): 2.10KB (462/600 pairs accepted, verified-greedy 2x2 block flips)
- Mask CMA-ES extras (O2): 0.70KB (additional pixel flips on top 100 hardest pairs)
- Channel-only RGB: 6.86KB (250 hardest × K=5 + next 250 × K=2)
- Total sidecar: 9.66KB
- Final score: **0.3010** (seg=0.029, pose=0.075, rate=0.197)

Comparison ladder:
- Model alone (no sidecar): 0.3318
- Original best (mask K=1 + ch RGB): 0.3022
- X2 (2x2 mask blocks alone) + ch RGB: 0.3013
- **X5 (X2 + CMA-ES on top) + ch RGB: 0.3010** ← current best

---

## What we tested from the previous deep research — full audit

### Compass deep research recommendations

#### Rank 1: Custom Range/ANS coder for sidecar streams
**Predicted impact: -0.003 to -0.005**
**Actual: WORSE by +0.0009 (16% larger sidecar)**

Tested with `constriction` library. Three streams: positions (empirical 2D CDF), channels (3-symbol arithmetic), deltas (Categorical over int8). Result: ANS+bz2 was 16% LARGER than bz2 alone because alphabet metadata + PMF storage overhead exceeded the entropy savings.

**Why it failed:** A4 analysis revealed actual delta entropy = **7.63 bits/symbol** (close to max 8 bits). The deltas are NOT heavy-tailed Laplacian like Compass assumed — they're nearly uniform. bz2 already gets ~3.9 bytes/delta which is essentially optimal for a near-uniform distribution. Compass's prediction assumed exp-Golomb (k=2) → 4.5 bits/delta but our data has too much entropy for that.

Also tested zstd l3-l22, zstd_l22 + trained dictionary, Zopfli, varint differential coordinates — ALL slightly worse than bz2. Bz2 is genuinely near-optimal for this data.

**Implication for future research:** entropy coding is NOT a path forward. The Compass prediction was based on wrong distributional assumptions.

#### Rank 2: Sub-pixel coordinates + 3-channel-per-position scoring
**Predicted impact: -0.002 to -0.004**
**Actual: WORSE by +0.005 (sub-pixel) and +0.001 (3-channel) and +0.010 (combined)**

- A1 sub-pixel + bilinear splat (4-bit fractional offsets each axis): score went from 0.3022 → 0.3072 (+0.005)
- A2 3-channel-per-position scoring (eval all 3 channels per top-K position): 0.3022 → 0.3032 (+0.001)
- X1 combined sub-pixel + 3-channel-best: 0.3022 → 0.3119 (+0.010)

**Why it failed:** Quantizing sub-pixel positions to 4-bit fractions adds noise that exceeds the precision gain. The integer-pixel positions actually align well after bilinear downsample to PoseNet's 384×512 input. The "max-grad channel" heuristic outperforms "best channel by |delta|" — the gradient direction already chose the right channel.

#### Rank 3: Tiny conditioned delta MLP (INR-style)
**Predicted impact: -0.005 to -0.010 (high risk)**
**Actual: WORSE by +0.011 to +0.013**

Tested 3 configs (N=10 patches/pair × hidden=16; N=20×16; N=30×24). MLP weights stored as INT8 (1-5KB). MLP couldn't learn meaningful per-pair patch generation.

**Why it failed:** 600 pairs is too few to fit a meaningful conditional model. The MLP overfits training pose vectors but doesn't generalize to inference. Loss curves were chaotic throughout 500 training iterations.

#### Rank 4: Verified mask flips with analytic fast-forward + Gumbel-softmax
**Not implemented** (engineering complexity too high in available time). Discussed below.

#### Rank 5: Cross-pair codebook with sparse offsets
**Predicted impact: -0.002 to -0.004**
**Actual: WORSE by +0.005 to +0.007 (best: K=32 N=10 was -0.005 worse)**

K-means clustering of pose vectors into K={8,16,32} clusters, optimized one shared "base correction template" of N positions per cluster. Per-pair sidecar = cluster id only.

**Why it failed:** Templates capture 60-70% of within-cluster pose error at HALF the bytes (4.7KB vs 8.7KB), but the per-pair fine-tuning that channel-only patches do is essential. Without per-pair offsets, pose error increases enough to negate the byte savings.

Honorable mention (O) **Compound channel-only patches (multi-channel per shared coord)**: also failed (+0.001) because threshold tuning was wrong (1620/1733 patches became 3-channel) — sidecar ballooned.

### Sidecar Optimization deep research recommendations

The "Sidecar Optimization for Video Compression" doc made similar predictions structured as 5 architectural paradigms. Mapped to our results:

#### 5.1 End-to-end Differentiable Patch Selection via Gumbel-Softmax
**Predicted "5-15% improvement vs greedy"**
**Actual: WORSE by +0.014**

O1 implemented: Gumbel-softmax over a top-256 candidate pool, anneal τ from 2.0 → 0.05, joint optimize positions + deltas + channels. Snapped to hard top-K at end. Result: catastrophically worse.

**Why it failed:** Soft selection during optimization adds noise that doesn't anneal away cleanly. The discrete-then-continuous greedy actually converges to a better local optimum than continuous-then-discrete. Cordonnier 2021 et al. predict this should work for image patches; doesn't transfer to our pose-loss-driven setting.

#### 5.2 Analytic ERF-guided fast mask verification
**Not implemented** (engineering complexity). Would have enabled K=2-3 mask verification in tractable time. Note: O2 CMA-ES achieved similar effect (using gradient-free search to find better K=2 mask flips than greedy alone) and got -0.0003.

#### 5.3 Implicit Neural Representation (Tiny Delta Network)
**Same as Compass Rank 3 — failed (+0.011)**. 600-sample data limit.

#### 5.4 Wyner-Ziv Cross-Pair Shared Dictionaries
**Same as Compass Rank 5 — failed (+0.005)**. K-means clustering doesn't beat per-pair patches.

#### 5.5 Sub-Pixel Bilinear Splatting
**Same as Compass Rank 2 — failed (+0.005)**. 4-bit fractional quantization adds noise.

### Other research-level approaches tested

#### Hessian eigenvector targeting (O3)
**Predicted: more efficient sparse perturbations on ViT (Patch-Fool literature)**
**Actual: WORSE by +0.0013**

Power iteration (3 steps) on PoseNet Hessian w.r.t. f1, picked top-K positions where eigenvector magnitude is largest. Result: nearly tied baseline but slightly worse. Power iteration introduces noise; first-order gradient already captures dominant direction.

#### ADMM joint mask+RGB optimization (O4)
**Predicted: escape local optima of alternating greedy**
**Actual: TIED baseline (+0.0001)**

3 rounds of alternating mask + RGB optimization with consensus. Result: essentially tied baseline. This actually CONFIRMS that the alternating greedy IS at a local optimum — ADMM exploration around it doesn't find escape.

---

## What ACTUALLY worked (the things that succeeded)

### O2: CMA-ES for mask flips (top 100 K=2)
**Score: 0.3020 (delta -0.0003)** — first non-failure of the night

Gradient-free CMA-ES over the top-30 candidate position pool (initial pool from gradient). Population 12, generations 20. For each candidate, exhaustively try all 5 classes, verify via full forward.

**Why it worked:** verified-greedy K=1 leaves slack because the gradient direction is only ~25% reliable for mask flips. CMA-ES samples populations of (position, class) combinations and finds combinations that work jointly — overcomes the gradient-direction unreliability.

### X2: 2x2 block mask flips (verified greedy)
**Score: 0.3013 (delta -0.0009) — ~3× better than O2**

Per pair: top-N candidate (x,y) positions (gradient-pooled over 2x2), verify each by trying all 5 classes for the WHOLE 2x2 block, accept the (position, class) with biggest actual loss reduction. Storage: 5B per block (same as 1-pixel mask flip).

**Why it worked:** A 2x2 mask block has ~4× the receptive-field impact of a single pixel for the SAME byte cost. The mask gets resized to the generator's 384×512 input via embedding, where the conv receptive field then propagates a single pixel's effect over ~7×7 output region. With a 2x2 input change, the receptive footprint roughly doubles.

### X5: X2 + CMA-ES (PERFECT additive stack)
**Score: 0.3010 (delta -0.0012) ← CURRENT BEST**

X2 finds 2x2 block flips on top 600 (462 pairs accepted, 2.1KB).
Then CMA-ES (O2) finds additional 1-pixel flips on top 100 hardest pairs of THAT state (146 flips, 0.7KB).
Then channel-only RGB re-found on the doubly-mask-modified frames (6.86KB).

**Critical finding:** X2 gain (-0.0009) + O2 gain (-0.0003) = -0.0012 perfectly. The two methods find DIFFERENT, NON-OVERLAPPING mask improvements. This means there's likely MORE slack — additional optimization techniques applied on top might compose similarly.

We tried X6 (X2 + CMA-ES top 200): -0.0011 (slightly worse than X5). Diminishing returns on extending CMA-ES coverage past top 100.

---

## Updated key insights

1. **Score 0.3010 is genuinely close to the local optimum** for this approach (mask + per-pair RGB patches). 22 different experiments tried, only 3 beat baseline (X2, O2, X5). All others were worse.

2. **Mask sidecar improvements are the only path that worked.** Every single RGB patch modification (sub-pixel, 3-channel scoring, compound, codebook, MLP, Gumbel) failed. The simple max-grad channel-only RGB is genuinely well-tuned.

3. **bz2 is near-optimal for our delta distribution** (entropy 7.63 bits/symbol). Custom entropy coding can't help.

4. **Stacking compatible mask methods works perfectly additively.** X2 + O2 = X5 (-0.0009 + -0.0003 = -0.0012). Suggests there might be MORE non-overlapping mask methods to combine.

5. **The differentiable / continuous-relaxation / gradient-free / second-order methods all FAILED.** ADMM tied baseline. Gumbel was catastrophic. Hessian was slightly worse. CMA-ES (O2) was the lone success — but only as a SUPPLEMENT to greedy, not a replacement.

6. **600 pairs is too few for any conditional learning.** MLP failed badly. Codebook failed. Anything that tries to "amortize" across pairs fails because per-pair patches encode genuinely per-pair info.

7. **Patch saturation is real.** K=7 per pair for RGB is the sweet spot. K=8+ doesn't help. Mask K=1 verified-greedy + extra K=2 from CMA-ES is the sweet spot.

8. **The entire sidecar is at the byte/pose Pareto frontier.** Adding bytes (more patches/flips) hurts more than it helps via rate term. Reducing bytes (codebook, MLP) loses more pose than it saves bytes.

---

## What's left untried — directions for the NEXT deep research

We're at **0.3010** and need **0.299** to round down to 0.29 (the next tier). That's only **0.002 to find** but every algorithmic and entropy-coding approach we've tested has failed. We need genuinely NEW ideas.

### Untested paradigms

#### A. 3x3 mask block flips (extension of X2)
2x2 worked. What about 3x3? More receptive impact per byte. We didn't test this. Worth maybe -0.0005 to -0.001 if the receptive field scales similarly.

But: 3x3 mask change might be TOO big — could shift class boundaries enough to flip multiple SegNet output regions, hurting seg term.

#### B. Mask-flip CMA-ES on the BLOCK pattern (not single pixels)
O2 did CMA-ES on single-pixel positions over a top-N pool. We never combined CMA-ES with X2's block flips (O2-on-X2 patterns). Could find better block placements than greedy.

#### C. Fast-forward mask-delta computation (Sidecar 5.2 / Compass Rank 4)
For each candidate mask flip, only recompute the OUTPUT region affected by the receptive field (~30-50 pixel patch around the flip). This was 100× speedup according to Compass, would enable K=3-5 verified mask flips per pair.

We didn't implement it because of engineering complexity. But given X2 + O2 both improve mask, more mask flips might help further.

#### D. Mask sidecar designed for DIM 0 only (yaw)
Per-dim analysis: dim 0 (yaw) was 57.8% of pose error originally and the RGB patches drove it down 78%. After RGB patches, dims 1, 2, 5 dominate. Maybe specifically targeting dims 1, 2, 5 in a SEPARATE sidecar (mask flips chosen for their effect on those dims) could help.

#### E. Adversarial sample selection (which 600 pairs to "boost" most)
We rank pairs by pose error magnitude. But maybe the marginal-impact ranking is different. Some pairs might be at the "edge" where small fix → big improvement.

#### F. PoseNet activation analysis
For each pair, identify which intermediate PoseNet activation is most "wrong" (deviation from ground truth). Target sidecar at fixing that specific activation.

#### G. Adversarial prefix codes for the whole archive
The submission archive is a ZIP. ZIP DEFLATE's compression depends on the order of files. Ordering files to maximize cross-file redundancy might save 100-500 bytes (~0.0003-0.001).

#### H. Submission ZIP exploits
The archive must be a valid ZIP. Are there ZIP-specific compression tricks (e.g., DEFLATE block-size tuning, dictionary injection) that could squeeze more?

#### I. Train a neural network to OUTPUT the sidecar bytes
End-to-end: a network takes (pose, mask, pre-existing_sidecar_state) → outputs the optimal next byte to write. This is research-level but could find globally better encodings than per-pair greedy.

#### J. Direct pose-vector modification at decode time (skip the gen forward)
Currently sidecar modifies generator OUTPUTS. What if it modified the pose VECTOR fed to the generator? A 6-byte delta per pair (i8 × 6) = 3.6KB total for 600 pairs. We tested a version of this earlier (sidecar_pose_input.py) but didn't fully evaluate.

### Low-cost untested ideas

- 3x3 block mask flips
- 4x4 block mask flips  
- CMA-ES extending X2 (gradient-free over BLOCK patterns)
- Variable block sizes per pair (1x1 for fine pairs, 3x3 for coarse pairs)
- Mask flip + RGB delta JOINT optimization per pair (instead of sequential)
- Different mask CLASS choices per block: 2x2 with all 5 classes tried per pixel within block

### Computational hardware question

If we had an unlimited GPU budget and 24h, what compute-intensive method might find -0.002? Things like:
- Per-pair gradient-free search with very large populations (100k samples per pair)
- Exhaustive enumeration of all top-N positions × 5 classes (~30 × 5 = 150 evals per K-iteration per pair, fully verified)
- Massive sub-pixel grid search (16×16 = 256 sub-positions per coordinate)

---

## Specific deep research questions for the NEXT round

1. **Why did all algorithmic Compass predictions fail?** The literature is consistent (ViT sparse adversarial robustness, learned image compression with side info, etc.) but our specific setup violated some assumption. Which one?

2. **What's the theoretical lower bound on sidecar size given pose_dist=0.000558?** Compass computed ~3-4KB info-theoretic floor. We're at 9.66KB. Where's the 5KB of "waste"?

3. **Are there ZIP / DEFLATE-specific compression tricks** that could shave 100-500 bytes off the final archive?

4. **Mask sidecar specifically — what other novel attack surfaces exist?**
   - We tested 1-pixel verified-greedy (mask K=1)
   - We tested 2x2 block verified-greedy (X2 — winner)
   - We tested CMA-ES gradient-free 1-pixel (O2 — second winner, stacks with X2)
   - What's left? 3x3? 1xN strip? Diagonal? L-shape?

5. **Patch composability beyond X2+O2:** since X2 + O2 stacked perfectly, maybe other PAIRS of methods would also stack. Which ones haven't been tried in combination?

6. **Generator architecture exploits:** the generator is FP4-quantized 92K parameters. Are there architecture-specific tricks (e.g., exploiting the embedding lookup table, the FiLM conditioning, the upsampling) that could be a sidecar?

7. **Submission rounding exploit:** scores round down at submission, so 0.299x → 0.29. We're at 0.3010 — only 0.002 from a full tier improvement. What's the minimum-effort path to break 0.30 (below 0.30 → rounds to 0.29)?

8. **Multi-mask sidecars:** currently mask flips are stored as discrete (x, y, class). What about mask EMBEDDING modifications? Modifying the embedding-table values directly? This bypasses the discrete class constraint.

9. **Channel-only RGB + auxiliary tiny patches:** our channel-only is 6-byte per patch. Could a TINY 2-bit patch (e.g., just sign + small magnitude) at additional positions add value at very low byte cost?

10. **Dimensional analysis:** pose_term dominates (0.075 vs seg 0.029 vs rate 0.197). To get to 0.299, we need to drop ~0.002 from pose+rate. Pose is at 0.075 — to drop pose_term by 0.002 we need pose_dist from 0.000558 to 0.000511 (~8% MSE reduction). At marginal byte cost ~0.0007/KB, that's roughly ~3KB worth of additional patches if perfectly efficient.

---

## What we want from the next deep research

1. **Top 3 untested ideas that could realistically save -0.002**, ranked by likely impact.
2. **Any literature on pose-loss-specific patch optimization** that we missed (PoseNet/FastViT-T12 attack surfaces).
3. **ZIP/DEFLATE specific archive optimization** techniques.
4. **The minimum-effort "round to 0.29" path** if it exists.

Goal: get below **0.30 (rounds to 0.29)**. We're at 0.3010 — 0.0011 too high.

---

## Reference files

- `autoresearch/sidecar_combined_lean.py` — original 0.3022 baseline
- `autoresearch/explore_x2_mask_blocks.py` — X2 winner (2x2 block flips)
- `autoresearch/explore_o2_cmaes_mask.py` — O2 winner (CMA-ES extension)
- `autoresearch/explore_x5_x2_plus_cmaes.py` — X5 best combo
- `autoresearch/explore_*.py` — every other failed experiment with detailed code
- `autoresearch/sidecar_results/*.csv` — full results tables
- `OVERNIGHT_RESEARCH_LOG.md` — overnight tracking doc
- `autoresearch/colab_run/gen_continued.pt` — the base model (score 0.3318 alone)
