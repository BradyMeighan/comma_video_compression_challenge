# Sidecar pipeline — how we got from 0.3318 → 0.2999

Documenting the full chain of techniques applied on top of `autoresearch/colab_run/gen_continued.pt` to reach **score 0.2999** (rounds down to 0.29).

Score formula: `100 × seg_dist + sqrt(10 × pose_dist) + 25 × rate`
where `rate = (MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes) / UNCOMPRESSED_SIZE`.

## Score progression

| Stage | Score | Δ vs raw model | Notes |
|---|---|---|---|
| Raw model output (no sidecar) | 0.3318 | — | Starting point |
| v1 baseline sidecar (K=1 mask + RGB tiered) | 0.3022 | −0.0296 | mask=1866B + rgb=6804B = 8670B |
| X5 (X2 blocks + CMA-ES K=2 single-pixel + RGB) | 0.3010 | −0.0308 | mask blocks 2802B + rgb 6855B |
| **v2 per-pair selection + LZMA2 wrap** | **0.2999** | **−0.0319** | All methods, picked per-pair, single LZMA2 stream 9756B |

## The model

- `autoresearch/colab_run/gen_continued.pt` — 54078 bytes (FP4-quantized 92K-param generator)
- Architecture (`autoresearch/train.py` Generator):
  - Trunk: full U-Net (stem→s1→down→d1→up→fuse→f1)
  - Head1: dual FiLMRes (no merge, no pre) → out
  - Head2: r1 (Res) + pre + out
  - pose_mlp: 3-layer Linear(6,64,64,64) FP16
  - All Conv2d charged at FP4 (4-bit) per `prepare.estimate_model_bytes`
  - Linear charged at FP16 except where wrapped in 1×1 QConv2d (FP4)
- Inputs: 5-class mask (224×400), pose vector (6 dims)
- Outputs: f1, f2 frames upsampled to OUT_H × OUT_W (384×512)

## The sidecar layers (in application order)

Each layer operates on the *output state* of the previous layer. The decoder applies them in the same order to reconstruct.

### 1. X2 — 2x2 mask block flips (top 600 hardest pairs)

Modify the input mask before the generator. A 2×2 block flip changes the gen's whole feature map via the conv receptive field — much more leverage than a single-pixel flip per byte.

- **Format:** `u16 x, u16 y, u8 class` = 5 bytes per patch
- **Method:** `verified_greedy_block_mask_batched` (in `v2_shared.py`)
  - Compute pose loss gradient w.r.t. one-hot mask
  - Pool gradient over 2×2 blocks, take top-N candidate positions
  - Batch all (n_candidates × 5_classes) candidates in a SINGLE gen forward (5–10x speedup vs serial)
  - Accept the candidate that lowers actual pose loss
- **Result:** 514 of top 600 pairs accepted a 2×2 block

### 2. CMA-ES K=2 single-pixel mask flips (top 100 hardest)

For the hardest pairs only — find K=2 single-pixel mask flips via gradient-free CMA-ES.

- **Format:** `u16 x, u16 y, u8 class` = 5 bytes per flip
- **Method:** `cma_es_mask_for_pair_batched` (in `v2_shared.py`)
  - From the pose-loss gradient, restrict candidates to top 30 positions
  - CMA-ES over which K=2 of those 30 to flip + which class
  - pop=12, gens=15 → 180 evals/pair, batched in single gen forward per generation (~10x speedup)
- **Result:** 73 of top 100 found beneficial flip pairs

### 3. S2 patterns — variable-shape mask flips (top 100)

Replaces (or supplements) single-pixel CMA-ES with multi-shape patterns. Tries 1×1, 3×3, 1×4 horizontal strip, 4×1 vertical strip, 2×2.

- **Format:** `u16 x, u16 y, u8 pattern_id, u8 class` = 6 bytes per patch (pattern_id and class each fit in 4 bits → could be 5 bytes packed)
- **Method:** `cma_es_pattern_for_pair` (in `v2_s2_strip_cmaes.py`)
  - Genome: K × 4 floats = `[x, y, pattern_id, class]` for K=2 patches per pair
  - Discrete fields decoded via floor(continuous weight)
  - pop=12, gens=18 → 216 evals/pair, batched per generation
- **Pattern usage** (from S2 standalone): mostly 1×4 and 4×1 strips and 3×3 — strips capture horizontal/vertical edges efficiently

### 4. C3 — pose-vector input deltas (top 200)

Modify the input pose vector before the generator. Per-dim quantized to int8.

- **Format:** `u16 pair_id, i8 d1, i8 d2, i8 d5` = 5 bytes per pair
- **Targeted dims:** 1, 2, 5 (dominant residuals after RGB)
- **Per-dim scale:** `[0.001, 0.005, 0.005, 0.001, 0.001, 0.005]`
- **Method:** `find_pose_deltas_gridsearch` (in `v2_c3_pose_vector.py`)
  - Gradient-FREE grid search over 7 values per target dim → 7³ = 343 candidates per pair
  - Evaluated in chunks of 32 (avoids INT32 overflow in CUDA kernels with 343-batch feats)
  - Picks the delta tuple that minimizes pose loss for that pair
  - **Why grid-free:** Adam-via-autograd through the FP4-quantized gen hung indefinitely (45 min no progress); grid search avoids autograd entirely
- **Result:** 199 of top 200 pairs found an improvement

### 5. Channel-only RGB patches (top 250 K=5 + next 250 K=2)

Per-pixel single-channel RGB modifications applied to f1 AFTER gen forward.

- **Format:** `u16 x, u16 y, u8 channel_id (0-2), i8 delta` = 6 bytes per patch
- **Method:** `find_channel_only_patches` (in `sidecar_channel_only.py`)
  - Gradient through PoseNet w.r.t. f1 → pick top-K pixels by max-channel `|grad|`
  - For each pixel, optimize an int8 delta on the chosen channel via Adam (lr=2.0, n_iter=80)
  - Batches 8 pairs per forward
- **Tier allocation:** 250 hardest get K=5 patches each, next 250 get K=2 each (diminishing returns past tier 2)

### 6. Per-pair method selection (the 0.2999 unlock)

For each of the top 600 pairs, evaluate every subset of `{X2, CMA-ES, pattern, pose}` (up to 16 combos) and pick the subset that minimizes pose loss.

- **Method:** `per_pair_select` (in `v2_per_pair_select.py`)
  - Builds (mask_oh, pose) tensor for each viable combo
  - Batches all combos for a pair into one gen forward
  - 600 pairs × ≤16 combos in ~70 seconds total
- **No metadata cost:** each method has its own bz2/lzma stream; a pair just appears in the streams it picked
- **Distribution of picks:** x2=448, cmaes=47, pattern=76, pose=149, **none=110**
  - 110 of top 600 pairs were better with NO patches than any of the available candidates — those original baseline patches were actually hurting

### 7. LZMA2 outer wrap (replaces per-stream bz2)

After per-pair selection, instead of compressing each method's patch dict with bz2 separately, concatenate the raw bytes (with section headers) and compress as a single LZMA2 stream.

- **Format:** Magic + length-prefixed sections for {x2, cmaes, pattern, pose, rgb}
- **Compressor:** `lzma.compress(raw, format=FORMAT_XZ, preset=6)`
- **Effect:** raw 19160B → 9756B (vs 10954B per-stream bz2). Saves 1198B (~−0.0007 score).

## Files

- `autoresearch/v2_shared.py` — batched optimizers (X2, CMA-ES) and shared State class
- `autoresearch/v2_cache_builder.py` — runs X2 + CMA-ES + RGB once, caches to `v2_cache/`
- `autoresearch/v2_c1_lzma2.py` — initial LZMA2 wrap experiment
- `autoresearch/v2_c2_3x3_residual.py` — 3×3 blocks on X2-rejected pairs (regression — not used)
- `autoresearch/v2_c3_pose_vector.py` — pose-vector grid-search deltas
- `autoresearch/v2_s2_strip_cmaes.py` — variable-pattern CMA-ES
- `autoresearch/v2_per_pair_select.py` — **the final pipeline that produces 0.2999**

## Sidecar byte breakdown (final)

| Method | Stream bytes (bz2) | Pairs treated |
|---|---|---|
| x2 (2×2 blocks) | 2064 | 448 |
| cmaes (single-pixel) | 481 | 47 |
| pattern (variable shape) | 921 | 76 |
| pose-delta | 565 | 149 |
| rgb (channel-only) | 6923 | ~500 |
| **per-stream bz2 total** | **10954** | |
| **LZMA2 combined wrap** | **9756** | |

## What didn't work

- **3×3 blocks on X2-rejected pairs** (C2) — pairs X2 rejected are the easiest; no juice for bigger blocks. Net +0.0001 (regression).
- **Adam optimization of pose deltas via autograd through FP4 gen** — hung indefinitely. Replaced with gradient-free grid search.
- **Larger CMA-ES populations** — diminishing returns; pop=12 gens=15 captures most of the value.

## What's left untried

- Extending mask methods to pairs 600–1199 (probably small wins, mostly "none" picks)
- Pose deltas on ALL 1200 pairs (cheap method, might help)
- 5-bit packed pattern_id + class (saves 1B/pattern × 76 pairs × 2 patches = 152B ≈ −0.0001)
- Continued joint+pose training of the generator itself (planned next: 4h on 3090)
