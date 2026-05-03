# Research state — comma video compression challenge

**Deadline:** May 3, 2026 (today). One night of work left.
**Current best:** **0.2921** (per `v2_unified_results.csv`).
**Stretch goal:** 0.2899 (rounds down to 0.28). Closer goal: stay safely below 0.30.

This doc briefs another model on where we are, what's been tried, and what remains.

## Score formula

`score = 100 × seg_dist + sqrt(10 × pose_dist) + 25 × rate`
where `rate = (MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes) / UNCOMPRESSED_SIZE`.
The eval rounds DOWN so 0.2999 → 0.29.

## Pipeline that produces 0.2921

Stack on `autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt` (e80 model):

1. **Cache builder** (`autoresearch/v2_cache_builder.py`):
   - X2: 2x2 mask block flips on top 600 pairs (verified greedy, batched)
   - CMA-ES K=2 single-pixel mask flips on top 100 pairs (batched population eval)
   - Channel-only RGB on top 250 K=5 + 250-500 K=2 (cached for reference)
2. **Per-pair selector** (`autoresearch/v2_per_pair_select.py`) — for each of top 600 pairs, evaluates 16 subsets of `{x2, cmaes, pattern, pose}` and picks lowest-loss.
3. **S2 patterns** (variable-shape mask flips: 1x1/3x3/1x4 strip/4x1 strip/2x2) on top 100 — collected by per-pair selector.
4. **C3 pose deltas** with **wider range (-5..5)** on dims (1, 2, 5), top 200 — gradient-free 7³=343 grid search.
5. **RGB tier** (winning config): top 100 K=3 + top 400 K=1 — pose-targeted on f1.
6. **LZMA2 wrap** of the combined raw stream (replaces per-stream bz2).

**Final byte breakdown (0.2921 winner):**
- mask block patches (x2/cmaes/pattern after per-pair selection): ~3000B
- pose deltas (top 200, ±5 range): ~400B
- RGB on f1 (top 100 K=3 + 400 K=1): 3581B
- Combined LZMA2 stream: **6260B**

## Score progression

| Stage | Score | Δ |
|---|---|---|
| Raw e80 (no sidecar) | 0.2986 | — |
| + per-pair sel (mask + pose) bz2 streams | 0.2953 | −0.0033 |
| + LZMA2 wrap | 0.2945 | −0.0008 |
| + wider C3 + slim RGB tier | **0.2921** | **−0.0024** |

(For reference: old `gen_continued.pt` model with same pipeline = 0.2999. e80 is the better base.)

## The model

`autoresearch/colab_run/3090_run/gen_3090.pt.e80.ckpt` — 92K param FP4-quantized generator.
- Full U-Net trunk: stem→s1→down→d1→up→fuse→f1
- Head1: dual FiLMRes (no merge, no pre) → out  
- Head2: r1 (Res) + pre + out
- pose_mlp: 3-layer Linear(6,64,64,64) FP16
- model_bytes ~54078

**Critical:** `prepare.apply_fp4_to_model(gen)` MUST be called after `load_state_dict` on EMA-saved ckpts (we hit a 3.66 score bug on the first try without this — fixed in `v2_shared.State.__init__`).

## Eval observation about modules.py

- **PoseNet sees BOTH frames** (preprocesses to 6-channel YUV from f1+f2)
- **SegNet ONLY sees f2** (`x = x[:, -1, ...]` in modules.py:108)
- All current RGB patches are on **f1** → they NEVER affect SegNet's input
- Tested seg-targeted f2 patches (cross-entropy via SegNet gradient) → REGRESSED (cost too many bytes; seg_dist already at noise floor of 0.000291)

## Per-dim pose error (e80, computed via sidecar_analyze)

| Dim | RMS | Used in C3? |
|---|---|---|
| 0 | 0.0306 | ❌ |
| 1 | 0.0351 | ✓ |
| 2 | 0.0289 | ✓ |
| 3 | 0.0111 | ❌ (small) |
| 4 | 0.0080 | ❌ (smallest) |
| 5 | 0.0264 | ✓ |

Tested swapping (1,2,5) → (0,1,2) and 4-dim (0,1,2,5): **both worse** despite dim 0 having higher output error. Reason: model is less responsive to dim 0 input changes than to dim 5.

## Tournament v1 results (10 RGB hypotheses on top 100 pairs)

Winner: **K=2 single-pixel pose-gradient on f1**. f1 vs f2 same. K=2 beats K=1, K=5, K=8 on byte/benefit ratio.

## Tournament v2 results (14 hypotheses)

Top 3 tied at NET=−0.0024:
1. **C3d wider deltas (1,2,5) ±5** at 403B — most byte-efficient
2. K=1×2 iterative RGB at 1177B
3. K=1×3 iterative RGB at 1523B

**Multi-channel patches all regressed.** Combined seg+pose grad ≈ pose alone (because seg-grad on f1 is zero).

## Failed approaches (don't try again)

- **Seg patches via SegNet on f2** — bytes don't pay back; seg already at noise floor
- **Multi-channel RGB patches** (2 or 3 channels at same pixel) — regressed
- **Combined seg+pose loss for f1 patch positions** — same as pose-only (f1 doesn't affect seg)
- **Pose dims (0,1,2)** — model less responsive than (1,2,5)
- **3x3 mask blocks on X2-rejected pairs** — rejected pairs are easy, no juice
- **Per-stream bz2** — LZMA2 on combined raw saves ~25%
- **K=8 patches** in tournament — bytes don't pay back
- **K=1×3 vs K=1×2 iterative** — 3 passes don't beat 2 (diminishing returns)

## Untried but promising

- Finer RGB tier sweeps near `top 100 K=3 + 400 K=1` (e.g., top 80 K=4 + 350 K=1, top 150 K=3 + 350 K=1)
- C3 even wider range (-7, ±10) — see if it keeps improving past ±5
- Drop CMA-ES single-pixel patches — only 37 pairs picked it, may not pay 400B
- Mask method ablations (drop X2/pattern/CMA-ES individually)
- K=3 iterative on top 100 (vs K=3 single)

## Key files

- `SIDECAR_PIPELINE.md` — historical pipeline doc (slightly out of date — 0.2999 era)
- `autoresearch/v2_shared.py` — shared State, batched optimizers, score helpers
- `autoresearch/v2_cache_builder.py` — builds X5 cache against current model
- `autoresearch/v2_per_pair_select.py` — per-pair method selection
- `autoresearch/v2_unified_pipeline.py` — produces the 0.2921 winner
- `autoresearch/v2_tournament_v2.py` — last hypothesis sweep
- `autoresearch/sidecar_analyze.py` — per-pair / per-dim error analysis
- `autoresearch/sidecar_results/v2_unified_results.csv` — all unified pipeline configs

## Compute

- 3090 (24GB VRAM). bs=8 hits OOM at activations; bs=4 is safe (~10GB).
- bs=4 typical training takes ~50s/epoch.
- Cache builder takes ~15 min on 3090.
- Full per-pair pipeline takes ~25 min.
- Hypothesis tournament on 100 pairs takes ~10-15 min.
- C3 grid-search per pair is the slow phase (343 candidates × ~5s/pair × 200 pairs = 16 min). PoseNet/Generator backprop slow due to FP4.

## Submission format

We need a decoder script that takes the sidecars + model and reproduces frames. NOT YET BUILT — that's next step after we squeeze final wins. Format is per-stream sidecars or a single LZMA2 archive.

## Hard constraints

- Don't modify or overwrite `autoresearch/colab_run/gen_continued.pt` (original model, not used anymore but kept).
- `gen_3090.pt.KNOWN_GOOD_0.2999_BACKUP` — backup of original model.
- `v2_cache/` — current cache built against e80. Backup directories `v2_cache_OLD_MODEL_BACKUP/` and `v2_cache_e80_BROKEN_NO_FP4/` exist — DON'T USE either.

## Current blocking issue

None. Pipeline produces 0.2921 cleanly. Just need to mine remaining bytes/dim/tier optimizations to squeeze further.
