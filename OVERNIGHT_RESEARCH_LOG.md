# Overnight Research Exploration — May 2 2026

This doc tracks the systematic exploration of every approach mentioned in:
- `autoresearch/compass_artifact_wf-78d8b629-54e2-4529-8bae-b48c18f6c954_text_markdown.md` (Compass deep research)
- `autoresearch/Sidecar Optimization for Video Compression.md` (alternate deep research)

**Starting state:** mask K=1 top600 + channel-only RGB 250_K5+250_K2 → score **0.3021** (delta -0.0297) at 8.6KB

**Target:** beat 0.30, ideally land 0.285-0.295 per Compass 24h plan.

**Approach:** test each method in priority order via `master_overnight.sh`. All experiments share a common baseline (built once by `explore_baseline_builder.py`) and only modify what they're testing.

---

## Pipeline status

`master_overnight.sh` (PID 6605, started 01:06 May 2) chains experiments:

1. **explore_baseline_builder.py** → builds + saves baseline patches/frames (~12 min)
2. **explore_e1_ans.py** — entropy: constriction ANS for 3 streams
3. **explore_e2_zstd.py** — entropy: zstd levels + dictionary
4. **explore_e3_zopfli.py** — entropy: Zopfli vs bz2 vs zlib
5. **explore_a4_dim_bits.py** — analyze delta distribution + bit-allocation feasibility
6. **explore_a5_diff_coords.py** — differential coord encoding (varint gaps)
7. **explore_a2_3chan_scoring.py** — eval all 3 channels per candidate position
8. **explore_a3_compound.py** — multi-channel patches at shared coord
9. **explore_a1_subpixel.py** — sub-pixel coords + bilinear splat
10. **explore_m1_codebook.py** — k-means cross-pair codebook (K=8/16/32)
11. **explore_m2_tiny_mlp.py** — INR-style tiny MLP (most ambitious)

Logs:
- Master log: `autoresearch/sidecar_results/master_overnight.log`
- Individual: `autoresearch/sidecar_results/explore_*.log`
- Results CSVs: `autoresearch/sidecar_results/{e1_ans,e2_zstd,e3_zopfli,a1-a5,m1,m2}_results.csv`

---

## Master priority list (priority × confidence)

| ID | Method | Source | Effort | Expected | Status |
|----|--------|--------|--------|----------|--------|
| **E1** | constriction ANS coder for 3 streams (positions, channels, deltas) | Compass Rank 1 | 4-6h | -0.003 to -0.005 | queued |
| E2 | zstd + trained dictionary | Compass | 1h | -0.001 to -0.002 | queued |
| E3 | Zopfli for ZIP DEFLATE | Sidecar Opt | 30min | -0.0005 | queued |
| A1 | Sub-pixel coords + bilinear splat | Compass Rank 2 | 2h | -0.001 to -0.002 | queued |
| **A2** | 3-channel-per-position scoring | Compass Rank 2 | 1h | -0.001 to -0.002 | queued |
| A3 | Compound channel-only patches (shared coord) | Compass O | 2h | -0.001 | queued |
| A4 | Pose-dim-aware bit allocation analysis | Compass N | 1h | -0.0005 | queued |
| A5 | Differential coord encoding | Compass K | 1h | -0.0005 | queued |
| **M1** | Cross-pair k-means codebook | Compass Rank 5 / Sidecar 5.4 | 4h | -0.002 to -0.004 | queued |
| M2 | Tiny delta MLP (INR) | Compass Rank 3 / Sidecar 5.3 | 6-10h | -0.005 to -0.010 (high risk) | queued |

**Phase 2 (research-level) — queued via `phase2_runner.sh`:**

| ID | Method | Source | Effort | Expected | Status |
|----|--------|--------|--------|----------|--------|
| **O1** | Gumbel-Softmax differentiable patch selection (top-256 pool, anneal τ) | Sidecar 5.1 | 4h | -0.001 to -0.003 | queued |
| **O2** | CMA-ES gradient-free mask K=2 on top 100 (top-30 candidate pool) | Sidecar 6.1 | 4h | -0.001 to -0.003 | queued |
| **O3** | Hessian eigenvector targeting (3 power-iter, 1-channel patches) | Sidecar 6.3 | 3h | -0.001 to -0.002 | queued |
| **O4** | ADMM-style 3 rounds of alternating mask + RGB | Sidecar 6.2 | 4h | -0.001 to -0.003 | queued |
| **X1** | Sub-pixel + 3-channel-best COMBO (Compass actual Rank 2) | Compass | 2h | -0.002 to -0.004 | continuous loop |
| **X2** | 2x2 mask block flips (verified greedy, 5 candidate classes) | novel | 2h | -0.001 to -0.003 | continuous loop |

The phase2_runner.sh enters a **continuous loop** after O1-O4: scans `autoresearch/explore_x*.py` and runs any new ones. So I can keep adding experiments and they'll pick up automatically.

---

## Results

### Baseline reference (locked in)
- mask K=1 top600 verified-greedy + channel-only RGB 250_K5+250_K2 = **0.3021** at 8.6KB

### Experiment results

| ID | Variant | sb_total | score | Δ vs baseline | notes |
|----|---------|----------|-------|---------------|-------|
| baseline | mask+ch_RGB (current best) | 8.7KB | 0.3022 | 0 | reference |
| **E1** | ANS 3-stream + bz2 | 10.1KB | 0.3032 | +0.0009 | 🚨 WORSE — alphabet/PMF overhead too big |
| **E2** | zstd_l22 | 8.8KB | 0.3023 | +0.0001 | matches bz2 |
| E2 | zstd_l22 + trained dict | 9.9KB | 0.3031 | +0.0008 | dict overhead |
| **E3** | Zopfli | 8.8KB | 0.3023 | +0.0001 | matches bz2 |
| **A4** | analysis only | — | — | — | delta entropy=7.63bits/sym → bz2 near-optimal |
| **A5** | diff-coded coords | 8.9KB | 0.3024 | +0.0001 | varint gaps slightly worse |
| **A2** | 3-chan scoring | 9.1KB | 0.3032 | +0.0010 | 🚨 WORSE — heuristic underperforms max-grad |
| **A3** | compound patches | 11.6KB | 0.3035 | +0.0013 | 🚨 WORSE — threshold too low, 1620/1733 became 3ch |
| **A1** | sub-pixel + bilinear splat | 10.6KB | 0.3072 | +0.0050 | 🚨 WORSE — 4-bit fractional quantization noise |
| **M1** K=8 N=10 | k-means codebook | 3.6KB | 0.3092 | +0.0069 | smaller bytes but pose tanks |
| **M1** K=16 N=10 | | 4.0KB | 0.3086 | +0.0064 | |
| **M1** K=16 N=20 | | 4.6KB | 0.3077 | +0.0054 | |
| **M1** K=32 N=10 | | 4.7KB | 0.3073 | +0.0051 | best M1 but still worse |
| **M2** N=10 h=16 | tiny MLP | 3.0KB | 0.3136 | +0.0113 | 🚨 MLP can't generalize from 600 pairs |
| **M2** N=20 h=16 | | 3.6KB | 0.3141 | +0.0118 | |
| **M2** N=30 h=24 | | 5.7KB | 0.3155 | +0.0132 | |
| **O1** | Gumbel-Softmax differentiable | 7.9KB | 0.3167 | +0.0144 | 🚨 fails — soft selection adds noise |
| **O3** | Hessian eigenvector | 8.7KB | 0.3036 | +0.0013 | nearly ties — power iter overlaps with grad |
| **O4** | ADMM joint mask+RGB (3 rounds) | 9.0KB | 0.3023 | +0.0001 | TIES baseline — confirms local optimum |
| **🎯 O2** | CMA-ES mask flips top100 K=2 | 9.2KB | **0.3020** | **-0.0003** | 🏆 NEW BEST! Found 146 mask flips greedy missed |
| **X1** | sub-pixel + 3-chan combo | TBD | TBD | TBD | running (continuous loop) |
| **X2** | 2x2 mask blocks | TBD | TBD | TBD | queued |

---

## Key findings so far (2026-05-02 ~02:50)

### 🚨 Major finding: Compass / Sidecar deep research predictions DID NOT TRANSFER to our setup.

**Phase 1 result: ALL 14 experiments WORSE than baseline.** The simple `mask K=1 + max-grad channel-only RGB` is genuinely well-tuned and at the local optimum.

#### Why entropy coding failed:
A4 analysis: actual delta entropy = **7.63 bits/symbol** (close to max 8 bits). Our deltas are NOT heavy-tailed Laplacian like Compass assumed — they're near-uniform. bz2 already does ~3.9 bytes/delta which is near-optimal for this distribution.

#### Why algorithmic improvements failed:
- **A1 sub-pixel**: 4-bit fractional quantization adds noise. The original integer-pixel positions actually align well after bilinear downsample to PoseNet input. Sub-pixel "precision" is fake precision.
- **A2 3-channel scoring**: the "pick best channel by |delta|" heuristic underperforms simple "max-grad channel". Adam fits a delta to whichever channel — channel choice is mostly determined by gradient direction.
- **A3 compound patches**: too many positions had all-3-channel deltas above threshold; sidecar ballooned. Stricter threshold might help.

#### Why architectural changes failed:
- **M1 codebook**: cross-pair templates capture 60-70% of pose error at half the bytes, BUT the per-pair fine-tuning that channel-only patches do is essential. Net pose worse.
- **M2 tiny MLP**: 600 pairs is too few to fit a meaningful conditional model. MLP overfits training, doesn't generalize.

### What this means

- **Score 0.3022 may be near the practical floor** for this approach (mask + per-pair RGB patches).
- The remaining hope is in **research-level (O1-O4)** where we change the optimization paradigm itself, not just the heuristics.
- If O1-O4 also fail, the algorithmic ceiling is confirmed and the only remaining lever is **a better base model** (the H100 retrain).

## How to read results in the morning

```bash
# Master timeline:
cat autoresearch/sidecar_results/master_overnight.log

# Best result hunt:
for f in autoresearch/sidecar_results/{e1,e2,e3,a1,a2,a3,a4,a5,m1,m2}_*.csv; do
  echo "=== $f ==="
  cat "$f"
done

# Look for negative delta_vs_baseline = improvement
```

---

## What to do with the results

1. **If E1 (ANS) gives -0.003 to -0.005**: this is locked-in. Use for final submission encoding.
2. **If A2 or A1 finds new RGB patches** with `score < 0.3021`: replace channel-only RGB encoder.
3. **If M1 codebook beats baseline** at smaller bytes: switch to template-based RGB sidecar.
4. **If M2 MLP works** (unlikely but high upside): replaces all per-pair patches with weights.

The most important question to answer in the morning: **did total score drop below 0.295?** If yes, we have a clear path. If no, the sidecar approach may be near its floor and we need to focus on the H100 base-model retrain instead.

---

## Files

- `autoresearch/explore_baseline_builder.py` — builds shared baseline
- `autoresearch/explore_e1_ans.py` — ANS entropy coder
- `autoresearch/explore_e2_zstd.py` — zstd
- `autoresearch/explore_e3_zopfli.py` — Zopfli
- `autoresearch/explore_a1_subpixel.py` — sub-pixel + bilinear splat
- `autoresearch/explore_a2_3chan_scoring.py` — 3-channel scoring
- `autoresearch/explore_a3_compound.py` — compound patches
- `autoresearch/explore_a4_dim_bits.py` — bit allocation analysis
- `autoresearch/explore_a5_diff_coords.py` — differential coord encoding
- `autoresearch/explore_m1_codebook.py` — cross-pair k-means codebook
- `autoresearch/explore_m2_tiny_mlp.py` — tiny delta MLP (INR)
- `autoresearch/master_overnight.sh` — orchestrator

Plus the H100 continue-training run is also in progress (separate process, not local). Both should produce results by morning.
