# Research Status — autoresearch/apr29

Last updated after ~130 experiments. Score: baseline `2.142` → best single `1.415` (median ~1.5). Gap to leaderboard #1 (Quantizr 0.33) is still 4×. The remaining gap is unlikely to close via more incremental tuning of the current architecture — the proxy budget is exhausted on this design class.

## Why grinding stopped working

The base architecture was cloned from Quantizr (same `C1=56, C2=64, DM=1, GroupNorm(2), SiLU, FiLM in Head1, mask emb + coords stem`). Their score is 0.33 because they spend ~1080 epochs over 600 pairs (~228× our compute). We have 5 min / ~50 epochs / 80 train pairs. Every architectural addition that costs activation memory steals epochs from the trunk and crashes seg. The pose-conditioning surface (where every win in this branch landed) appears saturated for this scale.

A fresh agent should not iterate on the same design. The breakthrough has to come from **exploiting the eval pipeline** — `modules.py`, `prepare.py`, the FP4 codebook, the mask/pose byte budgets — not from another FiLM variant.

## Hard-won wins (kept commits, in order)

| exp | change | score | category |
|---|---|---|---|
| 1, 2 | FT_LR 5e-5 → 5e-4 | 1.64 | hyperparam |
| 4 | JT_LR 1e-5 → 5e-5 | 1.61 | hyperparam |
| 11 | pose-finetune loss MSE → SmoothL1 β=0.1 | 1.57 | algorithmic ✓ |
| 17 | pose_mlp 2-layer → 3-layer | 1.40 | architectural ✓ |
| 28 | QAT_FRAC 0.5 → 0.7 | 1.38 | schedule |
| 37 | Head1 add concat-pose merge layer | 1.46 | architectural ✓ |
| 43 | GRAD_CLIP 1.0 → 0.5 | 1.40 | hyperparam |
| 74 | EMA on joint stage (decay=0.9) | 1.46 | algorithmic ✓ (decay value is budget-tuned) |
| 81 | 3-step ERR_BOOST schedule (9→49→98 in last 5%) | 1.39 | schedule |
| 112 | COND_DIM 48 → 64 | 1.52 | architectural ✓ |
| 118 | trunk-output FiLM modulating h1 features only (h2 sees raw trunk) | **1.42** | architectural ✓ |

**Karpathy-flagged confidence:** the ✓ rows (algorithmic + architectural) should transfer to the 19h full run. The unmarked rows (hyperparam values, schedule timings, EMA decay) are budget-tuned and may evaporate at scale — keep them on the proxy but don't trust them at full budget.

## Karpathy autoresearch principle (why budget tuning is dangerous)

Karpathy's nanoGPT/speedrun rule: **search algorithms, not hyperparameters**.
- Architecture, loss formulation, optimizer choice → tend to transfer across scales.
- Specific LR, weight decay, grad clip, EMA decay, schedule timings → usually need re-tuning at full scale; "wins" on a short proxy often evaporate.
- His agent only modifies algorithmic code; hyperparameters use known-good defaults from literature/scaling laws.
- Express schedules as fractions of total steps, never absolute epoch counts.
- When results are close, run multiple seeds and average — a single proxy run has ~±0.2 noise here.

For this branch, that means the `1.42` best-single is partly real (algorithmic wins should transfer) and partly luck (hyperparam wins are ~50/50). If the full-run result is significantly worse than predicted, the suspect changes to ablate first are: GRAD_CLIP=0.5, FT_LR=5e-4, JT_LR=5e-5, EMA decay=0.9, ERR_BOOST_HI=49, the 3-step boost schedule, QAT_FRAC=0.7.

## What was tried and **clearly didn't work** (don't re-run)

- Time reallocation between stages (T_ANCHOR/T_FT/T_JT) — anchor needs 55%
- Frame1 seg supervision (faded) in anchor — extra segnet forward halved anchor epochs
- EMA at decay=0.99 across all stages — 50 epochs is too short for it to forget initial random h1
- Cosine LR decay per stage to 0.1× — too aggressive, h1 starves
- Optimizer reset at QAT transition — within noise
- Color bias parameter (raw, pose-conditioned, on h1 output) — pose worse
- Pose→affine warp on h1 features — pose much worse
- Flow-warp h1 = grid_sample(h2_output) with pose-derived flow — slow + artifacts
- Residual h1 = h2 + delta — pose 2.4×, frame2-shaped output can't satisfy posenet
- Init h1 from h2 weights — same problem as residual
- Fourier feature encoding of pose — pose values too coarse to benefit
- Transformer-style pose_mlp (LN + 2 res MLP blocks) — worse
- Pose-conditioned channel attention (softmax) — worse
- Squeeze-Excitation in trunk Res blocks — too slow
- LayerScale (CaiT-style) — within noise
- Adaptive per-class miss-rate weighting — within noise
- Boundary-weighted CE — within noise
- KL-on-logits aux in joint seg loss — within noise (also blew up once due to GPU-OOM, retried fine)
- Inverse class-frequency weighting — slower, worse
- Pixel shuffle upsample — worse
- Split decoders (shared encoder, separate decoder per head) — caused activation memory blowup, retest only with `expandable_segments` allocator
- Deeper trunk (extra Res blocks) — costs epochs, seg crashes
- Wider trunk (C2=80, C1=64) — costs rate AND epochs
- Wider HEAD_HIDDEN (60, 64) — within noise to worse
- Smaller batch (2) or larger (8) — both worse than 4
- SGD+momentum for finetune — worse than AdamW
- AdamW betas (0.9, 0.95) or (0.9, 0.999) — (0.9, 0.99) is the sweet spot
- weight_decay 0.0 or 0.05 — 0.01 default fine
- GroupNorm groups 8 — worse
- GELU activation — within noise
- EMB_DIM 12 (twice) — within noise/worse
- Dual FiLM in Head1 (multiple times) — worse
- Dropout in Head1 — within noise
- LR warmup at start of finetune — worse

## NON-OBVIOUS levers from carefully reading `modules.py` + `prepare.py` + `frame_utils.py`

A fresh agent: read these files BEFORE designing experiments. Specific exploitable details:

### 1. PoseNet uses only the first 6 of 12 output dims
`modules.py:84` and `:101`: `out[h.name][..., : h.out // 2]`. The pose head has `out=12` but distortion uses only `out // 2 = 6`. The other 6 dims (likely log-variances or auxiliary heads) are computed but **ignored**. This means we have free output capacity at posenet's tail that doesn't matter — but more importantly, it tells us the pose loss is on a 6-D vector.

### 2. SegNet is fed ONLY frame2
`modules.py:108`: `x[:, -1, ...] # Use only last frame`. Our `h1` (frame1 generator) does NOT affect seg directly — only indirectly through posenet pairing. This is exploitable: **frame1 doesn't need to look like a frame at all**, it just needs to make `posenet(yuv6(frame1) ⊕ yuv6(frame2))` output the right pose vector.

### 3. PoseNet input is packed YUV420, not RGB
`frame_utils.py:51-78` and `modules.py:74`: input is 6 channels = `[Y00, Y10, Y01, Y11, U_sub, V_sub]` per frame, two frames stacked to 12. Our model generates RGB then `prepare.py:diff_rgb_to_yuv6` converts. Generating YUV directly skips a lossy conversion. Or: shape h1's RGB output to map well through that exact YUV chain.

### 4. PoseNet's normalization is `AllNorm` (BatchNorm1d(1) over flattened tensor)
`modules.py:28-33`. It normalizes the entire (batch × features) tensor as one batch dimension. **Running stats are frozen** in eval mode. This means feature scale matters less than relative pattern — there may be a "shift" of frame1 statistics that posenet doesn't punish.

### 5. The model is FROZEN at eval — adversarial / direct optimization possible
`prepare.py:265`: `apply_fp4_to_model(model)`, then `model.eval()` runs deterministically. `evaluate` uses `torch.inference_mode()`. PoseNet/SegNet weights never change. **Treat them as fixed, differentiable functions and reverse-engineer them.** E.g., gradient ascent in input space to find what frame1 makes posenet output the GT pose. We're already doing this implicitly via the finetune stage, but we could do it explicitly with adversarial-style search at eval time too if we cache activations.

### 6. mask bytes (219KB) and pose bytes (13KB) are FIXED in the rate
`prepare.py:39-41`. Total budget is `bytes / 37.5M`, with mask + pose taking ~232KB regardless of what we do. **Only model bytes vary**, and rate_term = `25 × bytes / 37.5M` ≈ `0.0007 × KB`. A 30KB model reduction = 0.02 score improvement. Going below ~50KB is the only way to push rate_term meaningfully.

### 7. FP4 codebook is `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` with sign bit
`prepare.py:66`. Block size 32, scale stored as 16-bit. Most weights map to one of 16 values. **Weights with low entropy compress much better via brotli** (the 78% ratio in `estimate_model_bytes`). If we structure conv weights to have many repeated values (e.g., explicit weight sharing, low-rank factorization that produces near-identical blocks), brotli will eat them.

### 8. `nn.Linear` weights are stored as 16-bit, NOT FP4
`prepare.py:113`. Conv2d/Embedding get FP4-quantized; Linear and BiasNorm are FP16. **This is why widening pose_mlp / FiLM linears was cheap and won.** Free strategy: shift more capacity into Linear layers (pose pipeline, FiLM projections) and away from Conv2d. Trunk capacity is expensive; pose-side capacity is cheap.

### 9. Pose vector is 6-D (3 trans + 3 rot, or twist coords)
The first 6 elements of posenet output. Different scales (translation in meters? rotation in radians?). MSE weights all 6 equally. **A loss that respects the natural scale (or normalizes per-dim from training-time variance) might be sharper.** Currently SmoothL1 with β=0.1 is just a fixed Huber transition.

### 10. PoseNet uses fastvit_t12 with `gelu_tanh` activation
`modules.py:25-26`. fastvit's first conv is a strided 3×3. **First-layer feature analysis**: probe what spatial/spectral patterns posenet's first conv responds to most strongly, then make h1 produce frames rich in those patterns. This is a "design for the discriminator" trick.

### 11. seg_dist = argmax-disagreement rate, NOT cross-entropy
`modules.py:111-113`. The eval metric is `argmax(out1) != argmax(out2)`. Our training loss uses CE, which is a soft proxy. **The gap between CE-converged and argmax-correct is real**: a logit can flip the argmax with tiny perturbations. Logit-margin-aware losses (e.g., margin loss on the top-2 logits) directly optimize the eval metric.

### 12. Eval uses `comp.round().to(torch.uint8)` before distortion
`prepare.py:286`. Output is rounded to integer pixels (uint8). Our training uses `diff_round` (STE through round). **There may be a small distribution mismatch** between training (smooth gradients through diff_round) and eval (hard round). Trying training-time hard-round on a small fraction of batches could close it.

### 13. Mask compression is fixed; pose compression is fixed
We can't change `MASK_BYTES = 219_588` or `POSE_BYTES = 13_194`. **They're sunk costs**. But the mask is the GT segmentation map — could we train the mask itself? No, prepare.py rejects modifications. The constants are fixed by infrastructure.

### 14. There's a `compute_distortion` helper inside posenet/segnet (modules.py:84, :112)
We're not using it directly in train.py — we recompute manually. Confirms our pose loss formula is right. No exploit, just confirmation.

### 15. `seq_len = 2` exactly, hardcoded in frame_utils.py
Two frames per pair. No room to use temporal context beyond pairs.

### 16. `camera_size = (1164, 874)`, `segnet_model_input_size = (512, 384)`
The eval upscales our 384×512 output to 874×1164 then segnet downscales it back to 384×512. **There's redundant resampling.** Our 384×512 output must survive 2.27× upscale + 2.27× downscale + uint8 quantize to still match segnet's argmax. This means our model's output should have **frequency content up to the post-resample band**, not max sharpness. Designing for the resample chain (e.g., output at slightly lower resolution and explicitly upsample) might preserve more signal.

## Truly out-of-the-box ideas a fresh agent should explore

These are not tried and not obvious:

1. **PoseNet adversarial inversion**: PoseNet is a fixed differentiable function. Optimize frame1 directly (per-pair, at training time) to produce the target pose vector — and use that as a "teacher" target for h1 to imitate. Currently we train h1 end-to-end against pose loss; this gives a 6-D scalar gradient. Inverting gives a per-pixel target that's much richer.

2. **Output YUV directly**: Skip RGB→YUV in the loss path. h1/h2 output 6-channel YUV, then pack with `pack_pair_yuv6` style without re-converting. Removes a lossy stage at training time.

3. **Weight sharing for compression**: Force conv weights in some layers to share parameters (tied weights) so brotli compresses them to almost nothing. Trade slight model expressivity for big rate savings. The Quantizr architecture has plenty of redundancy in trunk Res blocks.

4. **Low-rank conv factorization**: Replace `Conv2d(C1, C1)` with `Conv2d(C1, R) @ Conv2d(R, C1)` for small `R`. FP4 + brotli on a low-rank decomposition compresses much better than the dense form.

5. **Move capacity from Conv to Linear**: Conv weights cost FP4+scale = ~5 bits/param. Linear costs FP16 = 16 bits/param. But after brotli(0.78) and given that the trunk has WAY more conv params than the pose pipeline, the per-param cost calculus might still favor moving capacity into Linear-heavy paths (FiLM projections, pose_mlp, gates). Already partly validated by the wins.

6. **Logit-margin loss for seg**: Instead of CE, use `relu(margin - (top1 - top2))` so the model directly maximizes the gap between the right class's logit and the runner-up. This optimizes the actual eval metric (argmax disagreement) directly.

7. **Train against eval-time uint8 round**: At training time, alternate `diff_round` with hard `.round().to(uint8).float()` (no gradient) every N steps. Forces the model to be robust to the actual quantization noise it'll see.

8. **Posenet first-conv probe**: Compute the gradient of pose output w.r.t. each input pixel for a few pose targets. Identify which spatial regions matter most for pose. Concentrate h1 capacity there (e.g., higher-resolution decoding only in those regions).

9. **Fixed (non-quantized) pose-only frame1**: Generate frame1 as `frame2_yuv ⊕ pose_pattern(pose)` where `pose_pattern` is a tiny deterministic function (no learned weights, no rate cost) that paints a pose-encoding texture. PoseNet learns to read the texture; rate term drops dramatically since h1 has no learned weights.

10. **Distill posenet's intermediate features**: Train h1 to make `posenet.vision(yuv6(h1, h2))` match `posenet.vision(yuv6(real_f1, real_f2))` at an intermediate layer. Richer signal than 6-D pose MSE; easier to optimize.

11. **Use the unused 6 pose dims**: Posenet outputs 12; only 6 are graded. The other 6 may carry uncertainty info that, if matched, indirectly improves the 6 that count. Cheap auxiliary loss.

12. **Two-stage h1**: Generate a tiny low-res frame1 (e.g., 64×96) then upsample with a lightweight learned filter. Most of posenet's response is at low spatial frequency; high-res detail is wasted bytes.

13. **Resample-aware output**: Output at 384×512 designed to round-trip through 874×1164 ↑ and 384×512 ↓ losslessly. Use the eval's bilinear upscale chain explicitly in the training loss.

14. **Per-block quantization-friendly weight init**: Initialize conv weights so they snap cleanly to FP4 codebook values `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`. Reduces QAT-needed adaptation.

## Current chain & how to start the fresh agent

```
HEAD = 1d9746a (exp118: trunk-output FiLM for h1)
Branch: autoresearch/apr29
Best single: 1.415 | Median ~1.5–1.6 (run-to-run noise ~±0.2)
Time budget: 5 min training + eval per experiment
```

Files the fresh agent must read first (in order):
1. `autoresearch/program.md` — research strategy, score breakdown, what NOT to do
2. `modules.py` — frozen SegNet/PoseNet definitions (the discriminators)
3. `frame_utils.py` — `rgb_to_yuv6`, `seq_len`, `camera_size`, `segnet_model_input_size`
4. `autoresearch/prepare.py` — `evaluate()`, `apply_fp4_to_model()`, `estimate_model_bytes()`, `diff_round()`, `diff_rgb_to_yuv6()`
5. `autoresearch/train.py` — current generator + training loop (the only file the agent edits)
6. `autoresearch/results.tsv` — full history of what was tried (don't re-run reverts)

Then attack the eval pipeline, not the architecture.

---

# Prompt for the fresh agent

```
You're picking up an autoresearch loop on the comma video compression challenge. Branch is autoresearch/apr29, HEAD is exp118 (best single proxy score 1.415, median ~1.5). The goal is a lower score on the held-out 20-pair val set, which mirrors the leaderboard ranking on the full 600-pair, 19-hour training run.

BEFORE making any changes, read these files in this order and take notes:
  1. autoresearch/program.md — research rules, score breakdown
  2. modules.py — FROZEN SegNet (Unet+efficientnet_b2) and PoseNet (fastvit_t12+Hydra). These are the eval discriminators. Note especially:
     - SegNet uses ONLY the last frame (line 108)
     - PoseNet output has out=12 but distortion uses only first 6 dims (line 84)
     - Input to PoseNet is packed YUV420, 12 channels (2 frames × 6)
     - AllNorm is BatchNorm1d(1) over flattened tensor — frozen running stats
  3. frame_utils.py — rgb_to_yuv6 layout (line 51): [y00, y10, y01, y11, U_sub, V_sub]; seq_len=2; camera_size=(1164, 874); segnet_model_input_size=(512, 384)
  4. autoresearch/prepare.py — evaluate(), apply_fp4_to_model(), estimate_model_bytes() (Linear=FP16 not FP4!), diff_round(), the FP4 codebook
  5. autoresearch/train.py — the only file you edit
  6. autoresearch/results.tsv — every experiment so far, kept and reverted. Don't re-run reverts.
  7. autoresearch/research_status.md — full handoff notes including what was tried and 16 non-obvious leverage points from reading the eval code

CRITICAL: the previous agent tried ~130 experiments and exhausted incremental architectural variations on the current Quantizr-cloned design. The remaining gap to leaderboard #1 (4×) cannot close by another FiLM variant or pose_mlp tweak. The breakthrough must come from EXPLOITING the eval pipeline. Specifically:

- frame1 doesn't need to look like a frame — SegNet doesn't see it. It just needs to make posenet(yuv6(frame1) ⊕ yuv6(frame2)) output the right 6-D pose vector.
- PoseNet/SegNet are FROZEN, FULLY DIFFERENTIABLE black boxes. Treat them as oracles to invert, distill, or adversarially target.
- Linear weights are stored at FP16 (not FP4). Capacity in Linear layers (FiLM projections, pose_mlp, gates) is much cheaper per-rate than capacity in Conv2d.
- The eval metric is argmax-disagreement, NOT cross-entropy. Optimize the actual metric (logit-margin loss).
- Rate is dominated by Conv2d weights → low-rank factorization, weight sharing, and entropy-friendly structure compress much better via brotli(0.78).
- See research_status.md sections "NON-OBVIOUS levers" (16 items) and "Truly out-of-the-box ideas" (14 items) for specific exploits to consider. Most of those have NOT been tried.

KARPATHY AUTORESEARCH PRINCIPLE — this is the most important rule:

  >>> Search ALGORITHMS, not HYPERPARAMETERS. <<<

Architectural changes, loss formulation, optimizer choice — these tend to transfer from short proxy training to the 19h full run. Hyperparameter values (LR, weight decay, grad clip, EMA decay, schedule timings, ERR_BOOST) are budget-tuned: they help the proxy but mostly evaporate at full scale. The previous agent has 5 hyperparameter wins in the chain that may not transfer; do NOT add more. Lock LR, GRAD_CLIP, EMA decay, T_ANCHOR/T_FT/T_JT, QAT_FRAC, ERR_BOOST at their current values and only modify ALGORITHMIC code.

Specifically, when you find yourself tempted to tune a number, ask: "would this number need to be re-tuned at 22× the compute?" If yes, it's budget tuning — don't keep it on the proxy chain.

Express schedules as fractions of total steps, not absolute epoch counts. Schedules tuned to "epoch 25" or "last 5%" usually break at full scale.

When a result is within ±0.15 of the current best on a single run, it's noise — re-run before deciding (or just revert). Don't accumulate noise wins.

Run one experiment, decide keep/revert based on >0.15 improvement, commit-or-reset, repeat. Never stop. 5 minutes per run, ~10/hour, ~100 overnight. Use `python autoresearch/train.py > autoresearch/run.log 2>&1` with PYTHONUNBUFFERED=1 for live log.

The previous agent's full log lives in autoresearch/results.tsv. Read it before each experiment to avoid re-running reverts. Memory of accumulated lessons is in ~/.claude/projects/.../memory/autoresearch_lessons.md.

GO. Start by reading the files, then propose your first algorithmic experiment that targets one of the eval-pipeline exploits.
```
