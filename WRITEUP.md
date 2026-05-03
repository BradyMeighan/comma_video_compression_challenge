# From 4.39 → 0.30: a small mask-conditioned generator + autoresearch + sidecar patches

**Final score: ~0.30** (raw model 0.33 + sidecar 0.2999, both reproducible from this branch).
**Compression rate: 7.6×** (uncompressed 37.5 MB → compressed 286 KB).
**Approach:** train a tiny FP4-quantized generator that reconstructs frames from a stored 5-class semantic mask + 6-D pose vector, then optionally repair the worst pairs with a few KB of byte-level "sidecar" patches that exploit the eval pipeline's own gradients.

![score journey](writeup_assets/score_journey.png)

---

## 1. The model

The challenge scores submissions by `100 × seg_dist + √(10 × pose_dist) + 25 × rate`, where seg/pose distortions are measured by frozen SegNet and PoseNet "discriminators". Two facts shaped the entire architecture:

1. **SegNet only sees frame 2** of each pair ([modules.py:108](modules.py)). So frame 2 needs to look "right" to a U-Net+EfficientNet, but **frame 1 doesn't have to look like a frame at all** — it just needs to make `posenet(yuv6(f1) ⊕ yuv6(f2))` output the correct 6-D pose vector.
2. **Conv2d weights are stored as FP4** in `prepare.estimate_model_bytes` (4-bit codebook + 16-bit per-block scale, then brotli), but `nn.Linear` is stored at FP16. So Linear capacity costs ~4× more per parameter than Conv2d capacity — but capacity in pose-conditioning Linears (FiLM, MLP) is *much* cheaper to compress than capacity in the trunk.

We compress each video as a per-pair tuple `(mask, pose)` plus a single shared 54 KB generator. The generator is a 92 K-parameter U-Net with two heads:

![architecture](writeup_assets/architecture.png)

The trunk is shared between both frames; only Head 1 (frame 1, the pose-target frame) gets pose conditioning via two FiLMRes blocks, with the FiLM linears wrapped in 1×1 `QConv2d` so they get FP4 byte treatment instead of FP16. Quantization-Aware Training (QAT) kicks in at 70% of the anchor stage so the model learns to live with the FP4 codebook `[0, 0.5, 1, 1.5, 2, 3, 4, 6]`.

What the trained generator actually produces:

![hero pair 60](writeup_assets/hero_pair_60.png)

The reconstructions are visually unrecognizable — saturated, cartoon-like, with hallucinated colors — but they hit `seg_dist ≈ 0.027` and `pose_dist ≈ 0.08` on the held-out 600-pair test set. This is the central trick: when your evaluator is a neural network, you don't optimize for pixel fidelity, you optimize for whatever activation the evaluator happens to read out.

![generator output animated](writeup_assets/gen_reconstruction.gif)

The road geometry, sky/foliage band, and lane markings are clearly visible in the right column despite never being supervised by a pixel-level loss — they emerge purely as the *easiest* way for the generator to satisfy SegNet's argmax. A static grid with daytime/dusk pairs:

![generator grid](writeup_assets/gen_compare_grid.png)

---

## 2. Karpathy-style autoresearch loop

The generator's architecture wasn't designed by hand — it was found by ~195 short-budget proxy experiments, each a self-contained 5-minute training run. The loop follows Karpathy's nanoGPT-speedrun rule: **search algorithms, not hyperparameters**. An LLM agent reads the previous result, proposes a single change to `train.py` (architecture, loss formulation, optimizer choice), runs the proxy, decides keep/revert based on whether the score improved by more than the run-to-run noise floor (≈ ±0.15), commits or resets, and repeats. Hyperparameter values (LR, EMA decay, schedule timings) are deliberately *not* tuned in the loop — those rarely transfer from a 5-minute proxy to a 12-hour full run, while algorithmic wins usually do.

![autoresearch progression](writeup_assets/autoresearch_progression.png)

The chart shows the agent converging from baseline 2.14 to 1.36 over 195 experiments (green = kept, gray = reverted, red = running best). The vast majority of attempts fail; a handful of architectural decisions did most of the work. The keepers, in order:

| exp | change | proxy score | category |
|---|---|---|---|
| 11 | pose-finetune loss MSE → SmoothL1 (β=0.1) | 1.57 | algorithmic |
| 17 | pose_mlp 2-layer → 3-layer | 1.40 | architectural |
| 37 | Head1 concat-pose merge | 1.46 | architectural |
| 81 | 3-step ERR_BOOST schedule | 1.39 | schedule |
| 112 | COND_DIM 48 → 64 | 1.52 | architectural |
| 118 | trunk-output FiLM modulating h1 only | 1.42 | architectural |
| 168 | trunk_film FP4 via QLinear | 1.61 | rate (algorithmic) |
| 169 | FiLMRes.film FP4 zero-init | 1.51 | rate (algorithmic) |
| 171 | focal loss (γ=2) replaces ERR_BOOST | 1.45 | algorithmic |
| 172 | dual-FiLM Head1 with zero-init second FiLM | 1.45 | architectural |
| 182 | dual FiLMRes Head1 (final shape) | **1.36** | architectural |

The two highest-leverage *categories* turned out to be:

- **FP4-wrapping zero-initialized Linears** (exp 168/169). Wrapping a `Linear` in a 1×1 `QConv2d` saves ~9 KB per Linear because Conv weights are charged at FP4 (~5 bits/param) while plain Linears are charged at FP16. The constraint is that the init must be near-zero — Kaiming-initialized pose_mlp Linears tank pose at FP4 (exp 170 hit 1.5 pose). Total rate savings from this trick alone: ~25 KB → score −0.013.
- **Focal loss + boundary weighting** for the segmentation objective. The eval metric is `argmax-disagreement`, not cross-entropy ([modules.py:111](modules.py)) — the eval doesn't care if the wrong class has high logit, only that the right class has the highest. Focal loss (γ=2) downweights already-confident pixels and concentrates gradient on the hard ones, and `5×` boundary weighting concentrates further on the pixels where a single logit perturbation flips an argmax. Replicated wins across exp 164/171/186.

Things that *clearly* didn't work and shouldn't be retried by anyone forking this repo: deeper or wider trunks (cost epochs, seg crashes), pose-conditioning the frame-2 head (gradients from pose pull the trunk away from seg), Fourier pose encoding, optical-flow warps, residual `f1 = f2 + δ`, EMA decay > 0.95 on the proxy, and ~30 other variants documented in [autoresearch/results.tsv](autoresearch/results.tsv).

---

## 3. Training: A100 + targeted 3090 fine-tune

Once the architecture froze at exp 182, we ran the full training schedule on Colab A100 for 12 hours at batch size 16 — a three-stage curriculum:

![training curves](writeup_assets/training_curves.png)

1. **Anchor stage (55%)** — train Head 2 + trunk against `frame2 → segnet` with focal loss. QAT enables at 70% of this stage; you can see the loss spike from ~5 to ~17 when the FP4 codebook turns on at minute ~193, then re-converge.
2. **Finetune stage (27%)** — freeze Head 2, train Head 1 + pose_mlp against the pose target only (smooth-L1 over the 6-D PoseNet output). Loss drops cleanly by 2 orders of magnitude on the log y-axis.
3. **Joint stage (13%)** — unfreeze everything, train against the full `100·seg + √(10·pose)` loss.

The A100 12h run produced a model at score ≈ 0.41. We then noticed that pose loss was still trending down at the very end of joint, and that we had spare 3090 time, so we ran a 4-hour "joint+" continuation with `pose_weight=60` and `jt_lr=1e-5` (one third of the original joint LR). That alone took the score from 0.41 → 0.33. **The clean lesson: with the 0.7 EMA decay used in the final phase, the model was still in a noisy regime; a longer, lower-LR finish on more data converges to a meaningfully better point.**

The bigger surprise was the second continuation. We restarted from the e80 checkpoint of the 3090 run with an even smaller LR (`jt_lr=2e-6`, ~16× smaller) and lower `pose_weight=30`. That cut another ~0.03 of pose distortion, bringing the model to **score 0.2986** at epoch 0 of v3 (the model from gen_3090.pt.e80.ckpt). The targeted fine-tune does not touch architecture — it only repaints the weights into a slightly flatter local minimum that the final FP4 quantization rounds onto more cleanly. In retrospect this is consistent with Karpathy's observation that wide minima are friendlier to quantization, but we found it empirically by leaving an `EVAL_EPOCH_INTERVAL=10` watcher on the run and noticing that the periodic checkpoints were oscillating in a 0.005 band — that band shrank as LR shrank.

---

## 4. Sidecar patches (work-in-progress)

After the model converged we ran out of training-side leverage but had a few KB of rate budget left. The sidecar idea: the eval networks are *frozen, fully differentiable* — treat them as oracles to invert. For each pair, find a tiny byte-level perturbation to either the input mask, the input pose, or the output frame that lowers `seg_dist + pose_dist` more than the per-byte rate cost.

![sidecar pipeline](writeup_assets/sidecar_pipeline.png)

We ended up shipping five complementary methods, each searched per-pair, each compressed to its own bz2 stream and finally LZMA2-wrapped together:

- **X2** — flip a 2×2 mask block. One block changes the entire feature map via the conv receptive field, so the leverage per byte is enormous. We use *verified greedy* (try the top-N candidates from the gradient, actually re-run the generator, accept if loss drops) — pure gradient direction is unreliable because multiple flips compound destructively. ~64% acceptance rate, 5 B/patch.
- **CMA-ES single-pixel mask flips** for the hardest 100 pairs that X2 didn't fully fix. CMA-ES over `K=2` flips chosen from the top-30 gradient positions, batched per generation.
- **S2** — variable-shape mask flips (1×1, 3×3, 1×4 strip, 4×1 strip, 2×2). Strips capture road edges efficiently.
- **C3** — pose-vector deltas via int8 grid search over dimensions 1, 2, 5 (the dominant residuals). Adam-via-autograd through the FP4-quantized generator hung indefinitely; gradient-free 7³ grid search worked reliably.
- **Channel-only RGB patches** (250 hardest pairs get K=5, next 250 get K=2). Modify a single color channel of a single pixel of the *output* frame after the generator. PoseNet's first conv reads YUV at quarter-res, so single-channel single-pixel changes have outsized influence per byte.

The final unlock was **per-pair method selection**: for each of the top 600 pairs, evaluate every subset of `{X2, CMA-ES, S2, C3}` (up to 16 combos, batched into one generator forward) and pick the subset that minimizes pose loss. 110 of those 600 pairs ended up better with *no* sidecar patches than with any subset — meaning even the cheapest greedy patches were sometimes hurting. Combining method selection with an LZMA2 outer wrap (which exploits coordinate repetition across streams) got us from raw model 0.3318 → 0.2999.

The sidecar exploration is not finished. Things we wanted to try but didn't ship:

- Pose-vector deltas on all 1200 pairs (cheap method, currently only top 200).
- 5-bit packed `pattern_id + class` field (saves ~150 B → ~−0.0001).
- Adversarial-decode at *evaluation time*: instead of storing patches, run gradient descent through the eval networks at inflate time to refine each frame in-place. We have a working prototype (`submissions/adversarial_decode/inflate.py`) but the 30-minute eval-time budget on a T4 is tight and we ran out of clock to tune it.
- Mask-pixel flips drawn from a learned codebook so each flip is 2 bytes instead of 5.

---

## 5. Reproducing this submission

```bash
# Train (A100, 12h)
cd autoresearch && CONFIG=B FULL_DATA=1 EMA_DECAY=0.999 COSINE_LR=1 \
  GRAD_CLIP_OVERRIDE=0.5 TRAIN_BUDGET_SEC_OVERRIDE=$((12*3600)) \
  python train.py > run.log 2>&1

# Continue on 3090 (4h, joint+ stage with smaller LR + bigger pose weight)
MODEL_PATH=colab_run/gen.pt SAVE_MODEL_PATH=colab_run/gen_continued.pt \
  TRAIN_BUDGET_SEC_OVERRIDE=$((4*3600)) POSE_WEIGHT=60 JT_LR_OVERRIDE=1e-5 \
  python continue_train.py

# Targeted fine-tune (1.5h, smaller LR + smaller pose weight)
MODEL_PATH=colab_run/3090_run/gen_3090.pt.e80.ckpt SAVE_MODEL_PATH=colab_run/gen_v3.pt \
  TRAIN_BUDGET_SEC_OVERRIDE=$((90*60)) POSE_WEIGHT=30 JT_LR_OVERRIDE=2e-6 \
  EMA_DECAY=0.9995 python continue_train.py

# Build sidecar (per-pair selection of X2 + CMA-ES + S2 + C3 + RGB)
python autoresearch/v2_per_pair_select.py
```

All training scripts, the autoresearch results table, and the sidecar pipeline code are in [autoresearch/](autoresearch/). The full per-experiment log is at [autoresearch/results.tsv](autoresearch/results.tsv) and the agent's running notes at [autoresearch/research_status.md](autoresearch/research_status.md).
