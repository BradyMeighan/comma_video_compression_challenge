# Closing the last 2.4%: a battle plan for near-perfect student distillation

**The single most effective path to 99.5%+ pixel agreement combines three changes: replacing standard convolutions with inverted residual blocks (~6× effective capacity gain at same param count), adding a learnable low-rank spatial bias (~39K params encoding dashcam spatial priors), and switching to boundary-weighted channel-wise knowledge distillation.** These three interventions attack the problem at its root — the current MicroUNet lacks the effective capacity and spatial inductive bias to represent sharp class boundaries, while the distillation loss fails to concentrate learning on the ~2.4% of pixels where errors live. Because this is a memorization problem on 600 near-identical frames, not a generalization problem, every standard regularization technique should be removed and training should be pushed to convergence. An alternative "train big then compress" strategy using INT8 QAT offers an even safer path: train a 1–1.5M-param model to 99.5%+, then quantize to fit the ~1MB budget.

---

## The architectural bottleneck is effective capacity, not parameter count

The current MicroUNet uses standard 3×3 convolutions with base_ch=32. A standard 3×3 conv with 32 input and 32 output channels costs 9,216 parameters but produces only 32-dimensional intermediate features. **Replacing every standard 3×3 conv with a MobileNetV2-style inverted residual block (expand→depthwise 3×3→project) yields ~7× more parameters-per-effective-channel.** At expansion factor t=4, a 32-channel inverted residual block processes through 128-channel intermediate features while costing only ~1,312 parameters for the depthwise portion plus ~4,096 for the pointwise projections — roughly 5,400 total versus 9,216 for a standard conv, yet with **4× wider intermediate representations**.

This means a MobileUNet with base_ch=48 and inverted residual blocks fits in ~480K parameters but processes features through 192-channel intermediates — **6× the effective feature width** of the current architecture. For memorization of boundary details, this wider intermediate representation is critical: each spatial location needs enough channel dimensions to encode the nuanced class transition patterns that distinguish road-from-sidewalk at pixel-level precision.

Lightweight segmentation architectures validate this approach. Fast-SCNN achieves 68% mIoU on Cityscapes at 1.1M params, BiSeNet V2 reaches 72.6% at 156 FPS, and LERNet hits 69.5% mIoU at only 650K params — all using depthwise-separable or inverted residual blocks. These are generalization benchmarks; for pure memorization on 600 similar frames, the capacity headroom is enormous.

---

## Spatial bias and Fourier encoding exploit dashcam's rigid structure

Dashcam video has extraordinarily strong spatial priors: sky occupies the upper third, road the lower half, and the horizon sits at a near-constant vertical position. The current MicroUNet must burn network capacity rediscovering this spatial layout for every pixel in every frame. Two techniques eliminate this waste almost for free.

**A learnable low-rank spatial bias** added directly to the output logits gives the network a "base prediction" of the spatial class layout. A rank-16 factorization of a (5, 384, 512) bias tensor costs only **~39,000 parameters**: two matrices of shapes (5×384×16) and (16×512). This captures the dominant spatial modes — the five class regions as smooth spatial fields — and reduces the CNN's job from predicting all ~197K pixels to correcting the ~2.4% where the base map is wrong. For 600 frames from a single 1-minute drive, the spatial class distribution is so stable that even a low-resolution (96×128) learnable bias at **61K params** bilinearly upsampled to full resolution would capture ~95%+ of pixels correctly. The parameter cost trades favorably against the capacity the network currently wastes learning this spatial prior implicitly.

**Fourier positional encoding** (Tancik et al., 2020) adds high-frequency spatial basis functions that enable sharp boundary representation. The key insight from the NeRF literature is that standard networks exhibit "spectral bias" — they learn low-frequency functions first and struggle with the high-frequency transitions at class boundaries. Mapping each (x,y) coordinate through L=8–10 log-spaced frequency bands produces 32–40 Fourier feature channels (sin and cos at each frequency) that the network can linearly combine to represent arbitrarily sharp spatial transitions. The encoding itself has **zero trainable parameters**; the only cost is expanding the first conv layer from 3 to ~35 input channels, adding approximately **11,500 parameters**. Evidence from NeRF confirms this works: every successful neural radiance field uses positional encoding precisely because it enables memorization of high-frequency scene detail.

Simple CoordConv (appending raw normalized x,y coordinates) costs only ~576 parameters and provides a weaker but still useful positional signal. It should be viewed as the minimum baseline; Fourier encoding subsumes and extends it.

**Gradient quality for all three techniques is excellent.** Spatial bias is a linear addition (gradient = 1.0), Fourier features use smooth sin/cos functions with bounded gradients, and CoordConv appends constants. None introduce gradient discontinuities that would harm adversarial decode.

---

## Boundary-focused distillation directly targets the 2.4% error gap

The remaining errors cluster at class boundaries — precisely where the teacher's soft probabilities contain the richest "dark knowledge." Three complementary techniques attack this directly.

**Boundary-weighted KD loss** multiplies the per-pixel KL divergence by the inverse distance to the nearest class boundary: w(p) = 1 + α/(d(p) + ε), with α tuned to give boundary pixels 10–50× higher weight. This is trivial to implement (precompute distance transforms from teacher argmax maps) and forces the optimizer to allocate capacity to boundary accuracy. For a model already at 97.6%, the easy interior pixels contribute near-zero gradient signal — boundary weighting eliminates this wasted gradient flow and concentrates learning where it matters.

**Kervadec boundary loss** (MIDL 2019) reformulates boundary accuracy as a regional integral, multiplying softmax outputs by a precomputed signed distance map. On medical segmentation benchmarks, this improved Dice scores by **up to 8%** and Hausdorff distance by **up to 10%** over baseline regional losses. The loss must be combined with a regional loss (Dice or CE) using a scheduling parameter α that decreases from 1.0 to 0.01 over training, gradually shifting focus from bulk accuracy to boundary precision.

**Channel-wise knowledge distillation (CWD)**, published at ICCV 2021, is currently the state-of-the-art for segmentation distillation. Rather than matching features spatially, CWD normalizes each channel's feature map into a probability distribution via softmax, then minimizes KL divergence channel-by-channel. This focuses learning on the most salient spatial regions per feature channel. CWD improved PSPNet-R18 by **3.83% mIoU** on Pascal VOC and consistently outperformed Attention Transfer and spatial feature matching. For dimension mismatch between teacher and student feature maps, a 1×1 conv adapter (a few hundred parameters) maps student features to teacher dimensions.

An **auxiliary boundary detection head** — two conv layers (~10K params) predicting binary boundary maps — forces the shared encoder features to become boundary-aware during training. This head is removed at inference, adding zero runtime cost. The Semantic Boundary Conditioned Backbone (SBCB, 2023) framework showed that this technique visually conditions backbone features to highlight boundaries.

Combined, these boundary-focused techniques should contribute **+1.0–2.0% pixel agreement**, potentially pushing from 97.6% into the 98.6–99.6% range when paired with architectural improvements.

---

## Input gradient alignment is essential for adversarial decode quality

Standard KD transfers the teacher's output behavior but **fails to preserve the teacher's gradient landscape**, which is critical for adversarial decode. KDIGA (Shao et al., 2022) proved theoretically that aligning student and teacher input gradients preserves certified robustness — and by extension, ensures the student's loss landscape mirrors the teacher's for input-space optimization. The KDIGA loss term is:

**L_IGA = ‖∇ₓ f_student(x) − ∇ₓ f_teacher(x)‖²**

This is computationally expensive (requires computing input gradients for both models per training step, roughly 3× training time), but with only 600 frames and unlimited offline compute, it is entirely feasible. The IGDM paper (2023) further showed that matching input gradients guarantees output matching along the gradient direction — precisely the property needed when adversarial decode follows gradients through the student to generate frames.

Feature-matching distillation methods (FitNets, CWD, Attention Transfer) implicitly improve gradient quality because similar internal representations produce similar input-gradient structures. But only explicit gradient alignment guarantees it. **This should be added as a final training phase** after the model has converged on standard KD + boundary losses, fine-tuning for 5–10 epochs with the gradient alignment term.

---

## The "train big, then compress" alternative may be the safest path

If architectural improvements alone cannot close the gap, the highest-confidence strategy is to decouple the accuracy problem from the size problem entirely. Li et al. (ICML 2020) demonstrated that **heavily compressed large models achieve higher accuracy than lightly compressed small models at the same parameter budget**.

The size budget analysis reveals substantial headroom:

| Quantization | SegNet budget (~1MB) | Trainable params |
|---|---|---|
| FP16 | 500K params | Current territory |
| **INT8** | **1.0M params** | **2.4× current model** |
| INT4 | 2.0M params | 4.7× current model |

**At INT8 precision, a 1.0M-parameter model fits in ~1MB** — more than double the current architecture's capacity. Train this larger model with all the techniques above (inverted residual blocks, spatial bias, Fourier encoding, boundary-weighted CWD), achieving 99.5%+ accuracy, then apply INT8 quantization-aware training. QAT for segmentation models typically incurs **<1% mIoU loss** at INT8, well within the accuracy margin.

For adversarial decode, the quantized weights can be **dequantized to FP16/FP32** at runtime — the ~0.4% quantization error means gradients remain ~99% faithful to the full-precision version. This sidesteps the gradient quality concerns entirely. T4 GPUs natively support INT8 Tensor Core inference at **16× throughput** relative to FP32, and the smaller model means faster per-iteration adversarial decode.

**Structured pruning** (removing entire channels by importance) is the safest compression method for gradient quality because the surviving network is fully dense. If INT8 alone doesn't meet the size target, prune 20–30% of channels post-QAT and fine-tune for 10% of original training epochs. The Self-Compression approach by Cséfalvay & Imber (arXiv 2301.13142) elegantly makes bit-depth a differentiable per-channel parameter, but its aggressive compression (3% of bits remaining) likely degrades gradient quality beyond acceptable levels for adversarial decode.

**Critical warning**: Post-training quantization (PTQ) without QAT will likely fail catastrophically on small models. One benchmark showed DeepLab+MobileNetV2 dropping from 87.8% to 54.6% mIoU with naive PTQ. Small models lack the redundancy to absorb quantization noise — QAT is mandatory.

---

## Hash-grid encodings are theoretically ideal but practically complex

Multi-resolution hash encoding (Instant-NGP, Müller et al., 2022) is architecturally designed for exactly this kind of problem — memorizing a single scene's spatial structure at multiple resolutions. A hash grid with L=12 levels, T=2¹² entries, and F=2 features per entry costs **98,304 parameters** and provides 24-dimensional per-pixel spatial features that encode learned multi-scale spatial structure.

The fundamental challenge is that hash grids map (x,y) → features, learning a fixed spatial function, while this task requires (image, x, y) → class label. The hash grid can serve as a powerful spatial prior that a small CNN modulates based on image content — but this hybrid architecture requires careful design. The hash grid would encode "at this location, the default class distribution is [0.9 sky, 0.05 building, ...]" while the CNN provides frame-specific corrections.

Implementation requires custom CUDA kernels (the `tiny-cuda-nn` library) for efficiency, and integration with a UNet is non-trivial. **For a team already running adversarial decode pipelines, the engineering investment may not justify the marginal improvement over simpler spatial bias + Fourier encoding**, which capture most of the same spatial prior with standard PyTorch operations. However, if the simpler approaches plateau below 99.5%, hash grids represent the highest-ceiling spatial encoding option.

---

## Lottery ticket pruning is counterproductive for memorization

Multiple papers confirm that **sparsity impairs memorization**. Hoefler et al. (JMLR 2022) surveyed 300+ papers and concluded that "sparsity only revealed that sparsity impairs memorization." The Sparse Double Descent phenomenon (He et al., ICML 2022) found that moderate sparsity actually *worsens* overfitting behavior in a paradoxical way — it removes capacity needed for memorization without providing regularization benefits. Since the explicit goal here is memorization of 600 frames, pruning approaches should be used only as a post-training compression step (structured pruning of converged models), never as a training strategy.

---

## Concrete implementation plan, ranked by expected impact

The following combines all findings into a phased plan. Each phase builds on the previous, with expected cumulative accuracy targets.

**Phase 1 — Architecture redesign (days 1–3, target: 97.6% → 99.0%+)**

Replace all standard 3×3 convolutions with inverted residual blocks (expansion factor t=4). Increase base channels from 32 to 44–48. Add rank-16 factorized spatial bias on output logits (~39K params). Append L=8 Fourier-encoded coordinate channels to input (~12K additional first-layer params). Remove all regularization (dropout, weight decay, augmentation) — this is pure memorization. Total architecture: ~480K params.

**Phase 2 — Boundary-focused distillation (days 3–7, target: 99.0% → 99.4%)**

Replace output-only KD with multi-component loss: (a) pixel-wise KL divergence at T=3 with boundary distance weighting (20× at boundaries), (b) CWD on final encoder features via 1×1 adapter, (c) Kervadec boundary loss with α-scheduling from 1.0 to 0.01, (d) auxiliary boundary detection head during training (removed at inference). Train for 100+ epochs with aggressive learning rate (cosine annealing from 1e-3 to 1e-6).

**Phase 3 — Gradient quality and hard example mining (days 7–12, target: 99.4% → 99.5%+)**

Add KDIGA input gradient alignment loss for final 10–20 epochs of training. Run adversarial decode to identify remaining disagreement pixels. Create a boundary-focused retraining dataset with 50× sampling weight on disagreement regions. Retrain for 20 epochs on this weighted dataset. Repeat the adversarial-decode-and-retrain cycle 2–3 times.

**Phase 4 — Compression for deployment (days 12–15)**

Apply INT8 quantization-aware training (fake quantization nodes with STE) for final 20 epochs. Expected size: ~480KB at INT8. If size budget allows, skip compression entirely and deploy at FP16 (~960KB). For adversarial decode runtime, dequantize to FP16 for gradient computation.

**Alternative fast path — train big then compress (if Phase 1–3 stalls below 99.3%)**

Train a 1.2M-param model (same architecture, base_ch=64) with INT8 QAT from the start. This model will easily reach 99.5%+ on 600 frames. Apply structured pruning to remove ~20% of channels. Final size: ~960KB at INT8. Dequantize for adversarial decode gradients.

---

## Conclusion

The 97.6% plateau is not a fundamental capacity limit — it reflects three correctable deficiencies: inefficient use of the parameter budget (standard convolutions waste ~6× capacity versus depthwise-separable alternatives), absence of spatial inductive bias (forcing the network to learn dashcam spatial structure from scratch), and a distillation loss that treats boundary pixels the same as trivially-correct interior pixels. **The highest-ROI single change is replacing standard convolutions with inverted residual blocks**, which alone may push accuracy past 99% by providing sufficient effective capacity to memorize boundary details. Adding a learnable spatial bias and Fourier coordinate encoding costs fewer than 51K parameters combined and eliminates the network's need to spend capacity on spatial layout. Boundary-weighted CWD distillation then concentrates all remaining learning on the ~2.4% of pixels that actually need it. The "train big then INT8 compress" alternative offers the safest path if the <500K direct approach proves insufficient — at INT8, a 1M-param model fits in ~1MB while providing more than enough capacity for perfect memorization of 600 frames.