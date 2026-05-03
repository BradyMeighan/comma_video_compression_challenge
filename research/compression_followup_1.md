# Neural network compression for adversarial decode: five paths to 500 KB

**SVD compression preserves gradients but fine-tuning destroys compressibility — and that tension defines the entire engineering challenge.** Across five research directions, the most promising combined strategy is: a conditioned SIREN (~372 KB) to replace the SegNet, a differentiable photometric pose proxy (0 KB) or PoseLUT (~126 KB) to replace the PoseNet, and Riemannian-constrained low-rank training to keep any residual learned weights on a compressible manifold. Each approach has mature PyTorch tooling and strong theoretical grounding. What follows is a deep analysis of each path, with concrete architectural recommendations and known failure modes.

---

## 1. Staying on the low-rank manifold during fine-tuning

The core problem is well-characterized: standard gradient descent operates in the full parameter space and has no geometric reason to stay near a low-rank subspace. Three families of solutions exist, ordered by practicality.

**LoRA-on-SVD (simplest, guaranteed compressible).** Freeze the SVD-compressed weights W₀ = U_r·S_r·V_rᵀ and train only a low-rank residual ΔW = A·Bᵀ, where A ∈ ℝ^{m×r_lora} and B ∈ ℝ^{n×r_lora}. The final weight W = W₀ + A·Bᵀ is reconstructable from compact factors by construction. Storage per layer: r(m+n+1) + 2r_lora(m+n) values. AdaLoRA (ICLR 2023) extends this with SVD-parameterized adaptation and dynamic rank pruning, while DoRA (ICML 2024) decomposes into magnitude and direction for better performance, with HuggingFace PEFT support for Conv2d layers. The limitation is that LoRA parameters add *on top of* the frozen SVD factors, so at a 500 KB budget the residual capacity is extremely tight.

**GeoTorch hard constraints (most principled).** The GeoTorch library provides a one-line API — `geotorch.low_rank(self.conv, "weight", rank=r)` — that constrains Conv2d weights to a fixed-rank manifold using Riemannian retraction at each optimizer step. Weights physically cannot leave the manifold. The computational overhead is **~15–25%** per step for the retraction/projection operation. CDOpt offers an alternative "constraint dissolving" approach that transforms manifold-constrained optimization into unconstrained optimization, letting standard Adam/SGD work without modification. For the Stiefel manifold specifically, PyTorch natively supports `torch.nn.utils.parametrizations.orthogonal()` since version 1.9.

**SVD training with regularization (most flexible).** Yang et al. (CVPRW 2020) decompose each weight W = U·diag(s)·Vᵀ as trainable variables, adding orthogonality regularization ‖UᵀU − I‖² and Hoyer sparsity on singular values. The most directly relevant recent work is **Low-Rank Prehab** (Qin et al., Dec 2025), which introduces a pre-compression fine-tuning stage using spectral regularization on Fisher-whitened weights. Their key insight: "compression loss arises because training drifts away from the manifold of low-rank, near-optimal solutions." BSR (2023) combines beam-search rank selection with modified stable rank regularization for smoother training dynamics. GaLore (ICML 2024, Meta) takes a different tack by projecting *gradients* into low-rank subspaces rather than constraining weights, reducing memory 65.5% while allowing full-parameter learning.

**Practical recommendation for this pipeline:** Initialize with SVD of pretrained weights, apply 3–5 epochs of Low-Rank Prehab to encourage spectral compactness, then SVD-truncate and fine-tune the factors using GeoTorch's `low_rank` constraint. For a Conv2d(64, 128, 3×3) at rank 4, this stores just **2,820 values** (5.6 KB at float16). The key failure mode is fixed-rank rigidity — if rank is too low for a critical layer, quality degrades disproportionately. Per-layer adaptive rank allocation (AdaLoRA-style) mitigates this.

---

## 2. SIREN replaces the UNet at 372 KB

Instead of compressing a 38.5 MB UNet, training a compact implicit neural representation from scratch offers a fundamentally different path. After evaluating hash-grid MLPs, SIREN, Fourier feature networks, GaborNet MFN, and tiny ConvNets, **conditioned SIREN emerges as the clear winner** for adversarial optimization.

**Why SIREN dominates on gradient quality.** Any derivative of a SIREN is itself a SIREN — the derivative of sin is cos, which is a phase-shifted sin. This guarantees **C∞ smooth Jacobians at all orders**, exactly what the adversarial decode pipeline requires. By contrast, hash-grid neural fields (Instant-NGP style) induce high-frequency noise in spatial derivatives. Chetan et al. (CVPR 2025) measured **4× angular error** in autodiff gradients from hash-grid fields compared to polynomial-fitting alternatives. Hash collisions create discontinuities that corrupt adversarial signal — a deal-breaker for this use case.

**The recommended architecture** uses FiLM-conditioned SIREN with per-image latent codes:

- Input: normalized pixel coordinates (x, y) ∈ [0,1]² plus a 48-dim learned latent code z_i per image
- Layer 0: FiLM-conditioned SIREN layer where the latent modulates gain γ and bias β of the sinusoidal activation
- Layers 1–3: Standard SIREN layers, 192-wide, with sin(ω₀·(W·h + b)) activations
- Output: Linear layer → C segmentation logits

**Parameter breakdown at float16:** per-image latents (1000 × 48) = 96 KB; FiLM projections = 36 KB; SIREN layers ≈ 120K params = 240 KB. **Total: ~372 KB**, well under the 500 KB budget with room for the pose component.

The training strategy is knowledge distillation: run the teacher UNet on all training images to generate soft segmentation maps, then train the SIREN student from scratch using MSE on soft logits (preserving gradient information better than hard labels). COIN (Dupont et al., 2021) demonstrated that an 8K-param SIREN can fit a single 393K-pixel image, and COIN++ (2022) extended this with meta-learned modulations for multi-image compression. IOSNet (MICCAI 2022) and NOIR (2026) apply coordinate-based INR decoders to medical image segmentation with competitive Dice scores versus UNet, validating the approach for dense prediction.

**GaborNet MFN is the strong second choice.** Multiplicative filter networks create an exponential number of basis functions from linear parameters through element-wise multiplication of Gabor wavelets, providing excellent compression ratios with smooth gradients and analytically characterizable frequency spectra. A 4-layer GaborNet fits in ~200 KB. The ecosystem is less mature than SIREN's but the Bosch Research implementation is available.

**Expected quality tradeoff:** A 372 KB SIREN memorizing 1000 images at 256×256 must encode ~65.5M pixel-label pairs in ~120K learnable params (~520 data points per parameter). Expect **85–92% pixel accuracy** versus the teacher's ~95%+, with 3–8% Dice score degradation. The SIREN initialization is sensitive — the specific uniform initialization from Sitzmann et al. is required, and ω₀ (default 30 for first layer) strongly controls representable frequencies.

---

## 3. Codebook quantization bridges the INT5-to-INT4 gap

Uniform INT4 quantization fails at 63% accuracy because it creates a **16-level staircase function** where the step size doubles versus INT5, quadrupling quantization variance (Δ²/12). Gradient errors compound multiplicatively across layers — with N sequential layers and per-layer error ε, total Jacobian error scales as (1+ε)^N − 1. The critical insight is that **forward accuracy and gradient quality degrade on different curves**: a model can have tolerable forward accuracy with a completely destroyed gradient landscape.

**Non-uniform codebook quantization with 16 centroids dramatically outperforms uniform INT4.** K-means centroids concentrate where weights cluster (near zero), providing fine resolution where it matters most. Outlier weights — which are critical for Jacobian structure — get their own centroids rather than being rounded to distant uniform levels. SqueezeLLM (ICML 2024) demonstrates this with Hessian-weighted k-means plus sparse decomposition for the most sensitive weights. The gradient ∂loss/∂centroid is well-defined and smooth since each centroid affects many weights simultaneously. A key architectural advantage: **codebook centroids can be made differentiable parameters** during adversarial optimization, allowing gradient flow even through quantized weights.

**HAWQ-V2's Hessian trace identifies the gradient-critical layers.** Computing per-layer Hessian traces (~20 gradient backprops) directly measures sensitivity. Layers with high trace values are exactly the layers that destroy gradients when quantized aggressively. The recommended mixed-precision strategy assigns 5–6 bits to high-sensitivity layers and 3–4 bits with k-means codebooks to less sensitive layers, achieving an average of **~3.5–4.0 bits with substantially better gradient quality** than uniform INT4.

**However, quantization alone cannot reach the 500 KB target.** For 23.5M combined parameters, 500 KB implies 0.17 bits/param — far below even binary networks. Even aggressive pruning (90%) combined with 4-bit codebook quantization yields ~1.2 MB. The most extreme published methods — AQLM at 2-bit, QuIP# with E8P lattice codebooks, QTIP with trellis codes — achieve usable quality at 2 bits for LLMs but have not been validated for Jacobian preservation in adversarial pipelines. The LOTION framework (2025) is particularly promising: it trains on the *expectation* of quantized loss under randomized-rounding noise, producing a differentiable objective that preserves all global minima and outperforms STE-based QAT at INT4.

**Quantization is best combined with architectural changes** (SIREN replacement for SegNet, geometric/LUT replacement for PoseNet) rather than used as the sole compression strategy. Where quantization applies — for the compact factors in SVD-decomposed layers or for tiny model weights — codebook quantization with Hessian-guided bit allocation is the method of choice.

---

## 4. Zero-parameter differentiable pose estimation via photometric alignment

The 56 MB PoseNet can be entirely eliminated using geometric methods that require **exactly zero learned parameters**. The challenge is that frames start as flat colored blobs during adversarial optimization — feature-based methods (SIFT + essential matrix, optical flow) require texture that doesn't exist initially.

**Differentiable photometric alignment with depth-aware warping is the recommended zero-parameter approach.** The pipeline works as follows: unproject frame 1 pixels to 3D using the depth buffer (P = K⁻¹·[u,v,1]ᵀ·d(u,v)), transform by the 6-DoF relative pose parameterized as an se(3) Lie algebra element, reproject to frame 2 (P' = R·P + t, [u',v'] = K·P'/P'_z), sample frame 1 at reprojected locations using `F.grid_sample`, and compute multi-scale photometric loss against frame 2. All operations use standard differentiable PyTorch operations. Gradients flow analytically: ∂L/∂pose through the projection chain, ∂L/∂frames through grid_sample.

**This works even with colored blobs** because blob boundaries shift with camera pose changes. The photometric loss gradient through the warp is nonzero at these boundaries. As adversarial optimization progresses and texture emerges, gradient quality improves automatically. The Kornia library provides production-ready implementations of all required geometric primitives: `depth_to_3d`, `transform_points`, `project_points`, `find_essential`, `find_homography_dlt`, and `motion_from_essential_choose_solution`.

For later stages when texture has developed, **BPnP** (CVPR 2020) and **EPro-PnP** (CVPR 2022 Best Student Paper, TPAMI 2024) provide zero-parameter differentiable PnP solver layers with exact Jacobians via the implicit function theorem. EPro-PnP outputs a pose *distribution* on SE(3) rather than a point estimate, making gradients well-defined even at ambiguous configurations. The ∇-RANSAC framework (ICCV 2023) adds differentiable robust estimation with Gumbel-Softmax sampling, now integrated into Kornia.

**Key failure modes to monitor:** SVD gradient instability when singular values are close (in essential matrix decomposition), local minima in photometric alignment without good initialization (use coarse-to-fine multi-scale optimization), and scale ambiguity in essential matrix estimation (resolved if depth buffer is available from the rendering pipeline). The most critical limitation is that **truly flat-colored regions produce zero photometric gradients** — only boundary pixels contribute. Using structural similarity (SSIM) rather than pure MSE as the photometric loss partially mitigates this.

---

## 5. PoseLUT achieves ~126 KB with full differentiability

If the zero-parameter geometric approach proves insufficient for early-stage optimization, a lookup-table architecture offers an extremely compact learned alternative. The design draws on SR-LUT (CVPR 2021), which pioneered converting neural network computation into cached table lookups, and MuLUT (2022–2023), which stacks multiple LUTs with complementary indexing in pseudo-3D architectures.

**The PoseLUT architecture has three components.** First, a tiny CNN feature extractor (~12K params, 48 KB): three Conv2d layers with stride-4, stride-4, stride-2 reducing a 224×224 input to a 32-dim feature vector. Second, a compressed memory of 1000 clustered representative poses (~78 KB): cluster 49K training frames into K=1000 groups via k-means, store 32-dim key vectors (64 KB) and 7-dim pose values in quaternion+translation format (14 KB). Third, differentiable soft-attention retrieval: compute cosine similarity between the query feature and all stored keys, apply softmax with learnable temperature, and compute the weighted sum of stored poses. **Total: ~126 KB.**

**Differentiability is straightforward.** All operations — CNN forward pass, matrix multiply for similarity, softmax, quaternion averaging — are standard PyTorch autograd operations. Gradients flow from pose output through attention weights through similarity scores through the CNN to the input image. For rotation interpolation, normalized linear interpolation (NLERP) of quaternions followed by normalization is differentiable and closely approximates SLERP for angular separations under 45°. The RoMa library from NAVER Labs provides production-quality differentiable `unitquat_slerp` if higher accuracy is needed.

**The NONA framework (2025) validates differentiable nearest-neighbor regression.** It uses SoftStep attention masking as a differentiable proxy for k-NN, imposing implicit pairwise and triplet losses on the embedding space. Similarly, the N3 Block (NeurIPS 2018) interprets k-NN as the limit of parametric categorical distributions with temperature → 0, enabling backpropagation through neighborhood selection. Differentiable Product Quantization (ECCV 2024) directly addresses compact descriptor storage for camera relocalization, achieving **1 MB local descriptor memory** while maintaining localization quality.

A critical design consideration: **quaternion sign ambiguity**. Since q and −q represent the same rotation, all quaternions must be flipped to the same hemisphere (dot product with a reference quaternion > 0) before weighted averaging. Without this, interpolated quaternions can collapse to near-zero magnitude with catastrophic results.

---

## Combining the approaches into a complete pipeline

The five research directions are not alternatives but **complementary layers of a compression stack**. The optimal combined architecture uses SIREN for segmentation (~372 KB) and either photometric pose alignment (0 KB) or PoseLUT (~126 KB) for pose estimation, yielding a total of **372–498 KB** — within the 500 KB target.

For any residual learned weights that require fine-tuning after compression, the Riemannian manifold approach (GeoTorch `low_rank` constraints or LoRA-on-SVD) ensures compressibility is maintained. Where quantization is applied to the compact SIREN or PoseLUT weights, Hessian-aware mixed-precision with codebook centroids preserves gradient quality far better than uniform quantization.

The decisive factor across all five areas is the same: **gradient quality is fundamentally different from forward accuracy**. Methods that look acceptable on accuracy metrics can have destroyed Jacobians (uniform INT4), while methods designed for smooth derivatives (SIREN, codebook quantization, soft-attention retrieval) preserve the gradient landscape that adversarial decode optimization depends on. Every architectural choice in this pipeline should be evaluated on its Jacobian smoothness first and forward accuracy second.