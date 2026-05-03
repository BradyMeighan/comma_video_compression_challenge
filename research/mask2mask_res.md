# Pushing mask2mask below 0.50: ten technical levers

The most promising paths to cut the score from 0.60 to below 0.50 are not evenly distributed across the ten questions. **Three interventions stand out as highest-impact**: switching to 3-bit quantization with a larger architecture (~50% more effective memorization capacity at the same byte budget), exploiting dashcam geometry with a ~20-parameter analytic flow model (freeing thousands of parameters currently spent on flow prediction), and pre-training on Cityscapes before fine-tuning on the 600 pairs (better initialization for a capacity-starved network). The remaining seven levers offer incremental or situational gains. This report synthesizes the latest research on each question and provides concrete, actionable recommendations.

---

## 1. Halving mask resolution saves ~40–60% with manageable quality loss

Dropping from 384×512 to 192×256 does not produce a clean 4× bitrate reduction in AV1. Because the dominant coding cost for 5-class maps is boundary perimeter (which scales linearly with resolution, not quadratically), the expected savings are **~40–60%**, bringing the mask video from ~209 KB down to an estimated **80–120 KB**. This is still significant—potentially freeing 90–130 KB for a larger generator.

The generator can compensate. The CVPR 2020 paper "Dual Super-Resolution Learning for Semantic Segmentation" (Wang et al.) shows that operating on half-resolution inputs and upscaling recovers most segmentation accuracy. "Label Super-Resolution Networks" (ICLR 2019) demonstrates learned label-map upsampling that approaches high-resolution ground truth quality when guided by visual cues. Since the generator already maps discrete masks to continuous RGB, it implicitly performs learned upsampling—retraining on lower-resolution masks simply asks it to do more spatial interpolation, which convolutional networks handle naturally.

**Nearest-neighbor is mandatory** for any explicit mask upsampling step, since bilinear interpolation creates meaningless fractional class values. Quality loss concentrates at boundaries: at half resolution, each low-res pixel covers a 2×2 block, potentially misclassifying **2–5% of boundary pixels**. For the SegNet evaluation, this boundary degradation would increase the error metric—but if the freed kilobytes enable a sufficiently better generator, the net effect could be positive. A 256×340 intermediate resolution (preserving aspect ratio at ~2/3 scale) may offer a better trade-off than a full 2× downscale.

---

## 2. AV1 is already near-optimal for this data

At **0.014 bits per pixel** averaged across 600 frames, AV1 operates remarkably close to the conditional entropy floor for temporally coherent 5-class maps. Its palette mode (designed for 2–8 color blocks), screen-content coding tools, and motion-compensated temporal prediction are almost purpose-built for this data type.

The strongest specialized alternatives are **CC-SMC** (Yang et al., JVCIR 2024), which uses quadtree partitioning plus context-based chain coding with interframe temporal coding, achieving >10% bitrate savings over HEVC-SCC; and **RECC** (InterDigital, 2025), which extends Freeman chain codes to 36 symbols across 3 layers with skip coding for shared boundaries. Both have public code. However, these approaches must replicate AV1's sophisticated motion compensation to exploit temporal redundancy—the dominant compression lever for a 600-frame driving sequence where most segmentation pixels are identical frame-to-frame.

Per-frame image codecs lose badly: PNG would yield ~1.2–3 MB total, JPEG-XL lossless ~0.8–2 MB—roughly **6–15× worse than AV1** because they cannot exploit temporal redundancy. A custom delta-coding scheme (XOR consecutive frames → chain-code the sparse differences → arithmetic encode) might match AV1 but is unlikely to beat it by more than 5–20%. The realistic improvement ceiling is **~170–200 KB**, saving at most ~40 KB. Engineering effort is high for modest gain.

A **learned autoencoder for masks is impractical** here. Even a minimal encoder-decoder adds 50–400 KB of parameters, consuming a large fraction of (or exceeding) the current 209 KB AV1 total. Neural codecs excel at lossy compression of continuous signals; for lossless categorical data, no published work demonstrates them beating traditional codecs. The model-size overhead makes this a dead end at current budget levels.

---

## 3. Three-bit quantization hits the memorization sweet spot

The landmark ParetoQ paper (Meta, February 2025) demonstrates that **2-bit and ternary (1.58-bit) models sit on the Pareto frontier** for the accuracy-vs-size trade-off, outperforming 4-bit quantization when the freed bits are reinvested in more parameters. A 2-bit MobileLLM-1B achieves 1.8 accuracy points higher than a 4-bit MobileLLM-600M at smaller total size. However, ParetoQ also reveals a **critical learning transition between 2 and 3 bits**: below 3-bit, models must "reconstruct" rather than "compensate" from the pretrained distribution, requiring ~3× more training effort.

For the specific task of memorizing 600 mask→RGB pairs, precision matters more than for generalization. Morris et al. (2025) measured practical memorization capacity at **~3.6 bits per parameter** for language models in bf16—meaning each parameter can encode roughly 3.6 bits of task-specific information through SGD optimization. At 4-bit quantization, the 3.6-bit capacity is the bottleneck (4 available bits, only 3.6 used). At 3-bit, the bit-width itself becomes the bottleneck (3 available < 3.6 usable). At 2-bit, only 2 bits per parameter survive—a severe constraint for dense RGB prediction where subtle color/texture variations matter.

**The practical recommendation is 3-bit quantization with a custom 8-value codebook** (analogous to NF4 but with 8 optimally placed values for the weight distribution). This allows ~560K parameters in the same ~210 KB budget, providing **~50% more effective memorization capacity** than the current 308K at 4-bit. Mixed-precision is an even safer path: keep the first convolution, last convolution, and skip connections at 4-bit (most sensitive layers) while quantizing intermediate layers to 3-bit.

Ternary (2-bit) and binary (1-bit) are **high-risk for this task**. TTQ (Trained Ternary Quantization) can match or exceed full-precision for classification, and BiDense shows binary networks approaching full-precision for dense prediction—but these results are for generalization, not memorization. With only 3–4 distinct weight values, encoding the precise texture patterns of 600 specific driving scenes is extremely challenging. The QAT literature explicitly warns that "error accumulation in intermediate layers is particularly severe for small datasets."

---

## 4. Why ~500K parameters at 3-bit likely beats 308K at 4-bit

Zhang et al.'s foundational ICLR 2017 result shows that networks memorize entire datasets when parameters exceed data points—but the relevant measure is not 600 samples, it is the **total output information**: 600 pairs × 2 frames × 384×512×3 bytes ≈ 706 MB of RGB data. The network must compress this into ~170 KB of weights, a compression ratio of ~4,000:1. Every additional effective bit of memorization capacity counts.

The bits-per-parameter analysis at a fixed ~210 KB budget:

- **308K params × 4-bit**: ~1,108K effective bits (3.6 × 308K, capped by practical capacity)
- **410K params × 3-bit**: ~1,230K effective bits (3.0 × 410K, capped by bit-width)
- **560K params × 3-bit**: ~1,680K effective bits (at 250 KB budget)
- **616K params × 2-bit**: ~1,232K effective bits (2.0 × 616K, capped by bit-width)

The 3-bit configuration at comparable or slightly larger budget delivers the most effective capacity. **Width (more channels) should be prioritized over depth** when scaling up, as wider networks have more parallel capacity for memorization. A U-Net-like architecture with 500–600K parameters at 3-bit (~225 KB) should significantly outperform the current design.

Friedland et al. (LLNL 2018) provide a complementary engineering rule: total network capacity is limited by bottleneck layers. Widening the narrowest layer of the current architecture may yield disproportionate improvement.

---

## 5. Pre-training on driving data helps even for pure memorization

Strong evidence supports pre-training before fine-tuning on the 600 fixed pairs. Hendrycks et al. (ICML 2019) showed pre-training reduces error on corrupted/small datasets by **38% relative** (area-under-error-curves of 14.8% vs 23.7%). The lottery ticket hypothesis (Frankle & Carlin, ICLR 2019) further demonstrates that initialization quality determines whether small networks can learn effectively—pre-trained subnetworks at 70% sparsity transfer universally to downstream tasks.

For a capacity-starved 308K-parameter model, pre-trained features provide texture and boundary priors that the network would otherwise spend capacity "discovering." Road surface textures, sky gradients, building facades, and vegetation patterns are shared across all driving datasets. **Pre-training on Cityscapes** (5,000 finely annotated images, same domain) with a SPADE-style spatially-adaptive normalization architecture, then fine-tuning aggressively on the 600 pairs, is the recommended approach.

**SPADE** (Park et al., CVPR 2019) is particularly relevant: it removes the encoder entirely, using a decoder-only architecture where segmentation masks modulate activations via learned spatially-varying affine parameters (γ, β) at every normalization layer. With 1×1 convolutions instead of 3×3, each SPADE layer adds only **~1–3K parameters**—feasible for the budget. **OASIS** (Schönfeld et al., ICLR 2021) improves on SPADE by 6 FID points and 5 mIoU by using a segmentation-network discriminator during training, which provides spatially and semantically aware feedback that improves boundary alignment.

Knowledge distillation from a larger model is another viable path. **TinyGAN** (Chang & Lu, ACCV 2020) successfully distills BigGAN to 16× fewer parameters. A practical pipeline: train a 5–10M parameter SPADE/OASIS model on Cityscapes, distill to the 308K target, then fine-tune on the 600 pairs. MobileStyleGAN's use of **wavelet-based representation** (predicting DWT coefficients instead of raw pixels) is worth exploring—it enables higher-quality generation from smaller feature maps.

---

## 6. A single attention layer at the bottleneck costs 5% and helps boundaries

The parameter cost of self-attention scales as ~4C² where C is the channel dimension. At the bottleneck of a typical tiny U-Net (8×8 or 16×16 spatial resolution, 64 channels), a single self-attention layer costs **~16K parameters—5.3% of the 308K budget**. At this resolution, the token count (64–256) makes quadratic attention computation trivial.

The benefit is global context that convolutions cannot provide: a sky pixel can attend to all road pixels to ensure consistent color temperature; building facades can maintain coherent texture across the full image width. Research on attention for semantic image synthesis confirms the value: OA-GAN uses attention-driven multi-fusion to improve fine-grained boundaries, and DCSIS reports **2.1–2.7 mIoU improvement** over OASIS by adding dual-conditioning attention paths.

For even lower overhead, **Squeeze-and-Excitation blocks** add only ~500–2,000 parameters (0.2–0.7% of budget) for channel attention, and **CBAM** adds spatial attention for ~700–3,000 parameters. These don't provide true global spatial reasoning but offer channel reweighting that helps the network prioritize boundary-relevant features. **Linear attention** (Katharopoulos et al., ICML 2020) reduces computation from O(N²) to O(N) at the same parameter count, enabling attention at higher spatial resolutions (32×32) if needed.

The recommended configuration: 1–2 SE blocks at intermediate layers (~1–2K params) plus one self-attention layer at the bottleneck (~16K params), totaling **~18K parameters (5.8% of budget)** for meaningful boundary quality improvement.

---

## 7. Boundary-only encoding is theoretically sound but AV1 is hard to beat

Storing only class boundaries and reconstructing full masks is a well-studied approach. **CC-SMC** (Yang et al., 2024) uses quadtree partitioning plus context-based chain coding with interframe temporal coding, achieving >10% savings over HEVC-SCC. **RECC** (InterDigital, 2025) extends to 36-symbol chain codes with skip coding for shared boundaries—the current state-of-the-art for lossless segmentation map compression. Both have public GitHub implementations.

Reconstruction from boundaries is **perfectly lossless**: chain codes define exact boundary pixels, and standard region-filling algorithms (scanline fill, watershed from labeled seeds) recover every interior pixel exactly. The **Three-Orthogonal Chain Code** (Bribiesca 1999) uses only 2 bits per boundary step, ~25% more efficient than Freeman's 3-bit chain code.

The challenge is that AV1's temporal prediction already achieves **~349 bytes per frame** for 600 frames. A boundary-only approach without temporal compression would require ~0.8–1.9 KB per frame (depending on boundary complexity), yielding 480–1,140 KB total—far worse. Only with sophisticated interframe chain-code delta coding (encoding boundary motion between frames) could boundary encoding approach AV1's efficiency. CC-SMC's interframe mode is the closest existing implementation, but whether it can beat a well-tuned AV1 encode of the full mask is uncertain and requires empirical testing.

---

## 8. The analytic flow model is the highest-leverage architectural change

The current design warps frame2 to produce frame1, requiring a flow prediction network that consumes parameters. For dashcam video with primarily forward translation, the entire optical flow field can be derived analytically from **~15–21 parameters**:

- **Focus of Expansion**: 2 parameters (FOE_x, FOE_y)—the vanishing point of the forward motion
- **Camera rotation**: 3 parameters (ωx, ωy, ωz)—yaw from steering, pitch from bumps
- **Translation magnitude**: 1 parameter
- **Per-class depth**: ~10–15 values (sky=∞, road=ground plane geometry, buildings=medium)

The flow at each pixel is computed analytically: the translational component radiates from the FOE with magnitude inversely proportional to depth, and the rotational component depends only on pixel position and rotation parameters. This is a **>10,000× reduction** compared to dense flow prediction (2 × 384 × 512 = 393K values) or a typical small flow network (tens of thousands of parameters).

Seg2Depth (IEEE 2024) validates that semantic labels combined with vanishing point geometry approximate depth well for driving scenes—no depth ground truth needed. The **Plane+Parallax framework** (Irani & Anandan, 1998) provides additional theoretical grounding: aligning two frames via a ground-plane homography removes rotation, reducing all remaining motion to 1D parallax from the epipole. Deep Planar Parallax (ICLR 2023) successfully applies this to neural depth estimation for driving.

If per-class depth is extended to a linear function of vertical position (base depth + slope per class), ~25–35 total parameters capture the full motion model. Road depth increases linearly with vertical position from the horizon; building depth depends on horizontal position. These parameters can be stored per frame pair at negligible cost (~40–70 bytes per pair × 300 pairs ≈ 12–21 KB), or predicted by a tiny head from the mask. Either way, this frees thousands of parameters currently used for flow prediction, redirecting them to improve RGB generation quality.

---

## 9. Shared encoder with feature warping saves 30–50% of parameters

The Video LDM architecture (Blattmann et al., 2023) establishes the dominant paradigm: a **100% shared spatial backbone** processes all frames identically, with only lightweight temporal layers (~10–20% additional parameters) handling frame-specific dynamics. Fast-Vid2Vid (Zhuo et al., ECCV 2022) achieves **8× computational savings** by generating only keyframes and motion-compensating the rest. Both validate that shared computation across frames is dramatically more parameter-efficient than independent per-frame generation.

**Feature warping outperforms image warping.** MoG (2025) explicitly compares the two: feature-level injection provides better subject consistency, while latent-level injection provides better background consistency—combining both levels yields the best results with "no additional parameters" when reusing existing temporal layers. For the compression challenge, warping intermediate features (after the shared encoder, before per-frame RGB heads) preserves both low-level textures and high-level semantics better than warping final RGB pixels, where information is already "baked in" and warping artifacts are more visible.

The recommended architecture: a shared encoder processes both masks through a common feature backbone, a tiny analytical flow module (see Question 10) provides the warp field, features are warped at an intermediate resolution, and lightweight per-frame refinement heads (~10–15% of total parameters each) produce final RGB. This should enable **30–50% parameter savings** versus independent generation, or equivalently, a 30–50% larger shared backbone at the same total parameter count.

---

## Concrete roadmap to push below 0.50

The ten levers are not independent—they interact synergistically. The highest-impact combination is:

1. **Switch to 3-bit quantization** with a custom 8-value codebook, scaling the architecture to ~500–560K parameters (~210–225 KB). This alone provides ~50% more effective memorization capacity.

2. **Replace the learned flow network with an analytic 20-parameter motion model**, freeing those parameters for RGB generation quality. Per-class depth derived from segmentation labels plus FOE and rotation parameters fully specify dashcam optical flow.

3. **Pre-train on Cityscapes** with SPADE-style spatially-adaptive normalization, then fine-tune on the 600 pairs. This gives the capacity-starved model better texture/boundary initialization.

4. **Add a single bottleneck attention layer** (~16K params) for global context that improves boundary coherence—the metric where SegNet errors concentrate.

5. **Adopt shared-encoder architecture** with feature warping, reinvesting the parameter savings into a wider backbone.

6. **Optionally reduce mask resolution** to ~256×340, saving ~60–80 KB from the mask video to allow a slightly larger model budget, if the generator can compensate.

The lower-impact levers—beating AV1 with custom codecs (marginal gains), boundary-only encoding (hard to beat temporal AV1), and learned mask codecs (impractical at this budget)—should be deprioritized unless the higher-impact changes have already been exhausted. The combination of 3-bit quantization, analytic flow, pre-training, attention, and shared features represents a coherent architectural redesign that addresses the fundamental bottleneck: squeezing more generation quality out of every available bit of model storage.