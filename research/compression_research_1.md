# Compressing video for machines, not humans

**The single most impactful finding: a "sandwiched" architecture — learned neural preprocessor before SVT-AV1, task-driven post-processor after decoding — can reduce bitrate by 20–30% at equivalent task accuracy, and open-source code exists today.** This matters because the competition's scoring formula penalizes both archive size and task degradation, so any technique that simultaneously shrinks the file and preserves SegNet/PoseNet features attacks both terms at once. The field driving these advances is Video Coding for Machines (VCM), now under active MPEG standardization (ISO/IEC 23888), with evaluation frameworks specifically designed for exactly this kind of pipeline. What follows is a practical playbook of seven technique families, ordered by estimated impact on pushing the score below 1.95.

---

## 1. Neural preprocessing before AV1 is the highest-leverage intervention

The core idea is simple: insert a small neural network between the raw frames and SVT-AV1 that learns to subtly modify pixels so the resulting bitstream is smaller *and* the decoded frames preserve exactly the features SegNet and PoseNet need. Three published approaches are directly applicable.

**Sandwiched Compression** (Guleryuz et al., Google, 2024) trains a U-Net pre-processor and post-processor end-to-end through a differentiable JPEG proxy. The pre-processor generates "neural code images" — visually natural frames with embedded spatial modulation patterns that survive lossy coding and carry information the post-processor can recover. Despite training only with a JPEG proxy, the learned pre/post-processors **generalize to HEVC, VVC, and AV1** because the neural codes look like natural images all codecs handle well. Reported gains reach **9 dB or 30% bitrate reduction** in transport scenarios. Code is publicly available at `github.com/google/sandwiched_compression` (TensorFlow, with Colab notebooks and video models).

**Preprocessing Enhanced Image Compression for Machine Vision** (Lu et al., IEEE TCSVT 2024) is even more directly relevant. A dual-branch neural preprocessor — 1×1 convolutions for pixel transforms plus a U-Net for semantic features — sits before a standard codec. It uses a learned compression network as a differentiable proxy for the non-differentiable codec and is trained with the downstream task loss (detection, segmentation). A FiLM-style modulation layer makes the single trained model **quantization-adaptive** across CRF values. The result: ~**20% bitrate savings** while maintaining detection and segmentation accuracy, standard-compatible with any codec.

**Perceptual Video Compression with Neural Wrapping** (Khan et al., CVPR 2025, Sony) explicitly targets **SVT-AV1**. A two-phase training aligns a neural codec surrogate to match SVT-AV1's distortion and rate behavior, then trains pre/post-processors through this aligned proxy. It outperforms the state-of-the-art neural codec DCVC-DC across the commercially relevant VMAF range of 40–93.

To adapt any of these to the competition pipeline, the training loss would be:

```
L = λ_seg × CrossEntropy(SegNet(postproc(decode(encode(preproc(x))))), GT_seg)
  + λ_pose × MSE(PoseNet(postproc(decode(encode(preproc(x))))), GT_pose)
  + λ_rate × estimated_bitrate
```

The differentiable proxy handles the `encode → decode` step during training. At inference, the real SVT-AV1 encoder replaces the proxy, and only the lightweight pre-processor runs on each frame before encoding.

---

## 2. Gradient-based importance maps tell you exactly where bits matter

Rather than hand-drawing ROI polygons, the downstream models themselves can reveal precisely which pixels influence the score. **Seg-Grad-CAM** (Vinogradova et al., 2020) adapts Grad-CAM to segmentation by reducing the pixel-wise softmax predictions to a scalar per class and backpropagating through SegNet's encoder to produce per-class spatial importance maps. For PoseNet, standard gradient saliency — computing ∂L_pose/∂x and taking the absolute value summed over channels — highlights regions around body joints and limb boundaries.

The practical pipeline is:

1. Forward-pass each frame through SegNet → backpropagate segmentation loss → importance map I_seg
2. Forward-pass through PoseNet → backpropagate pose MSE → importance map I_pose
3. Combine: I = max(I_seg, I_pose), normalized to [0, 1]
4. Downsample I to SVT-AV1's **64×64 block grid**
5. Map importance to QP offsets (e.g., high-importance blocks get −20, low-importance get +10)
6. Write the per-frame ROI map file and pass to SVT-AV1 via `--roi-map-file`

This is not hypothetical — **SVT-AV1 natively supports per-block QP offset maps** through its `--roi-map-file` flag. The format is a text file where each line contains a frame number followed by QP offsets for every 64×64 superblock in raster order. AV1 uses its segment feature (up to **8 segments per frame**) to apply these offsets via the alternate quantizer syntax. When ROI mode is enabled, it overrides the default variance-based adaptive quantization.

**AccMPEG** (Du et al., MLSys 2022) formalized this idea for video analytics: it learns per-macroblock "accuracy gradients" — how much encoding quality at each spatial location influences DNN accuracy — and uses a cheap model to predict these gradients in near real-time. The system achieved **10–43% reduction in inference delay** without hurting accuracy across six pre-trained detection and segmentation networks. The approach treats each DNN as a black box and works with any standard codec via ffmpeg.

For a driving-corridor video, the combined importance map will typically assign high importance to road surfaces, lane markings, vehicles, pedestrians, and traffic signs — the features both SegNet and PoseNet need — while aggressively compressing sky, distant vegetation, and featureless walls. The expected bitrate saving from this spatial reallocation alone is **15–30%** at equivalent task accuracy, based on the Semantic-Aware Compression literature (Warwick 2022 reported **2.7% mIoU improvement** plus significant bitrate savings using a similar two-level approach).

---

## 3. Task-driven post-processing outperforms generic unsharp masking

The current pipeline's Lanczos upscale + unsharp mask is a blunt instrument. Three categories of replacement offer large gains.

**Task-driven super-resolution (SR4IR, CVPR 2024)** is the most directly relevant technique. It introduces a Task-Driven Perceptual (TDP) loss computed from intermediate features of the downstream task network. When the SR network is trained with TDP loss from a segmentation model, it learns to recover class boundaries and texture patterns that matter for pixel-wise classification. When trained with pose estimation loss, it prioritizes keypoint-region sharpness. A Cross-Quality Patch Mix (CQMix) strategy prevents the task network from learning biased features from the SR output. Tested on EDSR and SwinIR backbones at up to **8× upscaling** for semantic segmentation, detection, and classification. Code is available at `github.com/JaehaKim97/SR4IR`.

**Swin2SR** (Conde et al., ECCV 2022 Workshop) has a dedicated `compressed_sr` mode that jointly removes compression artifacts and super-resolves — addressing both degradations simultaneously rather than sequentially. It was a top-5 solution at the AIM 2022 Compressed Image and Video SR Challenge. With **3.3M+ Replicate runs** and pretrained models on HuggingFace, it is production-ready. Code: `github.com/mv-lab/swin2sr`.

A critical empirical finding from Galteri et al. (ICCV 2017) guides the choice of loss function: **GAN-based restoration improved object detection mAP by 7.4 points** on JPEG-compressed images, while MSE-based and SSIM-based generators improved mAP by only 2.1 and 2.4 points respectively — and sometimes *degraded* detection performance despite improving PSNR. This confirms that perceptual or adversarial losses, not pixel-fidelity losses, are essential when the goal is downstream CV task performance.

For the specific pipeline, the recommended architecture replaces both Lanczos upscaling and unsharp masking with a single lightweight network (modified SwinIR or EDSR) trained with a combined loss:

- **TDP loss from SegNet**: preserves class boundaries and within-class texture
- **TDP loss from PoseNet**: preserves keypoint localization features
- **Adversarial loss**: GAN discriminator prevents over-smoothing
- **L1 reconstruction loss**: provides training stability

For pose estimation specifically, **SRPose** (Wang et al., ACM MM 2023) reformulates heatmap prediction as a super-resolution task, predicting heatmaps at higher spatial resolution than input features to reduce quantization error. It improved AP by **+1.0 to +1.4** on CrowdPose.

---

## 4. Content-adaptive downscaling preserves what uniform scaling destroys

Uniform 45% Lanczos downscaling treats every pixel identically — lane markings get the same treatment as empty sky. Three approaches break this assumption.

**Content-Adaptive Resampler (CAR)** (Sun & Chen, IEEE TIP 2020) learns spatially varying downscaling kernels per pixel, jointly trained with an upscaling network end-to-end. The downscaling kernels adapt to local content: edges and fine structures get sharper kernels that preserve them, while smooth regions get broader kernels that aid compression. CAR-downscaled images are comparably compressible to bicubic-downscaled (similar bpp with JPEG-LS) while producing significantly better super-resolved outputs. Code: `github.com/sunwj/CAR`.

**SRVC** (Khani et al., ICCV 2021) takes a different approach: it encodes low-resolution video with a standard codec plus periodic updates to a **content-adaptive SR network** specialized for short video segments. The SR model is overfit to specific video content, dramatically reducing model complexity while maintaining quality. It outperforms H.265 at high resolution even in slow mode. Code: `github.com/AdaptiveVC/SRVC`.

**AV1's built-in super-resolution mode** offers a simpler alternative. AV1 can internally downscale frames and apply a loop restoration filter during decoding. However, this uses a fixed linear filter, not a learned one — so the CAR approach with an external learned upscaler will yield better task-specific results.

For the competition, the highest-impact approach is to train a CAR downscaler jointly with a task-driven upscaler (SR4IR-style), creating a matched pair that optimizes the downscaled representation for both compressibility and downstream task recovery. The downscaler would learn to preserve lane markings and person contours at the expense of sky detail and road texture.

---

## 5. Bayesian optimization finds the optimal parameter sweet spot

With ~5–10 tunable parameters (downscale percentage, CRF, film-grain level, unsharp/post-processing strength, preset, AQ mode, QP offsets for ROI vs. non-ROI), the search space is too large for grid search but well-suited to Bayesian optimization.

**Optuna with multivariate TPE** is the recommended framework. Disney Research (2024) demonstrated this exact pattern — using BO to search per-title optimal encoding parameters offline, then training ML models to generalize to unseen content. The MainConcept + Optuna integration shows a production-ready template for HEVC parameter tuning with Pareto front visualization.

A practical implementation:

```python
import optuna

def objective(trial):
    downscale = trial.suggest_int("downscale_pct", 30, 65)
    crf = trial.suggest_int("crf", 22, 42)
    film_grain = trial.suggest_int("film_grain", 0, 30)
    unsharp = trial.suggest_float("unsharp_strength", 0.0, 2.5)
    roi_qp_offset = trial.suggest_int("roi_qp_offset", -30, 0)
    nonroi_qp_offset = trial.suggest_int("nonroi_qp_offset", 0, 20)
    
    seg_dist, pose_dist, size_mb = run_pipeline(...)
    return seg_dist * 100 + (10 * pose_dist)**0.5 + 25 * size_mb

sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=15)
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=100)
```

Key configuration details: enable `multivariate=True` to model parameter interactions (CRF and downscale are strongly correlated); set `n_startup_trials=15` for adequate initial exploration with 5+ parameters; expect convergence in **50–150 trials**. For expensive evaluations, **BoTorch** (Meta) offers more sample-efficient Gaussian Process surrogate modeling and supports composite objectives natively via `GenericMCObjective`, where the expensive black-box returns (seg_dist, pose_dist, archive_size) and the known scalar combination is applied analytically.

Additional SVT-AV1 parameters worth including in the search: `--aq-mode` (0/1/2), `--enable-qm` (quantization matrices), `--tune` (0=PSNR, 1=VQ, 2=SSIM), `--enable-cdef` (directional enhancement filter), and if using the PSY fork, `--sharpness` (−7 to 7) and `--variance-boost-strength` (1–4).

---

## 6. Film grain synthesis should be used for denoising but skipped at decode

AV1's film grain synthesis tool was designed for human viewers, not machines. It works by denoising the source at the encoder, compressing the clean frames, transmitting grain parameters as lightweight side information, and re-synthesizing noise at the decoder. Netflix reports **36% average bitrate reduction** at ≥1080p from this mechanism.

For machine vision, the optimal strategy exploits the denoising benefit while avoiding the noise penalty:

- **Enable `film-grain` at moderate levels (4–8)** during encoding with `film-grain-denoise=1` (default). This ensures SVT-AV1 internally encodes the denoised frames, which compress more efficiently and produce cleaner decoded output.
- **Skip grain synthesis at decode time.** Since grain synthesis is a post-decode filter, decoders can omit it. The decoded frames will be cleaner than the originals — beneficial for both SegNet (less noise in class boundaries) and PoseNet (less noise around keypoints).
- **Avoid high film-grain values (>15)** which remove genuine high-frequency detail needed for small-object detection and fine boundary delineation.

This approach attacks the archive_size term (smaller files from denoising) without degrading the seg_dist or pose_dist terms. The current pipeline uses `film-grain=22`, which is likely too aggressive for machine vision — the BO search should explore the 4–12 range to find the optimum.

---

## 7. The VCM ecosystem provides ready-made evaluation infrastructure

The MPEG Video Coding for Machines standardization (active since 2019, now part of ISO/IEC 23888) has produced **CompressAI-Vision** (`github.com/InterDigitalInc/CompressAI-Vision`), an open-source evaluation platform adopted as the official MPEG common evaluation framework. It supports object detection (Faster-RCNN, YOLO), instance segmentation (Mask-RCNN), **pose estimation** (via MMPose), and multi-object tracking. It integrates with standard codecs (H.264, H.265, VVC) and learned codecs from the CompressAI library, and provides BD-rate calculation, configurable YAML pipelines, and visualization tools.

The VCM standardization operates on two tracks. **Track 1 (pixel-domain)** applies task-aware preprocessing before a standard codec — directly analogous to the competition pipeline. **Track 2 (feature coding)** compresses intermediate neural features rather than pixels, achieving **79% bitrate savings** over pixel-domain coding on some datasets. While feature coding requires a custom decoder, it demonstrates the theoretical ceiling for machine-oriented compression.

Key VCM tools relevant to this pipeline include spatial resampling, ROI-based retargeting (3–57% bitrate reduction), temporal resampling, background simplification (converting non-ROI areas to grayscale), and bit-depth truncation. The **Compression Distortion Representation Embedding (CDRE, 2025)** technique sends a lightweight binary side-channel describing machine-perception-related distortions, enabling decoder-side compensation that achieved BD-rate savings of **9.83% for keypoint/pose detection** and **14.72% for instance segmentation** with fixed downstream models.

---

## Conclusion: a concrete roadmap to sub-1.95

The techniques above are not merely theoretical — they compose into a concrete upgraded pipeline. The three highest-impact changes, ordered by implementation difficulty:

1. **Replace hand-drawn ROI with gradient-saliency maps fed to SVT-AV1's `--roi-map-file`** (days of work, no training required). Run SegNet and PoseNet backward passes on each frame, combine importance maps, quantize to 64×64 block QP offsets. This alone should reduce archive size by 15–20% at equivalent task accuracy.

2. **Replace Lanczos + unsharp mask with a task-driven SR network** (1–2 weeks, uses SR4IR codebase). Fine-tune Swin2SR or EDSR with TDP loss from both SegNet and PoseNet. The GAN-based restoration literature predicts **3–7 mAP points** of improvement over generic upscaling.

3. **Train a sandwiched pre-processor using the Google codebase** (2–3 weeks, uses `sandwiched_compression` repo). The JPEG proxy trains in hours and transfers to AV1 without retraining. Add SegNet/PoseNet task losses to the training objective for task-aware optimization. Expected: 20–30% further bitrate reduction or equivalent task-accuracy gain.

Wrapping all tunable parameters — including ROI QP offsets, film-grain level, CRF, downscale factor, and post-processing strength — in an **Optuna multivariate TPE loop** for 100+ trials will find the global optimum across this expanded design space. The combination of task-aware spatial bit allocation, learned pre/post-processing, and systematic parameter optimization should push the composite score well below the current 1.95 threshold.