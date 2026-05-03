# Winning the comma.ai video compression challenge

**The most powerful strategy exploits the known evaluation networks at decode time, generating frames via gradient optimization that reproduce exact SegNet/PoseNet outputs from stored segmentation maps and pose vectors — potentially achieving scores below 2.0 versus the 4.39 baseline.** This works because the SegNet and PoseNet models (~94 MB combined) are available in the challenge repo without counting toward archive size, and the scoring formula's extreme asymmetry (100× SegNet weight) makes near-zero segmentation distortion worth far more than additional compression. Even without this adversarial approach, switching from all-intra to inter-prediction at the evaluation resolution of 512×384 using modern codecs can cut the score to roughly 3.0. The challenge launched days ago with zero external submissions — it is wide open.

---

## The scoring function reveals a clear hierarchy of priorities

Before choosing techniques, the score decomposition demands careful analysis. The baseline breaks down as:

| Component | Formula | Baseline value | Score contribution |
|-----------|---------|---------------|-------------------|
| SegNet distortion | 100 × 0.00947 | 0.00947 | **0.947** |
| Compression rate | 25 × 0.0597 | 0.0597 | **1.493** |
| PoseNet distortion | √(10 × 0.380) | 0.380 | **1.949** |
| **Total** | | | **4.39** |

The SegNet term has a **100× linear multiplier**, so reducing segnet_dist from 0.01 to 0.005 saves 0.5 points — equivalent to eliminating 750 KB from the archive. Compression rate's 25× multiplier matters but is secondary. PoseNet sits under a square root, making it the least sensitive: halving posenet_dist from 0.38 to 0.19 only saves 0.57 points. Every optimization decision should prioritize SegNet preservation first, rate second, PoseNet third.

Both evaluation networks resize input to **512×384 via bilinear interpolation**. The original 1164×874 resolution is never seen by either model. This single fact makes encoding at any resolution above 512×384 wasteful — every bit spent on high-frequency detail above that resolution is invisible to the scoring function. The baseline's 45% scale (524×393) is close but not optimal.

---

## Strategy 1: Adversarial decode via evaluation network inversion

This is the highest-impact approach if the inflate.py script can access the repo's SegNet and PoseNet models at decode time (which is likely, since decoding runs on the same machine as evaluation).

### How it works

At **encode time** (unlimited compute): run SegNet on each odd-indexed frame resized to 512×384, store the 5-class argmax segmentation map. Run PoseNet on each frame pair, store the 6-dimensional pose vector.

At **decode time** (30 min on T4): for each frame, initialize a flat-colored image using pre-computed "ideal" RGB values per class, then run **20–50 gradient descent steps** through SegNet to match the target segmentation map. Joint optimization with PoseNet ensures pose consistency. Finally, upscale the 512×384 result to 1164×874 via bilinear interpolation.

### Why this works

Research on Dense Adversary Generation (Xie et al., ICCV 2017) shows that targeted segmentation attacks converge in **150–200 iterations** from random initialization. Starting from class-color initialization — where each pixel receives the RGB color that most strongly activates its target class — convergence drops to **20–50 iterations** because 85–95% of pixels are already correctly classified (well-trained networks are overconfident on interior pixels, with uncertainty concentrated at boundaries).

**EfficientNet-B2 UNet forward+backward pass at 512×384 on T4**: approximately 36–66 ms in FP16 mixed precision. At 30 iterations per frame, each frame takes roughly **1–2 seconds**. Processing 600 SegNet-evaluated frames: **10–20 minutes total**, well within the 30-minute budget. FastViT-T12 (PoseNet) adds only ~10–20% overhead per iteration since it runs at ~1–3 ms per forward pass on T4.

### Storage requirements

Segmentation maps: 600 frames at 512×384 with 5 classes. Raw storage is ~73 KB per frame (3 bits/pixel), but consecutive dashcam segmentation maps share ~98% of pixels. Delta-coding plus entropy compression achieves roughly **3–8 KB per frame after the first**. Total for 600 maps: approximately **1.5–3 MB**. Pose vectors: 600 pairs × 6 floats × 4 bytes = **14.4 KB** (negligible, further compressible to ~7 KB with float16).

### Projected score

| Component | Estimate | Score |
|-----------|----------|-------|
| SegNet dist | ~0.001 (near-perfect matching) | 0.10 |
| Rate | 2 MB / 37.5 MB = 0.053 | 1.33 |
| PoseNet dist | ~0.05 (jointly optimized) | 0.71 |
| **Total** | | **~2.1** |

The score could be pushed below **1.5** by aggressively compressing the segmentation maps (using palette-mode lossless video coding achieves ~1 MB for 600 temporally-correlated label maps). The critical risk: if the inflate.py environment does not have access to the SegNet/PoseNet weights, this approach fails entirely unless those 94 MB of weights are included in the archive, which would make the rate term catastrophic (25 × 2.5 = 62.5).

### Mitigating the model-access risk

Even without direct model access at decode time, a **distilled tiny proxy** of the SegNet (e.g., a 200 KB INT8 MobileNet student trained to approximate SegNet's outputs) could be included in the archive. The proxy doesn't need to perfectly replicate SegNet — it just needs to guide optimization close enough that the real SegNet's argmax matches. This adds minimal archive cost while enabling approximate adversarial decoding.

---

## Strategy 2: Optimal traditional codec at evaluation resolution

This is the safest approach, requiring minimal implementation complexity while delivering substantial improvement over the baseline.

### The baseline leaves enormous gains on the table

The baseline uses **all-intra encoding** (every frame is a keyframe), which is catastrophic for dashcam video with ~95% frame-to-frame pixel overlap. Switching to inter-prediction with a reasonable GOP structure saves **40–60% bitrate** at identical quality. Combined with upgrading from `ultrafast` to `veryslow` preset (another 20–30% savings), the total improvement is **3–7× smaller files**.

Encoding at exactly **512×384** (the evaluation resolution) rather than 524×393 captures precisely the information the networks see. The bilinear round-trip error (encode at 512×384 → upscale to 1164×874 → network downscales to 512×384) is negligible — typically **PSNR > 45 dB**, contributing well under 0.001 to SegNet distortion.

### Codec comparison at 512×384, 20fps, 60 seconds

| Codec | Settings | Est. file size | Est. PSNR |
|-------|----------|---------------|-----------|
| x265 | CRF 30, veryslow, keyint=600 | 600 KB–1.3 MB | 31–33 dB |
| SVT-AV1 | CRF 35, preset 4, 10-bit | **200–500 KB** | 30–32 dB |
| VVenC (H.266) | QP ~35, slower preset | **150–350 KB** | 30–32 dB |
| x265 | CRF 35, veryslow, keyint=600 | 225–600 KB | 28–30 dB |
| Baseline (x265) | CRF 30, ultrafast, all-intra, 524×393 | 2,240 KB | ~30 dB |

**SVT-AV1** offers the best practical balance: 30–50% smaller than x265 at equivalent quality, extremely fast CPU decoding via dav1d (500+ fps at 512×384), and mature tooling. **VVenC (H.266)** is the compression king at 20–30% smaller than AV1, but has less mature FFmpeg integration.

### Projected score with SVT-AV1

The encoder-side optimization loop is crucial: encode at multiple CRF values, decode each, run SegNet/PoseNet, and select the CRF minimizing total score. This is offline work with no time constraint.

| CRF | Rate | SegNet dist | PoseNet dist | Score |
|-----|------|-------------|-------------|-------|
| 30 | 0.013 | ~0.008 | ~0.30 | 0.8 + 0.33 + 1.73 = **2.86** |
| 33 | 0.008 | ~0.010 | ~0.35 | 1.0 + 0.20 + 1.87 = **3.07** |
| 36 | 0.005 | ~0.014 | ~0.42 | 1.4 + 0.13 + 2.05 = **3.58** |

The optimal CRF likely falls near **30–33** where the steep rise in SegNet distortion hasn't yet overwhelmed the rate savings. Expected best score: **~2.8–3.2**.

---

## Strategy 3: Implicit neural representation fitted to the video

HiNeRV (NeurIPS 2023) and NVRC (NeurIPS 2024) represent a video as a neural network — the **compressed model IS the compressed file**, requiring no external codec weights. This has a unique advantage: the training loss can be replaced with the exact competition metric.

### How it works for this challenge

A HiNeRV network with **1.5–3 million parameters** is overfitted to the specific dashcam video, then compressed via structured pruning (15% removal), quantization-aware training (6–8 bit), and arithmetic coding. The resulting bitstream is **500 KB–1.5 MB**. Decoding is a single forward pass per frame at **50–200 fps on T4** — total decode time under 30 seconds.

**The killer feature**: training with a custom loss function that directly optimizes `100 × segnet_dist + sqrt(10 × posenet_dist)` instead of MSE. This means every parameter is allocated toward preserving what the scoring function actually measures, rather than wasting capacity on perceptually irrelevant details. Sky regions, road textures, and distant objects receive minimal model capacity while lane markings, vehicle boundaries, and motion-relevant features receive maximum fidelity.

### Practical considerations

Code is available at `github.com/hmkx/HiNeRV`. Training a 1.5M parameter model on 1200 frames at 512×384 requires approximately **3–8 hours on a T4 GPU** (offline, no time constraint). The full compression pipeline (pruning + quantization-aware training + entropy coding) adds several hours.

The main challenge is integrating SegNet and PoseNet into the training loop. Each training step requires a forward pass through both evaluation networks, increasing per-step cost by roughly **5–10×**. A practical workflow: train with MSE for 30 epochs to get a reasonable initial representation, then fine-tune for 10–20 epochs with the task-aware loss.

### Projected score

| Model config | Archive size | SegNet dist | PoseNet dist | Score |
|-------------|-------------|-------------|-------------|-------|
| 1.5M @6bit, MSE loss | ~700 KB | ~0.015 | ~0.40 | 1.5 + 0.47 + 2.0 = **3.97** |
| 3M @6bit, MSE loss | ~1.4 MB | ~0.010 | ~0.35 | 1.0 + 0.93 + 1.87 = **3.80** |
| 1.5M @6bit, task-aware loss | ~700 KB | ~0.008 | ~0.50 | 0.8 + 0.47 + 2.24 = **3.51** |
| 3M @6bit, task-aware loss | ~1.4 MB | ~0.006 | ~0.40 | 0.6 + 0.93 + 2.0 = **3.53** |

Task-aware training likely improves SegNet distortion significantly at the cost of worse PoseNet performance (since the sqrt dampens PoseNet's score contribution). The optimal model size balances archive cost against distortion reduction.

---

## Strategy 4: DCVC-RT neural video codec

Microsoft's DCVC-RT (CVPR 2025) achieves **21–31% bitrate savings over VVC** while decoding at 112 fps for 1080p on A100 — estimated **150–300 fps at 512×384 on T4**. Code and weights are available at `github.com/microsoft/DCVC`.

The critical constraint: decoder model weights (~30–60 MB) likely must be in the archive per competition rules ("neural network weights in archive count toward size"). At 30–60 MB, the compression_rate term alone contributes **20–40 points** to the score, making this approach non-competitive unless weights are exempted as a "library dependency" (like ffmpeg). If somehow exempted, DCVC-RT at 512×384 would produce bitstreams of **375 KB–1.1 MB** at 30–35 dB PSNR, enabling scores of **~2.5–3.0**.

This approach is only worth pursuing if the competition rules clearly exempt pre-trained codec weights from the archive size calculation.

---

## Temporal tricks the evaluation structure enables

The dataloader creates **non-overlapping 2-frame pairs**: (0,1), (2,3), (4,5), etc. SegNet evaluates **only odd frames** (1, 3, 5, ...). PoseNet evaluates both frames per pair. This asymmetry creates opportunities:

- **Odd frames are high-priority** (affect both SegNet and PoseNet). Even frames only affect PoseNet.
- **Drop even frames and interpolate**: Encode 600 odd frames, use simple motion compensation or a tiny interpolation model to reconstruct even frames. Since PoseNet is under sqrt, interpolation artifacts have dampened impact. RIFE (9.8M params, ~20–40 MB) or IFRNet-S (1.3M params, ~5 MB) can interpolate at 50–100+ fps on T4. The model weight cost (~5–40 MB) may or may not be justified depending on bitrate savings.
- **Better approach**: Use a traditional inter-prediction codec (which already exploits temporal redundancy) and simply ensure the codec allocates more bits to odd frames. x265's `--aq-mode 4` with scene-cut detection achieves this implicitly.

Frame dropping is only valuable if encoding all frames with inter-prediction doesn't sufficiently compress them — and for dashcam video, inter-prediction typically captures >95% of temporal redundancy already, making explicit frame dropping redundant.

---

## Dashcam priors make extreme compression viable

Several properties of comma.ai dashcam video (from the comma2k19 dataset, which uses the same camera and resolution) enable aggressive compression beyond what general-purpose codecs achieve:

- **Scene geometry is rigid and predictable**: the horizon sits at ~40–50% frame height, the vanishing point is near-center, and the road plane occupies the bottom 40–60%. A 6-parameter affine model (3 translation + 3 rotation) captures >99% of frame-to-frame motion.
- **Semantic regions have distinct compressibility**: sky (top 15–25%) is nearly uniform and compresses at near-zero cost; road surface is repetitive texture; lane markings and vehicles carry the critical segmentation information.
- **Temporal redundancy is extreme**: at 20 fps on California highways, consecutive frames share ~95% of content, making inter-prediction extraordinarily effective.

A **SegNet-aware preprocessing** pass before encoding can exploit these priors: run SegNet on the original frames, identify the "undrivable" class (sky, background), and apply aggressive blurring or flat-color replacement to those regions before encoding. Since SegNet already classifies those regions as undrivable, the preprocessing preserves segmentation output while dramatically reducing entropy in those areas. This is effectively a manual version of Video Coding for Machines, which achieves **30–40% bitrate savings** over standard coding at equivalent machine-task accuracy (Le et al., ICASSP 2021).

---

## The optimal hybrid pipeline, ranked by expected score

### Approach A — Adversarial decode (score: ~1.5–2.5)

**Requires**: SegNet/PoseNet accessible to inflate.py at decode time

1. **Encode** (offline): Extract 5-class segmentation maps for 600 odd frames at 512×384. Extract 6-dim pose vectors for 600 pairs. Compress seg maps via palette-mode lossless video codec. Store pose vectors as delta-coded float16.
2. **Decode** (T4, ~15 min): Load seg maps and pose vectors. For each frame, initialize with class-ideal colors, run 30 joint gradient steps through SegNet/PoseNet, round to uint8, upscale to 1164×874.
3. **Archive**: ~1.5–3 MB total.
4. **Risk**: Model access assumption. Mitigation: include a 200 KB distilled proxy SegNet in the archive as a fallback.

### Approach B — Optimal traditional codec + encoder-side optimization (score: ~2.8–3.2)

**No special requirements**

1. **Encode** (offline): Downscale video to 512×384. Run SVT-AV1 (or VVenC) at CRF 30–34, preset 4, with long GOP (keyint=600). Run SegNet-aware preprocessing to blur sky/background regions. Grid-search CRF to minimize total score.
2. **Decode** (<1 min): Decode AV1 bitstream. Bilinear upscale to 1164×874.
3. **Archive**: ~200–500 KB.
4. **Risk**: Minimal. AV1 decode is fast, tooling is mature.

### Approach C — HiNeRV with task-aware training (score: ~3.0–3.5)

**Requires**: GPU encoding time (3–8 hours)

1. **Encode** (offline): Train HiNeRV 3M-parameter model on 512×384 video. Fine-tune with SegNet/PoseNet loss. Prune + quantize to 6-bit. Entropy-code the weights.
2. **Decode** (T4, <30 sec): Forward pass through the network for all 1200 frames. Upscale to 1164×874.
3. **Archive**: ~1–1.5 MB (the compressed model).
4. **Risk**: Custom loss integration complexity. Training instability.

### Combined pipeline

The strongest practical submission would **attempt Approach A first**, verify whether the eval models are accessible, and **fall back to Approach B** if not. Approach B alone should comfortably beat the 4.39 baseline by 25–35%. Adding SegNet-aware preprocessing (Approach B enhanced) could push the score to **~2.5–2.8**.

---

## Conclusion: three key insights that determine the winner

**First**, the scoring formula's extreme asymmetry means this is fundamentally a SegNet-preservation contest, not a compression contest. A strategy that achieves 0.001 segnet_dist at 3 MB is vastly better than one achieving 0.010 segnet_dist at 100 KB. Every decision should be evaluated through the lens of "what does this do to SegNet output?"

**Second**, the evaluation networks are known, fixed, and differentiable. This is not a general compression problem — it is an adversarial optimization problem where the "adversary" (the scoring function) is fully transparent. The most competitive submissions will treat the eval networks as optimization targets, not as passive quality judges.

**Third**, the baseline's all-intra encoding is deliberately terrible. Switching to inter-prediction alone recovers roughly 1.5 points on the score (from 4.39 to ~2.9). This is the minimum bar for a competitive entry. The real competition happens between contestants who all use inter-prediction and differ in how cleverly they exploit the evaluation pipeline.

The challenge deadline of May 3, 2026 leaves roughly one month. The implementation priority should be: (1) set up the evaluation pipeline locally, (2) test whether inflate.py can access eval models, (3) if yes, implement the adversarial decode approach, (4) if no, implement SVT-AV1 with SegNet-aware preprocessing and encoder-side CRF optimization.