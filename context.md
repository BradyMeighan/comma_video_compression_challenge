# comma video compression challenge -- Comprehensive Context Document

This document contains every verifiable fact about the comma.ai video compression challenge, the repository structure, evaluation pipeline, neural network architectures, submission format, and existing submissions. All references cite exact file paths and line numbers from the repository at `C:\Users\modce\Documents\comma_video_compression_challenge`.

---

## 1. Challenge Overview

**What it is:** A video compression challenge run by [comma.ai](https://comma.ai). Participants must compress a dashcam video as small as possible while preserving semantic content and temporal dynamics, as measured by two specific neural networks (a SegNet and a PoseNet).

**Repository:** <https://github.com/commaai/comma_video_compression_challenge>

**Deadline:** May 3rd, 2026, 11:59pm AOE (`README.md:31`)

**License:** MIT, Copyright (c) 2026 comma.ai (`LICENSE:1-22`)

**Prize pool** (`README.md:32-35`):
- 1st place: comma four OR $1,000 + special swag
- 2nd place: $500 + special swag
- 3rd place: $250 + special swag
- Best write-up (visualizations, patterns, etc.): comma four OR $1,000 + special swag

**Leaderboard link:** <https://comma.ai/leaderboard> (`README.md:5`)

---

## 2. Scoring Formula

The score is computed as (lower is better) (`README.md:25`, `evaluate.py:92`):

```
score = 100 * segnet_distortion + 25 * rate + sqrt(10 * posenet_distortion)
```

Exact code from `evaluate.py:92`:
```python
score = 100 * segnet_dist +  math.sqrt(posenet_dist * 10)  + 25 * rate
```

### Component Definitions

| Component | Formula | Weight | What it measures |
|-----------|---------|--------|------------------|
| `segnet_distortion` | Average fraction of pixels where `argmax(pred_original) != argmax(pred_reconstructed)` | 100x multiplier | Semantic content preservation -- class-level segmentation agreement |
| `rate` | `archive.zip size / original file size (37,545,489 bytes)` | 25x multiplier | Compression efficiency |
| `posenet_distortion` | MSE of first 6 of 12 output dimensions between original and reconstructed frame pairs | sqrt(10x) | Temporal dynamics / ego-motion preservation |

### Weight Sensitivity Analysis (`DEEP_RESEARCH_PROMPT.md:19-25`)

- **SegNet (100x multiplier):** Even 1% pixel disagreement (0.01) costs 1.0 points. This is the dominant term.
- **Rate (25x multiplier):** Going from 6% to 3% compression rate saves only 0.75 points (25 * 0.03 = 0.75).
- **PoseNet (under square root):** `sqrt(10 * 0.38) = 1.95`. Halving PoseNet distortion from 0.38 to 0.19 saves only ~0.57 points (`sqrt(10*0.38) - sqrt(10*0.19) = 1.949 - 1.378 = 0.571`).

### Baseline Score Breakdown (`README.md:87-93`)

For `baseline_fast` (score 4.39):
- Average PoseNet Distortion: 0.38042614
- Average SegNet Distortion: 0.00946623
- Submission file size: 2,244,900 bytes
- Original uncompressed size: 37,545,489 bytes
- Compression Rate: 0.05979147
- Score components: `100 * 0.00947 = 0.947` (segnet) + `sqrt(10 * 0.380) = 1.949` (posenet) + `25 * 0.0598 = 1.495` (rate) = 4.39

---

## 3. The Video

**File:** `videos/0.mkv` (`README.md:16`)

**Properties** (verified via ffprobe):
- **Size:** 37,545,489 bytes (37.5 MB)
- **Duration:** 60.000 seconds (1 minute)
- **Frame rate:** 20 fps
- **Total frames:** 1,200
- **Resolution:** 1164x874 (coded: 1184x896)
- **Codec:** H.265 / HEVC, Main profile, level 93
- **Container:** Matroska (MKV)
- **Pixel format:** yuv420p
- **Color range:** tv (limited range)
- **Chroma location:** left
- **Has B-frames:** No (0)
- **References:** 1
- **Bitrate:** 5,006,065 bps (~5 Mbps)
- **Encoder:** Lavf60.16.100

**Source:** comma2k19 dashcam dataset. The specific segment is `b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc` (`public_test_segments.txt:1`, also stored as MKV metadata tag `SEGMENT`).

**Video names file:** `public_test_video_names.txt` contains a single line: `0.mkv` (`public_test_video_names.txt:1`).

---

## 4. SegNet Architecture

**Defined in:** `modules.py:103-128`

**Architecture:** `segmentation_models_pytorch.Unet` with `tu-efficientnet_b2` encoder (`modules.py:105`)

```python
class SegNet(smp.Unet):
  def __init__(self):
    super().__init__('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)
```

**Key parameters:**
- Encoder: `tu-efficientnet_b2`
- Output classes: 5
- Activation: None (raw logits)
- Encoder weights: None (no pretrained ImageNet weights; custom-trained)

**Weights file:** `models/segnet.safetensors` -- 38,502,892 bytes (~38 MB)

**Preprocessing** (`modules.py:107-109`):
1. Takes the **last frame** of each 2-frame sequence: `x = x[:, -1, ...]`
2. Resizes to `segnet_model_input_size = (512, 384)` via **bilinear** interpolation
3. Input shape: `(B, 3, 384, 512)` -- RGB float

The input transformation chain in `DistortionNet.preprocess_input` (`modules.py:143-148`):
1. Rearrange from `(B, T, H, W, C)` to `(B, T, C, H, W)` float
2. SegNet takes `x[:, -1, ...]` -- only the last frame (index -1, i.e., frame index 1 in 0-indexed 2-frame sequence)
3. Resize via `F.interpolate` to `(384, 512)` with `mode='bilinear'`

**Distortion computation** (`modules.py:111-113`):
```python
def compute_distortion(self, out1, out2):
    diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
    return diff.mean(dim=tuple(range(1, diff.ndim)))
```
- `out1` and `out2` are segmentation logits of shape `(B, 5, 384, 512)`
- `argmax(dim=1)` produces per-pixel class labels (0-4)
- Distortion = fraction of pixels where the argmax class disagrees
- Result shape: `(B,)` -- one scalar per sample

**Critical implication:** As long as the dominant class per pixel does not flip, there is zero distortion. Only the ranking of class logits matters, not their magnitudes.

---

## 5. PoseNet Architecture

**Defined in:** `modules.py:61-84`

### Backbone

**Architecture:** FastViT-T12 from timm (`modules.py:66`)

```python
self.vision = timm.create_model('fastvit_t12', pretrained=False, num_classes=VISION_FEATURES, in_chans=IN_CHANS, act_layer=timm.layers.get_act_layer(ACT_LAYER))
```

**Constants** (`modules.py:20-26`):
- `BN_EPS = 0.001`
- `BN_MOM = 0.01`
- `VISION_FEATURES = 2048`
- `SUMMARY_FEATURES = 512`
- `IN_CHANS = 6 * 2 = 12` (6 YUV420 channels per frame x 2 frames)
- `ACT_LAYER = 'gelu_tanh'`
- `HEADS = [Head('pose', 32, 12)]` -- single head named "pose" with hidden=32, out=12

**Weights file:** `models/posenet.safetensors` -- 55,835,560 bytes (~56 MB)

### Full Architecture Pipeline

1. **Vision backbone:** `fastvit_t12` -- outputs 2048-dim feature vector
2. **Summarizer** (`modules.py:67`): `Linear(2048, 512) -> ReLU -> ResBlock(512)`
3. **Hydra head** (`modules.py:45-59`): `ResBlock(512) -> Linear(512, 32) -> ReLU -> residual Linear(32,32) -> ReLU -> Linear(32, 12)`

### Normalization

Input normalization (`modules.py:64-65`):
```python
self.register_buffer('_mean', torch.tensor([255 / 2] * IN_CHANS).view(1, IN_CHANS, 1, 1))  # 127.5
self.register_buffer('_std', torch.tensor([255 / 4] * IN_CHANS).view(1, IN_CHANS, 1, 1))   # 63.75
```
Applied as: `(x - 127.5) / 63.75`

### Preprocessing (`modules.py:70-74`)

```python
def preprocess_input(self, x):
    batch_size, seq_len, *_ = x.shape
    x = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len, c=3)
    x = torch.nn.functional.interpolate(x, size=(segnet_model_input_size[1], segnet_model_input_size[0]), mode='bilinear')
    return einops.rearrange(rgb_to_yuv6(x), '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6)
```

Step by step:
1. Input is `(B, T=2, C=3, H, W)` float RGB
2. Rearrange to `(B*T, 3, H, W)` -- flatten batch and time
3. Resize each frame to `(384, 512)` via bilinear interpolation
4. Convert RGB to YUV6 via `rgb_to_yuv6()` -- produces `(B*T, 6, H/2, W/2)`
5. Rearrange back: stack 2 frames' 6 channels into 12 channels: `(B, 12, H/2, W/2)` i.e. `(B, 12, 192, 256)`

### Distortion Computation (`modules.py:82-84`)

```python
def compute_distortion(self, out1, out2):
    distortion_heads = ['pose']
    return sum((out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(dim=tuple(range(1, out1[h.name].ndim))) for h in self.hydra.heads if h.name in distortion_heads)
```

- Output is a dict: `{'pose': tensor of shape (B, 12)}`
- Distortion uses only the **first 6 of 12 output dimensions**: `[..., :6]` (since `h.out // 2 = 12 // 2 = 6`)
- Distortion = MSE between original and reconstructed outputs on those 6 dims
- Result shape: `(B,)` -- one scalar per sample

### Supporting Classes

**AllNorm** (`modules.py:28-33`): Normalizes all features as a single batch norm channel.
```python
class AllNorm(nn.Module):
    def __init__(self, num_features, eps=0.001, momentum=0.01, affine=True):
        self.bn = nn.BatchNorm1d(1, eps, momentum, affine)
    def forward(self, x):
        return self.bn(x.view(-1, 1)).view(x.shape)
```

**ResBlock** (`modules.py:35-43`): Two-branch residual block with AllNorm and ReLU. Each branch has `Linear -> Norm -> ReLU -> Linear -> Norm`. Output = `ReLU(a_out + block_b(a_out))` where `a_out = x + block_a(x)`.

**Hydra** (`modules.py:45-59`): Multi-head prediction module. Has a shared `ResBlock(512)`, then per-head: `Linear(512, hidden) -> ReLU -> residual(Linear(hidden, hidden) -> ReLU -> Linear(hidden, hidden)) -> ReLU -> Linear(hidden, out)`.

---

## 6. Color Space Details (YUV420)

**Defined in:** `frame_utils.py:51-78`

### RGB to YUV6 Conversion (`rgb_to_yuv6`)

Uses **BT.601 coefficients** (`frame_utils.py:60-63`):
```python
kYR, kYG, kYB = 0.299, 0.587, 0.114
Y = (R * kYR + G * kYG + B * kYB).clamp_(0.0, 255.0)
U = ((B - Y) / 1.772 + 128.0).clamp_(0.0, 255.0)
V = ((R - Y) / 1.402 + 128.0).clamp_(0.0, 255.0)
```

### 4:2:0 Chroma Subsampling (`frame_utils.py:65-71`)

Chroma channels (U and V) are 2x2 box-averaged:
```python
U_sub = (U[..., 0::2, 0::2] + U[..., 1::2, 0::2] + U[..., 0::2, 1::2] + U[..., 1::2, 1::2]) * 0.25
V_sub = (V[..., 0::2, 0::2] + V[..., 1::2, 0::2] + V[..., 0::2, 1::2] + V[..., 1::2, 1::2]) * 0.25
```

### YUV6 Channel Layout (`frame_utils.py:74-78`)

The 6 channels are the 4 luma sub-pixels + 2 chroma channels:
```python
y00 = Y[..., 0::2, 0::2]  # channel 0: top-left luma
y10 = Y[..., 1::2, 0::2]  # channel 1: bottom-left luma
y01 = Y[..., 0::2, 1::2]  # channel 2: top-right luma
y11 = Y[..., 1::2, 1::2]  # channel 3: bottom-right luma
return torch.stack([y00, y10, y01, y11, U_sub, V_sub], dim=-3)
```

Output shape: `(B, 6, H/2, W/2)` for input `(B, 3, H, W)`.

### YUV420 to RGB Conversion (`yuv420_to_rgb`, `frame_utils.py:159-183`)

Used for decoding video frames from PyAV. Uses **BT.601 limited range**:
```python
yf = (y_t - 16.0) * (255.0 / 219.0)
uf = (u_up - 128.0) * (255.0 / 224.0)
vf = (v_up - 128.0) * (255.0 / 224.0)

r = (yf + 1.402 * vf).clamp(0, 255)
g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0, 255)
b = (yf + 1.772 * uf).clamp(0, 255)
```

Chroma upsampling uses `F.interpolate` with `mode='bilinear'` and `align_corners=False` (`frame_utils.py:173-174`).

**Implication:** PoseNet is much more sensitive to luminance than chrominance. Chrominance is already spatially downsampled 2x in each dimension (4:2:0). The 4 luma sub-pixel channels carry most of the spatial information.

---

## 7. Evaluation Pipeline

### High-Level Flow (`evaluate.sh:1-77`)

1. **Verify files exist:** Check `archive.zip` and `inflate.sh` exist in the submission directory (`evaluate.sh:31-39`)
2. **Unzip archive:** `unzip -o archive.zip -d archive/` (`evaluate.sh:42-44`)
3. **Run inflate.sh:** `bash inflate.sh <archive_dir> <inflated_dir> <video_names_file>` (`evaluate.sh:47`)
4. **Assert .raw files exist:** Checks that every video name in the names file has a corresponding `.raw` file in the inflated directory (`evaluate.sh:50-64`)
5. **Run evaluate.py:** Computes distortion scores and generates `report.txt` (`evaluate.sh:69-74`)

### evaluate.py Details (`evaluate.py:1-112`)

**Default arguments** (`evaluate.py:10-18`):
- `batch_size`: 16
- `num_threads`: 2
- `prefetch_queue_depth`: 4
- `seed`: 1234
- `device`: auto-detect (cuda > mps > cpu)

**Data loading:**
- Ground truth: `DaliVideoDataset` (CUDA) or `AVVideoDataset` (CPU/MPS) loading from `videos/` directory (`evaluate.py:58-60`)
- Compressed: `TensorVideoDataset` loading `.raw` files from `<submission_dir>/inflated/` (`evaluate.py:67-69`)

**Evaluation loop** (`evaluate.py:72-83`):
- Iterates over batches from both ground truth and compressed datasets in parallel via `zip(dl_gt, dl_comp)`
- Asserts batch shape is `[B, seq_len=2, H=874, W=1164, C=3]` (`evaluate.py:77`)
- Calls `distortion_net.compute_distortion(batch_gt, batch_comp)` which returns `(posenet_dist, segnet_dist)` per sample (`evaluate.py:79`)
- Accumulates sums and counts across all batches

**Sample count:**
- 1200 frames / 2 frames per sequence = 600 sequences (non-overlapping pairs)
- Report header says "600 samples" (`README.md:87`)

**Rate computation** (`evaluate.py:63-65`):
```python
compressed_size = (args.submission_dir / 'archive.zip').stat().st_size
uncompressed_size = sum(file.stat().st_size for file in args.uncompressed_dir.rglob('*') if file.is_file())
rate = compressed_size / uncompressed_size
```

The uncompressed size is the total size of all files in the `videos/` directory (for single video: 37,545,489 bytes).

### DistortionNet Preprocessing (`modules.py:143-148`)

The `DistortionNet.preprocess_input` method receives raw batch `(B, T=2, H=874, W=1164, C=3)` uint8:
1. Rearranges to `(B, T, C, H, W)` float
2. Passes to `posenet.preprocess_input()` -- resizes all frames to 512x384, converts to YUV6, stacks 2 frames to 12 channels
3. Passes to `segnet.preprocess_input()` -- takes last frame only, resizes to 512x384

### Batch Structure

The dataloader creates **non-overlapping 2-frame sequences** (`frame_utils.py:10`): `seq_len = 2`.

Frame pairs: (0,1), (2,3), (4,5), ..., (1198,1199) -- 600 pairs total.

- **SegNet sees:** frames 1, 3, 5, 7, ..., 1199 (the "last" frame of each pair, `modules.py:108`)
- **PoseNet sees:** both frames of each pair (0,1), (2,3), etc. -- all 1200 frames via 2-frame windows

---

## 8. Submission Format

### Required Files (`README.md:98-103`)

A submission is a Pull Request to the repo that includes:
1. **A download link to `archive.zip`** -- the compressed data
2. **`inflate.sh`** -- a bash script that converts extracted `archive/` into raw video frames
3. **Optional:** compression script (`compress.sh`) and any other assets

### inflate.sh Contract

Called as: `bash inflate.sh <archive_dir> <inflated_dir> <video_names_file>` (`evaluate.sh:47`)

Must produce: `.raw` files in `<inflated_dir>/` -- one per video name, with `.raw` extension replacing the original extension.

### .raw File Format

A `.raw` file is a flat binary dump of uint8 RGB frames with shape `(N, H, W, 3)` where:
- N = number of frames (1200 for the test video)
- H = 874
- W = 1164
- 3 = RGB channels
- No header
- Data type: uint8

Size per frame: `874 * 1164 * 3 = 3,050,928 bytes`
Total .raw size for 1200 frames: `3,050,928 * 1200 = 3,661,113,600 bytes` (~3.6 GB)

### PR Template (`/.github/pull_request_template.md`)

Fields:
1. Submission name (directory name)
2. Upload zipped `archive.zip` (via drag-and-drop file upload, must work with `curl -L`)
3. `report.txt` content
4. Whether submission requires GPU for evaluation (inflation)
5. Whether compression script is included and should be merged
6. Additional comments / solution description

### .gitignore Exclusions for Submissions (`/.gitignore:225-228`)

```
archive.zip
archive/
inflated/
report.txt
```

These artifacts are excluded from git tracking, so they must be provided via download links.

---

## 9. Evaluation Environment

**Time limit:** 30 minutes (`README.md:114`, `.github/workflows/eval.yml:30`)

### CPU Runner (`README.md:114`)
- Runner: `ubuntu-latest`
- CPU: 4 cores
- RAM: 16 GB

### GPU Runner (`README.md:114`)
- Runner: `linux-nvidia-t4`
- GPU: NVIDIA T4 (16 GB VRAM)
- RAM: 26 GB

### CI/CD Workflow (`.github/workflows/eval.yml:1-145`)

**Trigger:** `workflow_dispatch` with inputs:
- `pr_number` (optional string)
- `submission_name` (required string)
- `submission_url` (required string -- link to archive.zip)
- `runner` (choice: `ubuntu-latest` or `linux-nvidia-t4`, default: `ubuntu-latest`)

**Environment variables:**
- `UV_GROUP`: `cu128` if T4 runner, else `cpu` (`.github/workflows/eval.yml:32`)
- `EVAL_DEVICE`: `cuda` if T4 runner, else `cpu` (`.github/workflows/eval.yml:33`)

**Steps:**
1. Checkout repo with PR merge ref: `refs/pull/{pr_number}/merge` (`.github/workflows/eval.yml:37-39`)
2. Check submission name uniqueness against master (`.github/workflows/eval.yml:41-47`)
3. Check GPU via `nvidia-smi` if T4 runner (`.github/workflows/eval.yml:50-53`)
4. Download archive.zip via `curl -L` to `submissions/<name>/archive.zip` (`.github/workflows/eval.yml:56-60`)
5. Install git-lfs and pull LFS files (`.github/workflows/eval.yml:62-71`)
6. Install uv (`.github/workflows/eval.yml:73-77`)
7. Install Python dependencies: `uv sync --group $UV_GROUP` (`.github/workflows/eval.yml:79-81`)
8. Install ffmpeg (`.github/workflows/eval.yml:83-85`)
9. Run evaluation: `uv run --group $UV_GROUP bash evaluate.sh --device $EVAL_DEVICE --submission-dir ./submissions/$name` (`.github/workflows/eval.yml:87-89`)
10. Upload artifacts (archive.zip + report.txt) (`.github/workflows/eval.yml:91-98`)

**PR Comment job** (`.github/workflows/eval.yml:100-145`):
- Runs after test job completes (always, if pr_number provided)
- Posts evaluation results or failure message as a PR comment

**PR Welcome workflow** (`.github/workflows/pr-welcome.yml:1-47`):
- Triggers on PR open to master
- Posts welcome comment and assigns reviewer `YassineYousfi`

### What inflate.sh Can Access at Eval Time

The CI checkout step (`.github/workflows/eval.yml:37-39`) checks out the **full repo** including the PR's code changes:
```yaml
ref: ${{ inputs.pr_number && format('refs/pull/{0}/merge', inputs.pr_number) || 'master' }}
```

This means `inflate.sh` (and any Python scripts it calls) can access:
- `frame_utils.py` (and all its functions: `yuv420_to_rgb`, `rgb_to_yuv6`, etc.)
- `modules.py` (and all its classes: `PoseNet`, `SegNet`, `DistortionNet`, etc.)
- `models/segnet.safetensors` (38 MB) -- pulled via LFS
- `models/posenet.safetensors` (56 MB) -- pulled via LFS
- The original video `videos/0.mkv` (37.5 MB) -- pulled via LFS
- Any code added by the PR itself (new Python files in submission directory, etc.)
- All installed Python packages (`uv sync` runs before evaluate.sh)

The baseline `inflate.py` already imports from `frame_utils`: `from frame_utils import camera_size, yuv420_to_rgb` (`submissions/baseline_fast/inflate.py:4`).

---

## 10. Rules (Exact Quotes)

From `README.md:117-120`:

> - External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. This applies to the PoseNet and SegNet.
> - You can use anything for compression, including the models, original uncompressed video, and any other assets you want to include.
> - You may include your compression script in the submission, but it's not required.

---

## 11. Existing Submissions (In-Repo)

### baseline_fast (Score: 4.39)

**Location:** `submissions/baseline_fast/`

**Compression** (`submissions/baseline_fast/compress.sh:42-47`):
```bash
ffmpeg -nostdin -y -hide_banner -loglevel warning \
  -r 20 -fflags +genpts -i "$IN" \
  -vf "scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos" \
  -c:v libx265 -preset ultrafast -crf 30 \
  -g 1 -bf 0 -x265-params "keyint=1:min-keyint=1:scenecut=0:frame-threads=4:log-level=warning" \
  -r 20 "$OUT"
```

Parameters:
- **Downscale:** 45% resolution via Lanczos filter (`iw*0.45`, `ih*0.45`), truncated to even: approximately 524x394 (exact: `trunc(1164*0.45/2)*2 = trunc(261.9)*2 = 522`, `trunc(874*0.45/2)*2 = trunc(196.65)*2 = 392` -- so 522x392)
- **Codec:** libx265, ultrafast preset
- **CRF:** 30
- **GOP:** All-intra (keyint=1, min-keyint=1, no B-frames, scenecut=0)
- **Output:** MKV container
- **Archive:** zip of the MKV files

**Inflation** (`submissions/baseline_fast/inflate.py:7-23`):
1. Decode MKV via PyAV
2. Convert YUV420 to RGB via `yuv420_to_rgb()`
3. If frame dimensions don't match `camera_size` (1164x874), upscale via `F.interpolate` with `mode='bicubic'` and `align_corners=False`
4. Clamp to [0, 255], round, convert to uint8
5. Write flat binary to `.raw` file

**Results:**
- Archive size: 2,244,900 bytes (~2.24 MB)
- Compression rate: 0.05979147 (~6.0%)
- SegNet distortion: 0.00946623 (contributes 0.947 to score)
- PoseNet distortion: 0.38042614 (contributes 1.949 to score)
- Rate contribution: 25 * 0.0598 = 1.495
- **Total score: 4.39**

### no_compress (Score: 25.0)

**Location:** `submissions/no_compress/`

**Compression** (`submissions/no_compress/compress.sh:26-32`): Simply copies the original MKV file into the archive directory and zips it.

**Inflation** (`submissions/no_compress/inflate.py:6-16`): Decodes the MKV and writes raw RGB frames. No resizing needed since it's the original video.

**Results:**
- Zero distortion (both SegNet and PoseNet)
- Rate = 1.0 (archive ~= original size)
- Score = 0 + 0 + 25 * 1.0 = **25.0**

---

## 12. Known External Submissions (from PRs)

### PR #10 -- haraschax (Score: 2.99)

From `DEEP_RESEARCH_PROMPT.md` context and leaderboard data:
- **Codec:** SVT-AV1
- **Downscale:** 35% resolution
- **CRF:** 30
- **Preset:** 4
- **GOP:** 180 frames
- **Archive size:** ~710 KB
- **Score: 2.99**

### PR #17 -- EthanYangTW (Score: 2.90, current leader)

From `DEEP_RESEARCH_PROMPT.md` context and leaderboard data:
- **Codec:** SVT-AV1
- **Downscale:** 45% resolution
- **CRF:** 32
- **Preset:** 0 (slowest/best quality)
- **GOP:** 180 frames
- **Archive size:** ~835 KB
- **Score: 2.90**

### PR #15 -- sweeter_codec8 (Closed, no official score)

- Included a `TinyAR` neural refinement model with quantized INT8 weights
- Demonstrated that it is possible to load safetensors and run custom neural networks at decode time
- Was closed without a score being assigned

### PR #14 -- YassineYousfi test (Score: 4.39)

- A test submission that simply replicates the baseline
- Score: 4.39 (identical to baseline_fast)

---

## 13. Frame Utils Details

**File:** `frame_utils.py`

### Constants (`frame_utils.py:10-13`)

```python
seq_len = 2
camera_size = (1164, 874)     # (width, height)
camera_fl = 910.               # focal length (unused in evaluation)
segnet_model_input_size = (512, 384)  # (width, height)
```

### HEVC Buffer / Frame Count Utilities

- `hevc_buffer_mmap(path)` (`frame_utils.py:15-19`): Memory-maps a HEVC file for DALI input.
- `_hevc_frame_count(path)` (`frame_utils.py:21-32`): Counts frames in raw HEVC by scanning for NAL unit start codes (`0x00 0x00 0x01`) and checking VCL slice types (NAL type <= 31).
- `_container_frame_count(path)` (`frame_utils.py:34-42`): Uses PyAV to count frames in container formats.
- `frame_count(path)` (`frame_utils.py:44-47`): Dispatches to HEVC or container method based on file extension.

### Dataset Classes

**VideoDataset (base)** (`frame_utils.py:80-108`):
- Common interface for all dataset types
- Accepts: `file_names`, `data_dir`, `batch_size`, `device`, `format`, `num_threads`, `seed`, `prefetch_queue_depth`
- Handles distributed training rank/world_size partitioning
- Files are partitioned across ranks: `self.file_names = self.all_file_names[self.rank::self.world_size]`

**DaliVideoDataset** (`frame_utils.py:110-157`):
- For CUDA devices only
- Uses NVIDIA DALI `fn.experimental.inputs.video` for GPU-accelerated video decoding
- `sequence_length=seq_len`, `last_sequence_policy="pad"`
- Yields `(path, idx, vid)` where `vid` is `(B, seq_len, H, W, C)` on GPU

**AVVideoDataset** (`frame_utils.py:185-216`):
- For CPU and MPS devices
- Uses PyAV for software video decoding
- Decodes frame-by-frame, converts via `yuv420_to_rgb()`, accumulates into sequences and batches
- Yields `(path, idx, batch)` where batch is `(B, seq_len, H, W, C)` uint8

**TensorVideoDataset** (`frame_utils.py:218-253`):
- Loads pre-decoded `.raw` files via numpy memory-mapping
- Constructor forces format to `.raw`: `super().__init__(format='raw', ...)`
- Memory-maps the file as `np.memmap(path, dtype=np.uint8, mode='r', shape=(N, H, W, C))`
- Frame count derived from: `N = file_size // (H * W * C)` where `H=874`, `W=1164`, `C=3`
- Groups frames into sequences and batches identically to AVVideoDataset

---

## 14. Modules Details

**File:** `modules.py`

### DistortionNet (`modules.py:130-158`)

Wraps both PoseNet and SegNet:

```python
class DistortionNet(nn.Module):
    def __init__(self):
        self.posenet = PoseNet()
        self.segnet = SegNet()
```

**Loading weights** (`modules.py:136-141`):
```python
def load_state_dicts(self, posenet_sd_path, segnet_sd_path, device):
    posenet_sd = load_file(posenet_sd_path, device=str(device))
    segnet_sd = load_file(segnet_sd_path, device=str(device))
    self.posenet.load_state_dict(posenet_sd)
    self.segnet.load_state_dict(segnet_sd)
```

**Preprocessing** (`modules.py:143-148`):
```python
def preprocess_input(self, x):
    batch_size, seq_len, *_ = x.shape
    x = einops.rearrange(x, 'b t h w c -> b t c h w', b=batch_size, t=seq_len, c=3).float()
    posenet_in = self.posenet.preprocess_input(x)
    segnet_in = self.segnet.preprocess_input(x)
    return posenet_in, segnet_in
```

**Forward pass** (`modules.py:150-152`):
```python
def forward(self, x):
    posenet_in, segnet_in = self.preprocess_input(x)
    return self.posenet(posenet_in), self.segnet(segnet_in)
```

Note the TODO comment: `# TODO run in bfloat16?`

**Compute distortion** (`modules.py:154-158`):
```python
@torch.inference_mode()
def compute_distortion(self, x, y):
    posenet_out_x, segnet_out_x = self(x)
    posenet_out_y, segnet_out_y = self(y)
    return self.posenet.compute_distortion(posenet_out_x, posenet_out_y), self.segnet.compute_distortion(segnet_out_x, segnet_out_y)
```

### Helper Utilities

- `get_viewer()` (`modules.py:12-16`): Finds image viewer (eog or xdg-open) for debug visualization.
- Path constants (`modules.py:17-18`):
  ```python
  segnet_sd_path = HERE / 'models/segnet.safetensors'
  posenet_sd_path = HERE / 'models/posenet.safetensors'
  ```

---

## 15. Repository Structure

### File Tree

```
comma_video_compression_challenge/
  .devcontainer/
    devcontainer.json                   # Python 3 devcontainer with git-lfs, ffmpeg, uv
  .github/
    pull_request_template.md            # PR template for submissions
    workflows/
      eval.yml                          # Main evaluation workflow
      pr-welcome.yml                    # Auto-comment + assign reviewer on PR open
  .gitattributes                        # LFS tracking rules
  .gitignore                            # Standard Python + submission artifacts
  .python-version                       # 3.11
  DEEP_RESEARCH_PROMPT.md               # Research prompt for competitive approaches
  LICENSE                               # MIT License, 2026 comma.ai
  README.md                             # Challenge description, rules, quickstart
  download_and_remux.sh                 # Downloads 64 test videos from HuggingFace
  evaluate.py                           # Main evaluation script
  evaluate.sh                           # Orchestrates unzip -> inflate -> evaluate
  frame_utils.py                        # Video loading, YUV conversion, dataset classes
  models/
    posenet.safetensors                 # 55,835,560 bytes (LFS)
    segnet.safetensors                  # 38,502,892 bytes (LFS)
  modules.py                            # PoseNet, SegNet, DistortionNet definitions
  public_test_segments.txt              # Source segment path from comma2k19
  public_test_video_names.txt           # "0.mkv"
  pyproject.toml                        # Project config and dependencies
  submissions/
    __init__.py                         # Empty
    baseline_fast/
      __init__.py                       # Empty
      compress.sh                       # x265 ultrafast compression script
      inflate.py                        # Decode + bicubic upscale
      inflate.sh                        # Shell wrapper for inflate.py
    no_compress/
      __init__.py                       # Empty
      compress.sh                       # Just copies original
      inflate.py                        # Just decodes original
      inflate.sh                        # Shell wrapper for inflate.py
  videos/
    0.mkv                               # 37,545,489 bytes (LFS)
```

### Git LFS Tracked Types (`.gitattributes:1-66`)

Extensive list including: `.7z`, `.arrow`, `.bin`, `.bz2`, `.ckpt`, `.ftz`, `.gz`, `.h5`, `.joblib`, `.lz4`, `.mlmodel`, `.model`, `.msgpack`, `.npy`, `.npz`, `.onnx`, `.ot`, `.parquet`, `.pb`, `.pickle`, `.pkl`, `.pt`, `.pth`, `.rar`, `.safetensors`, `.tar`, `.tflite`, `.tgz`, `.wasm`, `.xz`, `.zip`, `.zst`, `.pcm`, `.sam`, `.raw`, `.aac`, `.flac`, `.mp3`, `.ogg`, `.wav`, `.bmp`, `.gif`, `.png`, `.tiff`, `.jpg`, `.jpeg`, `.webp`, `.hevc`, `.h265`, `.mkv`, `.mp4`, `.h264`, `.mov`, `.avi`, `.webm`, `.wmv`, plus `saved_model/**/*` and `*tfevents*`.

### Python Version

`3.11` (`.python-version:1`)

### Dependencies (`pyproject.toml:5-14`)

Core dependencies:
- `numpy`
- `einops`
- `timm`
- `safetensors`
- `segmentation-models-pytorch`
- `tqdm`
- `pillow`
- `av`

Dependency groups for PyTorch:
- `cpu`: torch, torchvision (from pytorch-cpu index)
- `cu126`: torch, torchvision, nvidia-dali-cuda120
- `cu128`: torch, torchvision, nvidia-dali-cuda120
- `cu130`: torch, torchvision, nvidia-dali-cuda130
- `mps`: torch, torchvision (from default PyPI)

### DevContainer (`.devcontainer/devcontainer.json`)

```json
{
  "name": "comma-video-compression-challenge",
  "image": "mcr.microsoft.com/devcontainers/python:3",
  "postCreateCommand": "sudo apt-get update && sudo apt-get install -y git-lfs ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh"
}
```

---

## 16. Additional Test Data

### download_and_remux.sh (`download_and_remux.sh:1-79`)

Downloads and processes additional test videos from the comma2k19 dataset:
- **Source:** `https://huggingface.co/datasets/commaai/comma2k19/resolve/main/compression_challenge/test_videos.zip`
- **Size:** ~2.4 GB archive containing 64 driving videos as raw `.hevc` files
- **Process:** Downloads zip, extracts, remuxes each HEVC file listed in `public_test_segments.txt` into MKV containers at 20fps with `ffmpeg -c copy`
- **Output:** Numbered `.mkv` files in `videos/` directory, names written to `public_test_video_names.txt`

### public_test_segments.txt

Contains the comma2k19 segment path: `b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc`

This is the source for the single test video `0.mkv`.

### Grid Search Data

From `README.md:130-131`:
> Check out this large grid search over various ffmpeg parameters. Each point in the figure corresponds to a ffmpeg setting. The fastest encoder setting was submitted as the baseline_fast. You can inspect the grid search [here](https://github.com/user-attachments/files/26169452/grid_search_results.csv) and look for patterns.

The README includes a scatter plot image showing rate vs. distortion tradeoffs across various ffmpeg parameter combinations.

---

## 17. Previous Challenge Context

From `DEEP_RESEARCH_PROMPT.md` and general knowledge:

comma.ai previously ran the **commaVQ compression challenge**, which used VQ-VAE tokenized representations. Key findings from that challenge:

- Top submissions trained their own models to essentially memorize the data
- A Reddit thread confirms: "every top submission trained their own" model
- A competitor named Szabolcs-cs used "Self-Compressing Neural Networks"
- The current challenge explicitly allows using the provided models, original video, and any assets for compression (`README.md:119`)

---

## 18. Technical Constants Reference

| Constant | Value | Location |
|----------|-------|----------|
| `seq_len` | 2 | `frame_utils.py:10` |
| `camera_size` | (1164, 874) -- (W, H) | `frame_utils.py:11` |
| `camera_fl` | 910.0 | `frame_utils.py:12` |
| `segnet_model_input_size` | (512, 384) -- (W, H) | `frame_utils.py:13` |
| `BN_EPS` | 0.001 | `modules.py:20` |
| `BN_MOM` | 0.01 | `modules.py:21` |
| `VISION_FEATURES` | 2048 | `modules.py:22` |
| `SUMMARY_FEATURES` | 512 | `modules.py:23` |
| `IN_CHANS` | 12 (6 * 2) | `modules.py:24` |
| `ACT_LAYER` | 'gelu_tanh' | `modules.py:25` |
| `HEADS` | [Head('pose', 32, 12)] | `modules.py:26` |
| PoseNet normalization mean | 127.5 (255/2) | `modules.py:64` |
| PoseNet normalization std | 63.75 (255/4) | `modules.py:65` |
| Original video size | 37,545,489 bytes | `evaluate.py:64`, ffprobe |
| Frame count | 1200 | ffprobe |
| Sample count (evaluation) | 600 | 1200 / seq_len |
| Batch size (default) | 16 | `evaluate.py:10` |
| Frame size (bytes) | 3,050,928 | 874 * 1164 * 3 |
| Raw file size (1200 frames) | 3,661,113,600 | 1200 * 3,050,928 |
| SegNet classes | 5 | `modules.py:105` |
| PoseNet output dims | 12 | `modules.py:26` |
| PoseNet distortion dims | 6 (first half) | `modules.py:84` |
| YUV6 channels | 6 (y00, y10, y01, y11, U_sub, V_sub) | `frame_utils.py:78` |
| BT.601 Y coefficients | kYR=0.299, kYG=0.587, kYB=0.114 | `frame_utils.py:60` |

---

## 19. Evaluation Data Flow Diagram

```
archive.zip
    |
    v
[unzip] --> archive/0.mkv (or other format)
    |
    v
[inflate.sh] --> inflated/0.raw  (flat uint8 RGB, 1200 x 874 x 1164 x 3)
    |
    v
[TensorVideoDataset] --> batches of (B, 2, 874, 1164, 3) uint8
    |
    v
[DistortionNet.preprocess_input]
    |                          |
    v                          v
PoseNet path:              SegNet path:
  rearrange BTHWC->BTCHW     rearrange BTHWC->BTCHW
  float()                     float()
  flatten (B*T, C, H, W)     take last frame [:, -1]
  bilinear resize 512x384    bilinear resize 512x384
  rgb_to_yuv6 -> 6ch         input: (B, 3, 384, 512)
  stack 2 frames -> 12ch
  input: (B, 12, 192, 256)
    |                          |
    v                          v
PoseNet forward:           SegNet forward:
  normalize (x-127.5)/63.75   Unet forward pass
  fastvit_t12 -> 2048-dim     output: (B, 5, 384, 512)
  Linear(2048,512) + ReLU
  ResBlock(512)
  Hydra -> {'pose': (B, 12)}
    |                          |
    v                          v
PoseNet distortion:        SegNet distortion:
  MSE on first 6 dims of      argmax disagreement fraction
  original vs reconstructed    across all pixels
  output -> (B,)               -> (B,)
    |                          |
    v                          v
    +---- average over all 600 samples ----+
                    |
                    v
    score = 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate
```

---

## 20. Key Implementation Details for inflate.sh

### How the baseline inflate.sh works (`submissions/baseline_fast/inflate.sh:1-28`)

1. Determines `HERE` (submission directory) and `ROOT` (repo root)
2. Sets `SUB_NAME` to the submission directory basename
3. Receives three positional args: `DATA_DIR`, `OUTPUT_DIR`, `FILE_LIST`
4. For each video name in `FILE_LIST`:
   - Constructs source path: `DATA_DIR/<basename>.mkv`
   - Constructs destination path: `OUTPUT_DIR/<basename>.raw`
   - Changes directory to `ROOT` (repo root)
   - Runs: `python -m "submissions.${SUB_NAME}.inflate" "$SRC" "$DST"`

This means the inflate script runs as a Python module within the repo's package structure, enabling imports from `frame_utils`, `modules`, or any other repo module.

### The no_compress inflate works identically (`submissions/no_compress/inflate.sh:1-28`)

Same shell structure, different Python module (`submissions.no_compress.inflate`), which just decodes without resizing.

---

## 21. Exact Downscale Resolution in baseline_fast

The ffmpeg scale filter (`submissions/baseline_fast/compress.sh:44`):
```
scale=trunc(iw*0.45/2)*2:trunc(ih*0.45/2)*2:flags=lanczos
```

Calculation:
- Width: `trunc(1164 * 0.45 / 2) * 2 = trunc(261.9) * 2 = 261 * 2 = 522`
- Height: `trunc(874 * 0.45 / 2) * 2 = trunc(196.65) * 2 = 196 * 2 = 392`

**Encoded resolution: 522x392** (not 524x394 as sometimes approximated).

Both dimensions are forced to even numbers (required for YUV420 encoding).

---

## 22. Distributed Evaluation Support

The evaluation pipeline supports distributed CUDA evaluation (`evaluate.py:30-50`):
- Uses `LOCAL_RANK`, `RANK`, `WORLD_SIZE` environment variables
- Initializes NCCL process group when `world_size > 1`
- All-reduces distortion sums across ranks
- Only rank 0 computes the final score and writes the report

For the competition evaluation (single machine), `world_size = 1` and distributed features are unused.
