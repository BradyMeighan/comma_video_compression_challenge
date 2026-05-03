# Codec Optimization Probe Notes (v5)

Date: 2026-04-10

Goal: stress-test `research/codec_optimization.md` claims before spending REN retraining cycles.

## Scope

- Codec-only ablations first (no REN, no sidecars).
- Then one sidecar compatibility probe on the best codec-only candidate.
- Environment constraints:
  - `ffmpeg` with `libsvtav1` available.
  - Native `SvtAv1EncApp` binaries installed under `third_party/svt/`.
  - `libsvtav1` reports SVT-AV1 `v4.0.1`.

## Main Matrix (`tools/run_codec_matrix.py`)

Results saved to `results.csv`.

Top 7 tested:

1. `p0_like_8bit_crf34_p1_fg30_dn0_keyinf_bicubic_us2` -> **2.2315**
2. `ref_8bit_crf33_fg22_k180_lanczos_unsharp045` -> 2.2721
3. `p0_like_10bit_crf34_p1_fg30_dn0_k180_bicubic_us2` -> 2.2855
4. `p0_like_10bit_crf34_p1_fg30_dn0_keyinf_lanczos_us045` -> 2.3030
5. `p0_like_10bit_crf34_p1_fg30_dn0_keyinf_bicubic_us2` -> 2.3090
6. `p0_like_10bit_crf34_p1_fg30_defaultdn_keyinf_bicubic_us2` -> 2.3090
7. `p0_like_10bit_crf34_p1_fg0_keyinf_bicubic_us2` -> 2.8972

Observations:

- Report-style 10-bit P0 variants did **not** beat reference in this environment.
- `film-grain-denoise=0` vs default gave identical result in this wrapper path.
- Turning film grain off destroyed pose quality despite lower bytes.

## Additional "Ultimate-like" Spot Checks

These approximate the report's high-CRF "ultimate spatial" direction (without PSYEX-only flags):

- `ultimate_like_10bit_p4_crf45` (fg30, keyint=-1, bicubic+unsharp2)
  - `seg=0.00889520 pose=0.28446606 size=514.2KB score=2.9267`
- `ultimate_like_8bit_p4_crf45` (fg30, keyint=-1, bicubic+unsharp2)
  - `seg=0.00902356 pose=0.31073436 size=535.4KB score=3.0302`

Observation:

- High-CRF "ultimate" direction was strongly negative in this setup.

## Sidecar Compatibility Probe (Top Codec Candidate Only)

Candidate: `p0_like_8bit_crf34_p1_fg30_dn0_keyinf_bicubic_us2`

1. Base codec-only: **2.2315**
2. + even `dxyr` (`optimize_apply_even_dxyr.py`):
   - `frame0_dxyr_q.bin` = 1094 bytes
   - fast_eval: **1.5241**
3. + even `ab` (`optimize_apply_even_ab.py`):
   - `frame0_ab_q.bin` = 755 bytes
   - fast_eval: **1.3682**

Conclusion:

- This codec is sidecar-friendly, but still not near the current best integrated pipeline (`~1.12` with REN + multi-pass sidecars).
- Recommendation is **not** to run full sidecar/REN retraining for every codec candidate.
- Better strategy: codec triage -> one quick sidecar compatibility pass -> retrain REN only for truly superior codec baselines.

## Focused Refinement (`tools/run_codec_refine.py`)

Refined 9-case sweep around the report-style 8-bit branch.
Results saved to `results_refine.csv`.

Top findings:

1. `8bit_crf35_fg30_dn0_keyinf_bicubic_us2` -> **2.2042** (best codec-only in this family)
2. `8bit_crf34_fg30_dn0_k180_bicubic_us2` -> 2.2249
3. `base_8bit_crf34_fg30_dn0_keyinf_bicubic_us2` -> 2.2315

Pattern summary:

- In this family, `CRF 35` beats `CRF 34/33`.
- `fg=30` remained best; lowering to `26` or `22` hurt total score.
- Bicubic + stronger unsharp was better than the lower-strength/lanczos alternatives tested.

## Sidecar Probe on Refined Winner

Candidate: `8bit_crf35_fg30_dn0_keyinf_bicubic_us2`

1. Base codec-only: **2.2042**
2. + even `dxyr`:
   - `frame0_dxyr_q.bin` = 1085 bytes
   - fast_eval: **1.4859**
3. + even `ab`:
   - `frame0_ab_q.bin` = 755 bytes
   - fast_eval: **1.3362**
4. + odd `ab` pass 1 (`probe_odd_frame_ab_joint.py`, tag `v5crf35_oab1`):
   - `frame1_ab_q_v5crf35_oab1.bin` = 298 bytes
   - fast_eval: **1.3313**

Final take from this branch:

- This is the strongest codec-only branch found in this probe.
- It still underperforms the existing integrated REN+sidecar baseline (`~1.12`) by a large margin.
- Any further work on this branch should only continue if paired with a full REN re-train + full odd/even sidecar re-optimization loop.

## Native `SvtAv1EncApp` Installation + Validation

Installed binaries:

- `third_party/svt/psyex-v3.0.2B-win64/SvtAv1EncApp.exe`
  - Source: `BlueSwordM/svt-av1-psyex` release `v3.0.2-B` (Windows x86-64-v3 zip)
  - Version check: `SVT-AV1-PSYEX v3.0.2-B`
  - Verified flags: `--fgs-table`, `--sharp-tx`, `--hbd-mds`, `--complex-hvs`
- `third_party/svt/ac-v2.1.2-win64/SvtAv1EncApp.exe`
  - Source: `AmusementClub/SVT-AV1` release `v2.1.2-AC` (Windows x86-64 zip)
  - Version check: `SVT-AV1 v2.1.2`
  - Verified flags: `--film-grain`, `--film-grain-denoise`, `--keyint`, `--scd`

Note:

- Local machine now has a working MinGW build toolchain at `C:\mingw64` (`gcc`, `g++`, `mingw32-make`, `ninja`, `nasm`, `yasm`).
- Source builds are no longer blocked and were executed (see section below).

## Native Benchmarks (full clip)

Preprocess input for native app:

- `native_cases/input_582x436_10b.y4m` (Lanczos-downscaled from `videos/0.mkv`)

### PSYEX case A (aggressive)

- Case: `native_cases/psyex_p4_crf45_hbd`
- Settings: `--preset 4 --crf 45 --keyint -1 --scd 0 --sharp-tx 1 --hbd-mds 1 --complex-hvs 1 --film-grain 30 --film-grain-denoise 0`
- fast_eval: `seg=0.00559852 pose=0.10994318 size=1330.4KB score=2.5155`

### PSYEX case B (lower CRF)

- Case: `native_cases/psyex_p1_crf34_hbd`
- Settings: `--preset 1 --crf 34 --keyint -1 --scd 0 --sharp-tx 1 --hbd-mds 1 --complex-hvs 1 --film-grain 30 --film-grain-denoise 0`
- fast_eval: `seg=0.00425973 pose=0.04524502 size=5830.4KB score=5.0740`

### AC v2.1.2 case

- Case: `native_cases/ac_v212_p1_crf34`
- Settings: `--preset 1 --crf 34 --keyint -1 --scd 0 --film-grain 30 --film-grain-denoise 0`
- Codec-only fast_eval: `seg=0.00501456 pose=0.08806656 size=938.8KB score=2.0800`
- + even `dxyr`: `score=1.4539`
- + even `ab`: `score=1.2839`
- + odd `ab` pass 1 (gated test): `score=1.2779`

Native-path takeaway:

- PSYEX without a tuned `--fgs-table` and full parameter adaptation did not beat current branches.
- Older native AC build is competitive and sidecar-friendly, but current gated score (`1.2779`) still trails existing best integrated pipeline (`~1.12`).

## Source Builds + Benchmarks (MinGW)

Built from source:

- `third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe`
  - Version: `SVT-AV1-PSY v2.3.0-B`
  - Verified flags include: `--fgs-table`, `--psy-rd`, `--spy-rd`, `--sharpness`
- `third_party/svt/src/SVT-AV1-gitlab-mainline/Bin/Release/SvtAv1EncApp.exe`
  - Version: `SVT-AV1 b486d83 (release)`
  - Built from canonical GitLab repository (`https://gitlab.com/AOMediaCodec/SVT-AV1`)

Benchmark setup:

- Input: `native_cases/input_582x436_10b.y4m`
- Decode path for eval: bicubic upscale + `unsharp=5:5:2.0:5:5:0.0`
- Common encoder knobs: `--preset 1 --keyint -1 --scd 0 --film-grain 30 --film-grain-denoise 0`

Results:

1. `native_cases/psy_src_v230b_p1_crf34_fg30_dn0_keyinf`
   - `seg=0.00451202 pose=0.07327494 size=2136.4KB score=2.7639`
2. `native_cases/psy_src_v230b_p1_crf35_fg30_dn0_keyinf`
   - `seg=0.00458115 pose=0.08042954 size=1891.7KB score=2.6448`
3. `native_cases/mainline_src_p1_crf34_fg30_dn0_keyinf`
   - `seg=0.00565014 pose=0.12175018 size=916.0KB score=2.2930`
4. `native_cases/mainline_src_p1_crf35_fg30_dn0_keyinf`
   - `seg=0.00587967 pose=0.13091047 size=854.9KB score=2.3150`

Additional psy-only sanity checks (source-built `v2.3.0-B`):

- `native_cases/psy_src_v230b_p1_crf35_psyrd2_sharp3`
  - Settings delta: `--psy-rd 2.0 --sharpness 3`
  - `seg=0.00476940 pose=0.05438989 size=3491.9KB score=3.5954`
- `native_cases/psy_src_v230b_p1_crf36_psyrd2_sharp3`
  - Settings delta: `--psy-rd 2.0 --sharpness 3`
  - `seg=0.00479852 pose=0.06199022 size=3025.5KB score=3.3301`

Source-build takeaway:

- None of the newly built binaries beat the previously tested prebuilt AC v2.1.2 branch (`2.0800` codec-only).
- Source-built psy v2.3.0-B is strongly negative in this operating region due very large archive sizes.
- Enabling psy-oriented knobs (`--psy-rd`, higher `--sharpness`) made bitrate explode and was even worse.
- Source-built mainline is closer, but still clearly behind both AC v2.1.2 and the current best REN+sidecar pipeline (`~1.12`).

## Custom `--fgs-table` Sweep (native psy)

Motivation:

- Explicitly test the "custom film grain table" hypothesis (`--fgs-table`) that might preserve pose cues while reducing bitrate.

Implementation:

- Added reproducible sweep script: `tools/run_fgs_table_sweep.py`
- Generated table variants into `native_cases/fgs_tables/`:
  - `example.tbl` (source example)
  - `low.tbl`
  - `high.tbl`
  - `darkboost.tbl` (extra low-luma emphasis)
  - `chroma.tbl` (explicit chroma points)
- Results captured in `results_fgs.csv`
- Evaluation path unchanged: decode with bicubic upscale + unsharp, then `fast_eval`

### Source-built `SVT-AV1-PSY v2.3.0-B` (`--preset 1 --crf 35 --keyint -1 --scd 0`)

Baseline:

- `fgs_psy_src_baseline_crf35_fg30_dn0` (no table, `--film-grain 30`) -> **2.6448**

FGS table variants (`--film-grain 0 --film-grain-denoise 0 --fgs-table ...`):

- `example.tbl` -> `3.4681`
- `low.tbl` -> `3.5143`
- `high.tbl` -> `3.3784`
- `darkboost.tbl` -> `3.2103` (least-bad table variant)
- `chroma.tbl` -> `3.4729`

Observed pattern:

- SegNet term improved slightly vs baseline (lower seg distortion).
- Pose term regressed severely (large pose distortion increase), dominating total score.
- Archive size dropped modestly (about `1767KB` vs `1892KB`), but not enough to offset pose damage.

### Prebuilt `SVT-AV1-PSYEX v3.0.2-B` sanity checks

- `fgs_psyex_prebuilt_baseline_crf35_fg30_dn0` -> `4.6273`
- `fgs_psyex_prebuilt_example_crf35_fg0_dn0` -> `4.7592`

Conclusion:

- In this setup, custom `--fgs-table` is strongly negative.
- The hypothesis that deterministic synthetic grain can replace normal film-grain behavior for PoseNet did not transfer.
- Keep `--fgs-table` **off** for current optimization path unless a fundamentally different table-generation method is introduced.

