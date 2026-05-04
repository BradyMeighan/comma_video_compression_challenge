# vibe_coder_final_boss

Final score: **0.22878** on the public test set (600 pairs).

```
  PoseNet distortion : 0.00049457
  SegNet  distortion : 0.00027173
  archive.zip size   : 197,160 bytes
  Compression rate   : 0.00525    (= 197,160 / 37,545,489)
  Final score        : 100*seg + sqrt(10*pose) + 25*rate = 0.22878
```

Down from the repo baseline (`baseline_fast` ≈ 4.39), about a 19× reduction.

## Archive contents

`archive.zip` is a single `ZIP_STORED` entry named `p`:

```
[u32 mask_size][mask_bytes]            # 135,120 B  lossless tiled range-coded masks
[u32 pose_size][pose_bytes]            #   2,310 B  per-dim N-bit packed + brotli
[u32 model_size][model_bytes]          #  57,238 B  flat-FP4 generator weights + brotli
[u32 sidecar_size][sidecar_bytes]      #   2,376 B  lzma'd grouped bitpack patches + warps
                                       # ----------
                                       # 197,044 B  payload + ~116 B zip envelope = 197,160 B
```

`archive.zip` itself is hosted on the PR (see the upload link in the PR body) and is not checked into the repo.

## Decoder

```
bash inflate.sh <data_dir> <out_dir> <file_list>
```

Reads `<data_dir>/p`, generates `<out_dir>/0.raw` (uint8 RGB frames). Self-contained, no `autoresearch/` dependencies. Requirements:

- `g++` / `clang++` (compiles `range_mask_codec.cpp` on first run)
- python with `torch`, `brotli`, `einops`, `numpy`, `tqdm` (everything except `brotli` is in the upstream `pyproject.toml`; `inflate.sh` pip-installs `brotli` on the fly if missing)

Decode pipeline (`inflate.py`):

1. Range-decode 22 mask tiles, invert per-tile flips, transpose to `(600, 384, 512)` masks.
2. Bit-unpack pose (per-dim N-bit quant) + brotli decode.
3. Brotli + flat-FP4 decode the H3 generator weights, load into `GeneratorPoseLR`.
4. xz-decode + bitpack-decode the sidecar; apply mask edits and pose deltas in place.
5. Generator forward, `F.interpolate` to `(874, 1164)`, uint8 RGB.
6. Apply per-pair int8 (qx, qy) f1 warps to the upsampled output.

GPU is auto-detected (`torch.cuda.is_available()`). On a T4 the full inflate takes about 60-90 seconds wall time; on CPU it takes 4-6 minutes.

## What's in here

| File | Purpose |
| --- | --- |
| `inflate.sh` / `inflate.py` | decoder entry point |
| `model.py` | self-contained H3 generator (`GeneratorPoseLR`, FP4 helpers, constants) |
| `sidecar.py` | sidecar bitpack decode + apply (mask edits, pose deltas, f1 warps) |
| `range_mask_codec.cpp` | tiny binary arithmetic coder for the segmentation masks |
| `flat_fp4.py` / `schema_h3.py` | flat FP4 model packing |
| `test_on_colab.ipynb` | end-to-end T4 validation notebook |
| `README.md` | this file |

## Reproducing compression

The full encoder pipeline (autoresearch loop, sidecar search, codec sweeps) is too large to ship in the PR. It lives in our writeup repo at <https://github.com/BradyMeighan/vibe-coder-final-boss>. The full writeup of method, ablations, and citations: <https://vibe-coder-final-boss.pages.dev>.
