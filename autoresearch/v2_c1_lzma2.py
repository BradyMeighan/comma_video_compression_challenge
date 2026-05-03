"""C1: Solid LZMA2 archive container vs current bz2-per-stream.

Method: pack mask+pose+model+sidecar bytes into a single concatenated stream,
then test multiple cross-stream compression schemes:
  - bz2 (current — per-stream then concat)
  - lzma2 (xz preset=9e, various lc/pb settings)
  - 7z LZMA2 solid mode
  - zstd long-range
  - zopfli on a tar wrapper

For each: report total bytes + score impact.

NOTE: We don't actually need to build the .zip archive. We just need to know
how many bytes the COMPRESSED PAYLOAD takes. Score formula uses the byte count
of the post-compression sidecar/model bytes, not zip metadata bloat.

Actually — the score uses (MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes)
where mask+pose are FIXED constants from the dataset. Only sidecar_bytes can change.
So all we need to test is: can a different compression scheme make the *combined*
sidecar (mask + RGB + any other) smaller than bz2-per-stream?
"""
import sys, os, pickle, bz2, lzma, zlib, struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import v2_shared

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))


def serialize_combined_v5(mask_blocks, mask_pixels_extra, rgb_patches):
    """Build a single concatenated raw stream: mask blocks + extra pixel mask + RGB.
    Used for testing cross-stream compression."""
    parts = []
    # Magic + format version
    parts.append(b'V5\x00\x01')
    # Section 1: mask blocks (5B per: u16 x, u16 y, u8 c)
    parts.append(struct.pack("<I", len(mask_blocks)))
    for pi in sorted(mask_blocks.keys()):
        ps = mask_blocks[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for tup in ps:
            x, y, c = tup[0], tup[1], tup[2]
            parts.append(struct.pack("<HHB", x, y, c))
    # Section 2: extra pixel mask flips (5B per: u16 x, u16 y, u8 c)
    parts.append(struct.pack("<I", len(mask_pixels_extra)))
    for pi in sorted(mask_pixels_extra.keys()):
        ps = mask_pixels_extra[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c) in ps:
            parts.append(struct.pack("<HHB", x, y, c))
    # Section 3: channel-only RGB patches (6B per: u16 x, u16 y, u8 c, i8 d)
    parts.append(struct.pack("<I", len(rgb_patches)))
    for pi in sorted(rgb_patches.keys()):
        ps = rgb_patches[pi]
        parts.append(struct.pack("<HH", pi, len(ps)))
        for (x, y, c, d) in ps:
            parts.append(struct.pack("<HHBb", x, y, c, d))
    return b''.join(parts)


def main():
    import csv, time
    import zstandard as zstd
    import zopfli.zlib as zopfli_zlib

    # Load all best-known sidecars (need all of: mask blocks from X2/X5, RGB from X5)
    # We don't have the X5 patches saved separately — but we have the baseline
    # (mask K=1 pixel + channel-only RGB). Use that as the test bed.
    with open(OUTPUT_DIR / "baseline_patches.pkl", 'rb') as f:
        bp = pickle.load(f)
    mask_patches = bp['mask_patches']  # dict: pair_i -> [(x, y, c), ...]
    rgb_patches = bp['rgb_patches']    # dict: pair_i -> [(x, y, c, d), ...]
    score_bl = bp['score']
    sb_total_bl = bp['sb_total_bz2']
    model_bytes = bp['model_bytes']
    seg_dist_bl = bp['seg_dist']
    pose_dist_bl = bp['pose_dist']

    # Build "combined" stream from baseline: mask pixels treated as section 1 (no blocks)
    combined_raw = serialize_combined_v5({}, mask_patches, rgb_patches)
    print(f"=== C1: combined-stream compression test ===")
    print(f"Raw combined stream: {len(combined_raw)}B")
    print(f"Baseline (sum of per-stream bz2): {sb_total_bl}B / score {score_bl:.4f}\n")

    schemes = []

    def add(name, compressed_bytes):
        from v2_shared import compose_score
        # Sidecar bytes = compressed_bytes (combined in single stream)
        full = compose_score(seg_dist_bl, pose_dist_bl, model_bytes, len(compressed_bytes))
        delta = full['score'] - score_bl
        schemes.append((name, len(compressed_bytes), full['score'], delta))
        print(f"  {name:<35s}: {len(compressed_bytes):>6d}B  score={full['score']:.4f}  delta={delta:+.4f}")

    # bz2 levels
    for lvl in [1, 5, 9]:
        t0 = time.time()
        c = bz2.compress(combined_raw, lvl)
        add(f"bz2_l{lvl}_combined", c)

    # zlib levels
    for lvl in [6, 9]:
        c = zlib.compress(combined_raw, lvl)
        add(f"zlib_l{lvl}_combined", c)

    # LZMA preset combinations
    for preset in [6, 9]:
        for ext in [None, lzma.PRESET_EXTREME]:
            ext_str = '_e' if ext else ''
            try:
                c = lzma.compress(combined_raw, format=lzma.FORMAT_XZ,
                                    preset=preset | (ext or 0))
                add(f"lzma_p{preset}{ext_str}_combined", c)
            except Exception as e:
                print(f"  lzma_p{preset}{ext_str} failed: {e}")

    # LZMA with custom filter chain (lc=2, pb=0 per Compass recommendation)
    try:
        filters = [{
            "id": lzma.FILTER_LZMA2,
            "preset": 9 | lzma.PRESET_EXTREME,
            "dict_size": 64 * 1024 * 1024,
            "lc": 2, "pb": 0, "lp": 0,
        }]
        c = lzma.compress(combined_raw, format=lzma.FORMAT_XZ, filters=filters,
                            check=lzma.CHECK_NONE)
        add("lzma_p9e_lc2_pb0_lp0", c)
    except Exception as e:
        print(f"  lzma custom filter failed: {e}")

    # Zstd levels
    for lvl in [9, 19, 22]:
        cctx = zstd.ZstdCompressor(level=lvl)
        add(f"zstd_l{lvl}_combined", cctx.compress(combined_raw))
    # Zstd with --long
    try:
        cctx = zstd.ZstdCompressor(level=22, write_content_size=False)
        params = zstd.ZstdCompressionParameters.from_level(22, source_size=len(combined_raw))
        cctx2 = zstd.ZstdCompressor(compression_params=params)
        add(f"zstd_l22_long", cctx2.compress(combined_raw))
    except Exception as e:
        print(f"  zstd long failed: {e}")

    # Zopfli (high-iter DEFLATE)
    try:
        c = zopfli_zlib.compress(combined_raw, numiterations=15)
        add("zopfli_n15", c)
    except Exception as e:
        print(f"  zopfli failed: {e}")

    # Save results
    schemes.sort(key=lambda x: x[1])
    print("\n=== SORTED BY SIZE (smaller = better) ===")
    for name, sz, sc, delta in schemes:
        print(f"  {name:<35s}: {sz:>6d}B  score={sc:.4f}  delta={delta:+.4f}")

    with open(OUTPUT_DIR / "v2_c1_lzma2_results.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["scheme", "compressed_bytes", "score", "delta_vs_bz2"])
        w.writerow(["baseline_per_stream_bz2", sb_total_bl, score_bl, 0])
        for name, sz, sc, delta in schemes:
            w.writerow([name, sz, sc, delta])

    print(f"\nResults: {OUTPUT_DIR / 'v2_c1_lzma2_results.csv'}")


if __name__ == "__main__":
    main()
