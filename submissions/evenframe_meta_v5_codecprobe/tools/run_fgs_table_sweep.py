from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROBE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
NATIVE_ROOT = PROBE_ROOT / "native_cases"
TABLE_ROOT = NATIVE_ROOT / "fgs_tables"
INPUT_Y4M = NATIVE_ROOT / "input_582x436_10b.y4m"
RESULTS_CSV = PROBE_ROOT / "results_fgs.csv"
DECODE_FILTER = "scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0"
ARCHIVE_ORIGINAL_SIZE = 37_545_489.0


@dataclass(frozen=True)
class Case:
    name: str
    encoder_rel: str
    crf: int
    film_grain: int
    film_grain_denoise: int
    fgs_table: str | None = None
    extra_args: tuple[str, ...] = ()


def _run(cmd: list[str], cwd: Path, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        check=False,
        capture_output=capture,
    )


def _format_table(
    y_points: list[tuple[int, int]],
    cb_points: list[tuple[int, int]],
    cr_points: list[tuple[int, int]],
    *,
    random_seed: int = 10956,
    ar_coeff_lag: int = 0,
    ar_coeff_shift: int = 6,
    grain_scale_shift: int = 0,
    scaling_shift: int = 8,
    chroma_scaling_from_luma: int = 1,
    overlap_flag: int = 1,
    cb_mult: int = 0,
    cb_luma_mult: int = 0,
    cb_offset: int = 0,
    cr_mult: int = 0,
    cr_luma_mult: int = 0,
    cr_offset: int = 0,
) -> str:
    y_flat = " ".join(f"{x} {y}" for x, y in y_points)
    cb_flat = " ".join(f"{x} {y}" for x, y in cb_points)
    cr_flat = " ".join(f"{x} {y}" for x, y in cr_points)

    # For lag=0, cY has zero coefficients, and cCb/cCr each carry one value.
    lines = [
        "filmgrn1",
        f"E 0 18446744073709551615 1 {random_seed} 1",
        (
            f"\tp {ar_coeff_lag} {ar_coeff_shift} {grain_scale_shift} {scaling_shift} "
            f"{chroma_scaling_from_luma} {overlap_flag} "
            f"{cb_mult} {cb_luma_mult} {cb_offset} "
            f"{cr_mult} {cr_luma_mult} {cr_offset}"
        ),
        f"\tsY {len(y_points)} {y_flat}".rstrip(),
        f"\tsCb {len(cb_points)} {cb_flat}".rstrip(),
        f"\tsCr {len(cr_points)} {cr_flat}".rstrip(),
        "\tcY",
        "\tcCb 0",
        "\tcCr 0",
    ]
    return "\n".join(lines) + "\n"


def write_fgs_tables() -> dict[str, Path]:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)

    x_axis = [0, 20, 39, 59, 78, 98, 118, 137, 157, 177, 196, 216, 235, 255]
    base_y = [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5]
    low_y = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3]
    high_y = [8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8]
    dark_boost_y = [10, 10, 9, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3]

    tables: dict[str, str] = {
        "example.tbl": _format_table(
            list(zip(x_axis, base_y)),
            [],
            [],
            chroma_scaling_from_luma=1,
        ),
        "low.tbl": _format_table(
            list(zip(x_axis, low_y)),
            [],
            [],
            chroma_scaling_from_luma=1,
        ),
        "high.tbl": _format_table(
            list(zip(x_axis, high_y)),
            [],
            [],
            chroma_scaling_from_luma=1,
        ),
        "darkboost.tbl": _format_table(
            list(zip(x_axis, dark_boost_y)),
            [],
            [],
            chroma_scaling_from_luma=1,
        ),
        "chroma.tbl": _format_table(
            list(zip(x_axis, base_y)),
            [(0, 3), (128, 3), (255, 3)],
            [(0, 3), (128, 3), (255, 3)],
            chroma_scaling_from_luma=0,
            cb_mult=128,
            cb_luma_mult=192,
            cb_offset=256,
            cr_mult=128,
            cr_luma_mult=192,
            cr_offset=256,
        ),
    }

    out: dict[str, Path] = {}
    for name, text in tables.items():
        p = TABLE_ROOT / name
        p.write_text(text, encoding="utf-8")
        out[name] = p
    return out


def parse_fast_eval(stdout: str) -> dict[str, float]:
    match = re.search(
        r"seg=([0-9.]+)\s+pose=([0-9.]+)\s+size=([0-9.]+)KB.*?score=([0-9.]+)",
        stdout,
        flags=re.DOTALL,
    )
    if not match:
        raise RuntimeError("Could not parse fast_eval output.")
    seg = float(match.group(1))
    pose = float(match.group(2))
    size_kb = float(match.group(3))
    score = float(match.group(4))
    return {"seg": seg, "pose": pose, "size_kb": size_kb, "score": score}


def run_case(case: Case) -> dict[str, float | str]:
    case_dir = NATIVE_ROOT / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    ivf = case_dir / "0.ivf"
    raw = case_dir / "0.raw"
    archive_zip = case_dir / "archive.zip"

    for p in (ivf, raw, archive_zip):
        if p.exists():
            p.unlink()

    encoder = REPO_ROOT / case.encoder_rel
    if not encoder.exists():
        raise FileNotFoundError(f"Missing encoder binary: {encoder}")

    encode_cmd = [
        str(encoder),
        "-i",
        str(INPUT_Y4M),
        "-b",
        str(ivf),
        "--input-depth",
        "10",
        "--width",
        "582",
        "--height",
        "436",
        "--fps-num",
        "20",
        "--fps-denom",
        "1",
        "--preset",
        "1",
        "--crf",
        str(case.crf),
        "--keyint",
        "-1",
        "--scd",
        "0",
        "--film-grain",
        str(case.film_grain),
        "--film-grain-denoise",
        str(case.film_grain_denoise),
    ]
    if case.fgs_table is not None:
        encode_cmd += ["--fgs-table", str(TABLE_ROOT / case.fgs_table)]
    encode_cmd += list(case.extra_args)

    t0 = time.perf_counter()
    enc_proc = _run(encode_cmd, cwd=REPO_ROOT, capture=True)
    print(enc_proc.stdout)
    print(enc_proc.stderr)
    if enc_proc.returncode != 0:
        raise RuntimeError(f"Encode failed for {case.name}")

    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(ivf, arcname="0.ivf")
    archive_bytes = archive_zip.stat().st_size

    decode_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(ivf),
        "-vf",
        DECODE_FILTER,
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        str(raw),
    ]
    dec_proc = _run(decode_cmd, cwd=REPO_ROOT, capture=True)
    print(dec_proc.stdout)
    print(dec_proc.stderr)
    if dec_proc.returncode != 0:
        raise RuntimeError(f"Decode failed for {case.name}")

    eval_cmd = [
        sys.executable,
        "-m",
        "submissions.av1_repro.fast_eval",
        str(raw),
        str(archive_bytes),
    ]
    eval_proc = _run(eval_cmd, cwd=REPO_ROOT, capture=True)
    print(eval_proc.stdout)
    print(eval_proc.stderr)
    if eval_proc.returncode != 0:
        raise RuntimeError(f"fast_eval failed for {case.name}")

    parsed = parse_fast_eval(eval_proc.stdout)
    elapsed_s = time.perf_counter() - t0
    rate_term = 25.0 * archive_bytes / ARCHIVE_ORIGINAL_SIZE

    if raw.exists():
        raw.unlink()

    return {
        "name": case.name,
        "encoder": case.encoder_rel,
        "crf": case.crf,
        "film_grain": case.film_grain,
        "film_grain_denoise": case.film_grain_denoise,
        "fgs_table": case.fgs_table or "",
        "archive_kb": round(archive_bytes / 1024.0, 1),
        "rate_term": round(rate_term, 4),
        "seg": parsed["seg"],
        "pose": parsed["pose"],
        "score": parsed["score"],
        "elapsed_s": round(elapsed_s, 1),
    }


def main() -> None:
    if not INPUT_Y4M.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_Y4M}")

    write_fgs_tables()

    cases = [
        # Source-built psy v2.3.0-B baseline and fgs variants.
        Case(
            name="fgs_psy_src_baseline_crf35_fg30_dn0",
            encoder_rel="third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe",
            crf=35,
            film_grain=30,
            film_grain_denoise=0,
        ),
        Case(
            name="fgs_psy_src_example_crf35_fg0_dn0",
            encoder_rel="third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe",
            crf=35,
            film_grain=0,
            film_grain_denoise=0,
            fgs_table="example.tbl",
        ),
        Case(
            name="fgs_psy_src_low_crf35_fg0_dn0",
            encoder_rel="third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe",
            crf=35,
            film_grain=0,
            film_grain_denoise=0,
            fgs_table="low.tbl",
        ),
        Case(
            name="fgs_psy_src_high_crf35_fg0_dn0",
            encoder_rel="third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe",
            crf=35,
            film_grain=0,
            film_grain_denoise=0,
            fgs_table="high.tbl",
        ),
        Case(
            name="fgs_psy_src_darkboost_crf35_fg0_dn0",
            encoder_rel="third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe",
            crf=35,
            film_grain=0,
            film_grain_denoise=0,
            fgs_table="darkboost.tbl",
        ),
        Case(
            name="fgs_psy_src_chroma_crf35_fg0_dn0",
            encoder_rel="third_party/svt/src/svt-av1-psy-v2.3.0-B/Bin/Release/SvtAv1EncApp.exe",
            crf=35,
            film_grain=0,
            film_grain_denoise=0,
            fgs_table="chroma.tbl",
        ),
        # Prebuilt psyex sanity check with the same table.
        Case(
            name="fgs_psyex_prebuilt_baseline_crf35_fg30_dn0",
            encoder_rel="third_party/svt/psyex-v3.0.2B-win64/SvtAv1EncApp.exe",
            crf=35,
            film_grain=30,
            film_grain_denoise=0,
        ),
        Case(
            name="fgs_psyex_prebuilt_example_crf35_fg0_dn0",
            encoder_rel="third_party/svt/psyex-v3.0.2B-win64/SvtAv1EncApp.exe",
            crf=35,
            film_grain=0,
            film_grain_denoise=0,
            fgs_table="example.tbl",
        ),
    ]

    rows: list[dict[str, float | str]] = []
    for case in cases:
        print(f"\n=== Running {case.name} ===")
        result = run_case(case)
        print(f"RESULT {result}")
        rows.append(result)

    fieldnames = [
        "name",
        "encoder",
        "crf",
        "film_grain",
        "film_grain_denoise",
        "fgs_table",
        "archive_kb",
        "rate_term",
        "seg",
        "pose",
        "score",
        "elapsed_s",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote results to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()

