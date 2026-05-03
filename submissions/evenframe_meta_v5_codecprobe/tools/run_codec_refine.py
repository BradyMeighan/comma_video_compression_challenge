#!/usr/bin/env python
"""
Focused refinement sweep around the best report-style codec family.
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUB = Path(__file__).resolve().parents[1]
CASES_DIR = SUB / "cases_refine"
VIDEO = ROOT / "videos" / "0.mkv"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
FAST_EVAL_MODULE = "submissions.evenframe_meta_v4_crf34.fast_eval"


@dataclass(frozen=True)
class CodecCase:
    name: str
    scale_w: int
    scale_h: int
    pix_fmt: str
    preset: int
    crf: int
    svt_params: str
    decode_filter: str


CASES = [
    CodecCase(
        name="base_8bit_crf34_fg30_dn0_keyinf_bicubic_us2",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf33_fg30_dn0_keyinf_bicubic_us2",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=33,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf35_fg30_dn0_keyinf_bicubic_us2",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=35,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf34_fg26_dn0_keyinf_bicubic_us2",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=26:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf34_fg22_dn0_keyinf_bicubic_us2",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=22:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf34_fg30_dn0_k180_bicubic_us2",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=180:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:2.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf34_fg30_dn0_keyinf_bicubic_us15",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:1.5:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf34_fg30_dn0_keyinf_bicubic_us10",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=bicubic,unsharp=5:5:1.0:5:5:0.0",
    ),
    CodecCase(
        name="8bit_crf34_fg30_dn0_keyinf_lanczos_us045",
        scale_w=582, scale_h=436, pix_fmt="yuv420p", preset=1, crf=34,
        svt_params="film-grain=30:film-grain-denoise=0:keyint=-1:scd=0",
        decode_filter="scale=1164:874:flags=lanczos,unsharp=5:5:0.45:5:5:0.0",
    ),
]


def run(cmd: list[str], cwd: Path) -> None:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")


def run_capture(cmd: list[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return (p.stdout or "") + "\n" + (p.stderr or "")


def parse_fast_eval(text: str) -> dict[str, float]:
    m = re.search(
        r"seg=([0-9.]+)\s+pose=([0-9.]+)\s+size=([0-9.]+)KB.*?100s=([0-9.]+)\s+sqrtp=([0-9.]+)\s+25r=([0-9.]+)\s+score=([0-9.]+)",
        text,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError(f"Could not parse fast_eval output:\n{text}")
    seg, pose, size_kb, s100, sqrtp, r25, score = map(float, m.groups())
    return {
        "seg": seg,
        "pose": pose,
        "size_kb": size_kb,
        "s100": s100,
        "sqrtp": sqrtp,
        "r25": r25,
        "score": score,
    }


def main() -> None:
    if not VIDEO.exists():
        raise FileNotFoundError(VIDEO)
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float | int]] = []

    for idx, case in enumerate(CASES, start=1):
        print(f"\n[{idx}/{len(CASES)}] {case.name}", flush=True)
        case_dir = CASES_DIR / case.name
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        mkv = case_dir / "0.mkv"
        raw = case_dir / "0.raw"
        zpath = case_dir / "archive.zip"
        t0 = time.time()

        run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(VIDEO),
                "-vf", f"scale={case.scale_w}:{case.scale_h}:flags=lanczos",
                "-pix_fmt", case.pix_fmt,
                "-c:v", "libsvtav1",
                "-preset", str(case.preset),
                "-crf", str(case.crf),
                "-svtav1-params", case.svt_params,
                str(mkv),
            ],
            ROOT,
        )

        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.write(mkv, "0.mkv")
        archive_bytes = zpath.stat().st_size

        run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(mkv),
                "-vf", case.decode_filter,
                "-pix_fmt", "rgb24",
                "-f", "rawvideo", str(raw),
            ],
            ROOT,
        )

        out = run_capture([str(PY), "-m", FAST_EVAL_MODULE, str(raw), str(archive_bytes)], ROOT)
        metrics = parse_fast_eval(out)
        elapsed = time.time() - t0

        print(
            f"  score={metrics['score']:.4f} seg={metrics['seg']:.8f} "
            f"pose={metrics['pose']:.8f} zip={archive_bytes/1024:.1f}KB "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        rows.append(
            {
                "name": case.name,
                "scale_w": case.scale_w,
                "scale_h": case.scale_h,
                "pix_fmt": case.pix_fmt,
                "preset": case.preset,
                "crf": case.crf,
                "svt_params": case.svt_params,
                "decode_filter": case.decode_filter,
                "archive_bytes": archive_bytes,
                "seg": metrics["seg"],
                "pose": metrics["pose"],
                "score": metrics["score"],
                "elapsed_s": round(elapsed, 2),
            }
        )

    csv_path = SUB / "results_refine.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== Ranked by score ===")
    for r in sorted(rows, key=lambda x: float(x["score"])):
        print(
            f"{r['score']:.4f}  {r['name']}  seg={float(r['seg']):.8f} "
            f"pose={float(r['pose']):.8f}  bytes={int(r['archive_bytes'])}"
        )
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()

