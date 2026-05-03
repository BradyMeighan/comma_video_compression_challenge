"""Audit v2 sidecar results and target gap.

This is intentionally read-only: it parses the CSV artifacts, ranks every
reported score field, and prints byte/pose tradeoffs for the live unified run.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from prepare import MASK_BYTES, POSE_BYTES, UNCOMPRESSED_SIZE


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "autoresearch" / "sidecar_results"
TARGET = 0.2899
MODEL_BYTES_E80 = 54078


def score_from_terms(seg_term: float, pose_term: float, sidecar_bytes: int,
                     model_bytes: int = MODEL_BYTES_E80) -> float:
    rate_term = 25 * (MASK_BYTES + POSE_BYTES + model_bytes + sidecar_bytes) / UNCOMPRESSED_SIZE
    return seg_term + pose_term + rate_term


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def collect_scores() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(OUT.glob("*results.csv")):
        for row in read_csv(path):
            name = (
                row.get("cfg") or row.get("spec") or row.get("scheme") or
                row.get("variant") or row.get("method") or path.stem
            )
            for field in ("score_lzma", "score_bz2", "score"):
                val = row.get(field)
                if not val:
                    continue
                try:
                    score = float(val)
                except ValueError:
                    continue
                bytes_field = (
                    row.get("sb_lzma") if field == "score_lzma" else None
                ) or row.get("sb_total") or row.get("sidecar_bytes") or row.get("compressed_bytes") or row.get("sb_bytes")
                try:
                    sb = int(float(bytes_field)) if bytes_field not in (None, "") else None
                except ValueError:
                    sb = None
                rows.append({
                    "file": path.name,
                    "name": name,
                    "score_field": field,
                    "score": score,
                    "sidecar_bytes": sb,
                    "row": row,
                })
    rows.sort(key=lambda r: float(r["score"]))
    return rows


def print_leaderboard(rows: list[dict[str, object]], n: int = 15) -> None:
    print("=== true sidecar leaderboard ===")
    print(f"{'rank':<5}{'score':>10}  {'gap':>9}  {'bytes':>7}  {'field':<11}  {'name':<34} file")
    print("-" * 110)
    for i, r in enumerate(rows[:n], 1):
        score = float(r["score"])
        sb = r["sidecar_bytes"]
        gap = score - TARGET
        print(
            f"{i:<5}{score:>10.6f}  {gap:>+9.6f}  "
            f"{str(sb) if sb is not None else '?':>7}  "
            f"{str(r['score_field']):<11}  {str(r['name'])[:34]:<34} {r['file']}"
        )
    print()


def audit_unified() -> None:
    path = OUT / "v2_unified_results.csv"
    if not path.exists():
        print("No v2_unified_results.csv found.")
        return
    rows = read_csv(path)
    parsed = []
    for row in rows:
        parsed.append({
            "cfg": row["cfg"],
            "score": float(row["score_lzma"]),
            "score_bz2": float(row["score_bz2"]),
            "sb_lzma": int(row["sb_lzma"]),
            "sb_rgb": int(row["sb_rgb"]),
            "seg_term": float(row["seg_term"]),
            "pose_term": float(row["pose_term"]),
        })
    parsed.sort(key=lambda r: r["score"])
    best = parsed[0]
    print("=== unified v2 RGB tier audit ===")
    print(
        f"best={best['cfg']} score={best['score']:.6f} "
        f"gap_to_{TARGET:.4f}={best['score'] - TARGET:+.6f}"
    )
    best_pose_dist = best["pose_term"] ** 2 / 10
    needed_pose_term = best["pose_term"] - (best["score"] - TARGET)
    needed_pose_dist = max(0.0, needed_pose_term) ** 2 / 10
    print(
        f"terms: seg={best['seg_term']:.6f} pose={best['pose_term']:.6f} "
        f"pose_dist={best_pose_dist:.9f} sidecar={best['sb_lzma']}B"
    )
    print(
        f"if bytes+seg stay fixed, pose_term must drop to {needed_pose_term:.6f}, "
        f"pose_dist to {needed_pose_dist:.9f} "
        f"({(1 - needed_pose_dist / best_pose_dist) * 100:.1f}% lower MSE)."
    )
    print()

    print(f"{'cfg':<26}{'score':>10}{'d_score':>10}{'sb':>7}{'d_bytes':>9}{'pose':>10}{'d_pose':>10}")
    for r in parsed:
        print(
            f"{r['cfg']:<26}{r['score']:>10.6f}"
            f"{r['score'] - best['score']:>+10.6f}"
            f"{r['sb_lzma']:>7}{r['sb_lzma'] - best['sb_lzma']:>+9}"
            f"{r['pose_term']:>10.6f}{r['pose_term'] - best['pose_term']:>+10.6f}"
        )
    print()

    print("byte tradeoff checks vs best:")
    for r in parsed[1:]:
        saved_by_pose = best["pose_term"] - r["pose_term"]
        paid_by_bytes = 25 * (r["sb_lzma"] - best["sb_lzma"]) / UNCOMPRESSED_SIZE
        print(
            f"  {r['cfg']}: pose improves {saved_by_pose:+.6f}, "
            f"rate changes {paid_by_bytes:+.6f}, net {r['score'] - best['score']:+.6f}"
        )
    print()


def audit_result_hygiene(rows: list[dict[str, object]]) -> None:
    files = {r["file"] for r in rows}
    print("=== hygiene flags ===")
    if "v2_unified_results.csv" in files:
        print("* v2_summary.py misses v2_unified_results.csv because it only matches v2_*_results.csv.")
    stale_docs = [
        ROOT / "SIDECAR_DEEP_RESEARCH.md",
        ROOT / "_deep_research_files" / "top10" / "SIDECAR_DEEP_RESEARCH.md",
        ROOT / "SIDECAR_PIPELINE.md",
    ]
    for doc in stale_docs:
        if doc.exists():
            txt = doc.read_text(errors="replace")
            if "0.3010" in txt or "0.2999" in txt:
                print(f"* stale score references likely remain in {doc.relative_to(ROOT)}.")
    print("* per-pair selector chooses lowest pose loss per pair, not score/net-benefit; worth an ablation.")
    print("* unified RGB tier sweep is sparse around the current winner; likely the highest-ROI next sweep.")
    print()


def main() -> None:
    rows = collect_scores()
    print_leaderboard(rows)
    audit_unified()
    audit_result_hygiene(rows)

    if rows:
        best = float(rows[0]["score"])
        print("=== bottom line ===")
        print(f"current best score={best:.6f}; need {best - TARGET:.6f} more to reach {TARGET:.4f}.")
        kb_score = 25 * 1024 / UNCOMPRESSED_SIZE
        print(f"1 KiB sidecar costs {kb_score:.6f} score, so new bytes must be very targeted.")


if __name__ == "__main__":
    main()
