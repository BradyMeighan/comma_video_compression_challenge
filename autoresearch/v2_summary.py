"""Summarize all v2_*_results.csv into one ranked table.

Reads every CSV in OUTPUT_DIR matching v2_*_results.csv and prints:
  - per-experiment best score & delta vs X5 baseline (0.3010)
  - sorted leaderboard
"""
import csv, os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "autoresearch/sidecar_results"))
X5_BASELINE = 0.3010   # current best from prior runs

def main():
    rows = []
    for csv_path in sorted(OUTPUT_DIR.glob("v2_*_results.csv")):
        try:
            with open(csv_path, 'r') as f:
                r = csv.DictReader(f)
                for row in r:
                    spec = row.get('spec') or row.get('scheme') or csv_path.stem
                    score = row.get('score')
                    delta = row.get('delta') or row.get('delta_vs_bz2') or '0'
                    sb = row.get('sb_total') or row.get('compressed_bytes') or '?'
                    if score is None:
                        continue
                    try:
                        rows.append({
                            'file': csv_path.name,
                            'spec': spec,
                            'sb': sb,
                            'score': float(score),
                            'delta': float(delta),
                        })
                    except ValueError:
                        pass
        except Exception as e:
            print(f"  ! could not read {csv_path}: {e}")

    if not rows:
        print("[v2_summary] no results yet")
        return

    rows.sort(key=lambda r: r['score'])
    print(f"\n=== v2 LEADERBOARD (X5 baseline = {X5_BASELINE:.4f}) ===")
    print(f"{'rank':<5}{'score':<10}{'delta':<11}{'sb_bytes':<11}{'spec':<55}{'file'}")
    print("-" * 130)
    for i, row in enumerate(rows):
        delta_vs_x5 = row['score'] - X5_BASELINE
        marker = " <-- BEST" if i == 0 else ""
        print(f"{i+1:<5}{row['score']:<10.4f}{delta_vs_x5:+.4f}    {str(row['sb']):<11}"
              f"{row['spec']:<55}{row['file']}{marker}")
    print()

    best = rows[0]
    if best['score'] < X5_BASELINE:
        print(f"NEW WINNER: {best['spec']} score={best['score']:.4f} "
              f"(delta {best['score']-X5_BASELINE:+.4f} vs X5)")
    else:
        print(f"No improvement over X5 ({X5_BASELINE:.4f}); best v2: "
              f"{best['spec']} = {best['score']:.4f}")


if __name__ == "__main__":
    main()
