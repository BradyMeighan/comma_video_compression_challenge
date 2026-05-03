#!/usr/bin/env python
"""
Run all enhanced REN training experiments sequentially on a base checkpoint.
Each experiment resumes from the same checkpoint and tries a different objective.

Usage:
  python sweep_ren_enhanced.py --base submissions/evenframe_meta_v4_crf34/archive/ren_seg_v3_ch64.pt --channels 64
  python sweep_ren_enhanced.py --base <ckpt> --channels 48 --epochs 500
"""
import argparse, subprocess, sys, time, os
from pathlib import Path

ROOT = Path(__file__).parent


# (name, extra_args)
EXPERIMENTS = [
    ('qat',      ['--qat']),
    ('ema',      ['--ema']),
    ('hard',     ['--hard-oversample']),
    ('margin',   ['--margin-anneal', '--margin-target', '0.5']),
    ('noise',    ['--input-noise', '1.0']),
    ('all',      ['--qat', '--margin-anneal', '--hard-oversample', '--ema', '--input-noise', '0.5']),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=str, required=True, help='Base checkpoint to resume from')
    ap.add_argument('--channels', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=200)
    ap.add_argument('--skip', type=str, default='', help='Comma-separated experiment names to skip')
    ap.add_argument('--only', type=str, default='', help='Comma-separated experiment names to run (overrides skip)')
    args = ap.parse_args()

    base_path = Path(args.base)
    if not base_path.exists():
        print(f"ERROR: base checkpoint not found: {base_path}")
        sys.exit(1)

    skip = set(args.skip.split(',')) if args.skip else set()
    only = set(args.only.split(',')) if args.only else None

    experiments = EXPERIMENTS
    if only:
        experiments = [(n, a) for n, a in experiments if n in only]
    else:
        experiments = [(n, a) for n, a in experiments if n not in skip]

    print(f"Base checkpoint: {base_path}")
    print(f"Channels: {args.channels}, epochs: {args.epochs}")
    print(f"Running {len(experiments)} experiments: {[n for n, _ in experiments]}\n")

    total_start = time.time()
    results = []

    for name, extra_args in experiments:
        tag = f'_ch{args.channels}_{name}'
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT: {name} (tag={tag})")
        print(f"{'='*70}", flush=True)

        cmd = [
            sys.executable, 'train_ren_enhanced.py',
            '--resume', str(base_path),
            '--channels', str(args.channels),
            '--epochs', str(args.epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--patience', str(args.patience),
            '--tag', tag,
        ] + extra_args

        t0 = time.time()
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(cmd, cwd=str(ROOT), env=env)
        elapsed = time.time() - t0

        status = 'OK' if result.returncode == 0 else f'FAIL({result.returncode})'
        print(f"\n  {name}: {elapsed:.0f}s ({status})")
        results.append({'name': name, 'time': elapsed, 'status': status, 'tag': tag})

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE ({(time.time()-total_start)/60:.0f} min)")
    print(f"{'='*70}")
    for r in results:
        print(f"  {r['name']:8s}  {r['time']:6.0f}s  {r['status']}")

    print(f"\nSaved checkpoints in submissions/evenframe_meta_v4_crf34/archive/:")
    for r in results:
        print(f"  ren_seg_v3{r['tag']}.pt")
        if 'ema' in r['name'] or r['name'] == 'all':
            print(f"  ren_seg_v3{r['tag']}_ema.pt")

    print(f"\nTo see which won, look at the 'Final gain' line at the bottom of each experiment's output.")


if __name__ == '__main__':
    main()
