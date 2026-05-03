#!/usr/bin/env python
"""
Sweep REN training across multiple channel widths with early stopping.
Saves each model with a channel-specific tag, reports results.

Usage:
  python sweep_ren_channels.py
  python sweep_ren_channels.py --channels 16,24,32,48,64,80,96,128
  python sweep_ren_channels.py --patience 150 --max-epochs 3000
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--channels', type=str, default='16,24,32,48,64,80,96,128',
                    help='Comma-separated channel widths')
    ap.add_argument('--max-epochs', type=int, default=3000,
                    help='Max epochs per channel width')
    ap.add_argument('--patience', type=int, default=150,
                    help='Early stop after N epochs of no improvement')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-3)
    args = ap.parse_args()

    channels = [int(c) for c in args.channels.split(',')]
    print(f"Sweeping channels: {channels}")
    print(f"Max epochs: {args.max_epochs}, patience: {args.patience}")
    print(f"Batch size: {args.batch_size}, LR: {args.lr}")
    print()

    results = []
    total_start = time.time()

    for ch in channels:
        tag = f'_ch{ch}'
        print(f"\n{'='*70}")
        print(f"  TRAINING CHANNELS={ch}")
        print(f"{'='*70}", flush=True)
        t0 = time.time()

        cmd = [
            sys.executable, 'train_ren_seg_v3.py',
            '--epochs', str(args.max_epochs),
            '--batch-size', str(args.batch_size),
            '--lr', str(args.lr),
            '--channels', str(ch),
            '--tag', tag,
            '--patience', str(args.patience),
        ]

        try:
            result = subprocess.run(cmd, cwd=str(ROOT), env={**__import__('os').environ, 'PYTHONIOENCODING': 'utf-8'})
            elapsed = time.time() - t0
            print(f"\n  ch={ch} done in {elapsed:.0f}s (exit={result.returncode})")
            results.append({'ch': ch, 'time': elapsed, 'exit': result.returncode})
        except Exception as e:
            print(f"  ch={ch} FAILED: {e}")
            results.append({'ch': ch, 'time': time.time() - t0, 'exit': -1})

    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE ({(time.time()-total_start)/60:.0f} min total)")
    print(f"{'='*70}")
    for r in results:
        print(f"  ch={r['ch']:3d}: {r['time']:5.0f}s, exit={r['exit']}")

    print(f"\nResults saved in submissions/evenframe_meta_v4_crf34/archive/ren_seg_v3_ch*.int8.bz2")
    print("Check each model's final output line for seg gain and model size.")


if __name__ == '__main__':
    main()
