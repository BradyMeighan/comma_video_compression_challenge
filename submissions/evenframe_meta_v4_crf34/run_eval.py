#!/usr/bin/env python
"""
Full pipeline: compress → inflate → evaluate. One command.
Usage: python -m submissions.av1_repro.run_eval
"""
import subprocess, sys, os, zipfile, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent

def run(cmd, **kw):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, **kw)
    print(r.stdout)
    if r.stderr:
        # Filter out just the results if present
        for line in r.stderr.split('\n'):
            if 'Distortion' in line or 'score' in line.lower() or 'size' in line or 'Rate' in line:
                print(line)
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-500:]}")
    return r

def main():
    t0 = time.time()

    # Step 1: Compress
    print("=" * 60)
    print("STEP 1: Compress (SVT-AV1)")
    print("=" * 60)
    run([sys.executable, '-m', 'submissions.av1_repro.compress'], cwd=str(ROOT))

    # Step 2: Inflate
    print("\n" + "=" * 60)
    print("STEP 2: Inflate (decode + Lanczos upscale + unsharp)")
    print("=" * 60)
    archive_dir = HERE / 'archive'
    inflated_dir = HERE / 'inflated'
    inflated_dir.mkdir(exist_ok=True)

    # Unzip
    with zipfile.ZipFile(HERE / 'archive.zip', 'r') as zf:
        zf.extractall(archive_dir)

    run([sys.executable, '-m', 'submissions.av1_repro.inflate',
         str(archive_dir / '0.mkv'), str(inflated_dir / '0.raw')], cwd=str(ROOT))

    # Step 3: Evaluate
    print("\n" + "=" * 60)
    print("STEP 3: Evaluate")
    print("=" * 60)
    r = run([sys.executable, 'evaluate.py',
         '--submission-dir', str(HERE),
         '--uncompressed-dir', str(ROOT / 'videos'),
         '--report', str(HERE / 'report.txt'),
         '--video-names-file', str(ROOT / 'public_test_video_names.txt'),
         '--device', 'cpu'], cwd=str(ROOT))

    print(f"\nTotal time: {time.time()-t0:.1f}s")

    # Print report
    report = HERE / 'report.txt'
    if report.exists():
        print("\n" + report.read_text(encoding='utf-8', errors='replace'))

if __name__ == '__main__':
    main()
