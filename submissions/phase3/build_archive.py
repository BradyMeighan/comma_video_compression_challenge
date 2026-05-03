#!/usr/bin/env python
"""Build archive.zip from pre-computed frames for evaluation."""
import zipfile, os
from pathlib import Path

PHASE3 = Path('submissions/phase3')
DATA_DIR = PHASE3 / '0'
ARCHIVE = PHASE3 / 'archive.zip'

def main():
    with zipfile.ZipFile(ARCHIVE, 'w', zipfile.ZIP_STORED) as zf:
        for fname in ['frames_f0.npy', 'frames_f1.npy']:
            fpath = DATA_DIR / fname
            if fpath.exists():
                zf.write(fpath, f'0/{fname}')
                print(f"  Added {fname}: {fpath.stat().st_size/1e6:.1f} MB")

    size = ARCHIVE.stat().st_size
    rate = size / 37_545_489
    print(f"\narchive.zip: {size:,} bytes ({size/1e6:.1f} MB)")
    print(f"25*rate = {25*rate:.2f}")
    print(f"NOTE: This is uncompressed npy — real submission will use H.264")

if __name__ == '__main__':
    main()
