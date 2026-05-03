"""Watch a directory for new .ckpt files and eval them as they appear.

Usage:
  WATCH_DIR=autoresearch/colab_run/3090_run \
  RESULTS_CSV=autoresearch/colab_run/3090_run/eval_log.csv \
  python autoresearch/eval_watcher.py
"""
import sys, os, time, csv
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("FULL_DATA", "1"); os.environ.setdefault("CONFIG", "B")

from prepare import evaluate, gpu_cleanup
from train import Generator, load_data_full

WATCH_DIR = Path(os.environ.get("WATCH_DIR", "autoresearch/colab_run/3090_run"))
RESULTS_CSV = Path(os.environ.get("RESULTS_CSV", str(WATCH_DIR / "eval_log.csv")))
POLL_SEC = int(os.environ.get("POLL_SEC", "60"))
STOP_FILE = WATCH_DIR / "STOP_WATCHER"


def main():
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    data = load_data_full(device)

    seen = set()
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            r = csv.DictReader(f)
            for row in r:
                seen.add(row['ckpt'])
    else:
        with open(RESULTS_CSV, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['ckpt', 'score', 'seg_term', 'pose_term', 'rate_term', 'eval_time'])

    # defensive: clear any stale STOP_WATCHER from previous run
    if STOP_FILE.exists():
        STOP_FILE.unlink()
    print(f"[watcher] watching {WATCH_DIR} every {POLL_SEC}s", flush=True)
    print(f"[watcher] write {STOP_FILE} to stop", flush=True)
    while True:
        if STOP_FILE.exists():
            print("[watcher] STOP_WATCHER seen, exiting", flush=True)
            STOP_FILE.unlink()
            break
        new = sorted([p for p in WATCH_DIR.glob("*.ckpt") if p.name not in seen])
        for p in new:
            try:
                gen = Generator().to(device)
                sd = torch.load(p, map_location=device, weights_only=True)
                gen.load_state_dict(sd, strict=False)
                t0 = time.time()
                result = evaluate(gen, data, device)
                eval_time = time.time() - t0
                msg = (f"[watcher] {p.name:<30} score={result['score']:.6f} "
                       f"seg={result['seg_term']:.4f} pose={result['pose_term']:.4f} "
                       f"rate={result['rate_term']:.4f} ({eval_time:.0f}s)")
                print(msg, flush=True)
                with open(RESULTS_CSV, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerow([p.name, result['score'], result['seg_term'],
                                 result['pose_term'], result['rate_term'], eval_time])
                seen.add(p.name)
                del gen; gpu_cleanup()
            except Exception as e:
                print(f"[watcher] FAIL on {p.name}: {e}", flush=True)
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
