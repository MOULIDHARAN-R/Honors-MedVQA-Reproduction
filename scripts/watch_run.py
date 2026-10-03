"""
python -m scripts.watch_run --run run2 --split 1            # one-off status
python -m scripts.watch_run --run run2 --split 1 --follow   # keeps printing new epochs (Ctrl+C to stop)

Reads checkpoints/<run>/split_<k>/history.json. It never touches training.
Flags are HEURISTICS (the first run did not log gradient norms):
  LOSS+   train loss is > 0.5 above its best value so far          (old run: 1.77 -> 2.94)
  ACC-    test accuracy is > 0.05 below its best value so far      (old run: 0.53 -> 0.20)
  GN-MAX  grad-norm max  > 3x the median of earlier epochs' max
  GN-MEAN grad-norm mean > 3x the median of earlier epochs' mean
  >> COLLAPSE? when ACC- and LOSS+ occur together: stop with Ctrl+C and do not resume.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOTAL_EPOCHS = 50


def flags_for(h, i):
    r, prev = h[i], h[:i]
    out = []
    if prev:
        if r["train_loss"] > min(p["train_loss"] for p in prev) + 0.5:
            out.append("LOSS+")
        if r["test_acc"] < max(p["test_acc"] for p in prev) - 0.05:
            out.append("ACC-")
    if len(prev) >= 2 and "grad_norm_max" in r:
        if r["grad_norm_max"] > 3 * statistics.median(p["grad_norm_max"] for p in prev):
            out.append("GN-MAX")
        if r["grad_norm_mean"] > 3 * statistics.median(p["grad_norm_mean"] for p in prev):
            out.append("GN-MEAN")
    if "LOSS+" in out and "ACC-" in out:
        out.append(">> COLLAPSE? stop (Ctrl+C), do not resume")
    return out


def show(h, start):
    for i in range(start, len(h)):
        r = h[i]
        gn = f"{r['grad_norm_max']:7.2f} {r['grad_norm_mean']:6.2f}" if "grad_norm_max" in r else "      -      -"
        print(f"{r['epoch']:5d} {r['train_loss']:7.3f} {r['test_acc']:7.4f} {r.get('strict_acc', float('nan')):7.4f} "
              f"{gn}  {' '.join(flags_for(h, i))}")
    if len(h) > start:
        mean_t = sum(r["duration_sec"] for r in h) / len(h)
        left = max(0, TOTAL_EPOCHS - h[-1]["epoch"])
        print(f"      done {h[-1]['epoch']}/{TOTAL_EPOCHS} | {mean_t:.0f}s/epoch | about {left * mean_t / 3600:.1f} h left")


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None          # missing or half-written; try again


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run2")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--history", default=None)
    a = ap.parse_args()
    path = Path(a.history) if a.history else PROJECT_ROOT / "checkpoints" / a.run / f"split_{a.split}" / "history.json"

    print("epoch    loss     acc  strict  gn_max gn_mean  flags")
    seen = 0
    while True:
        h = load(path)
        if h and len(h) > seen:
            show(h, seen)
            seen = len(h)
            if h[-1]["epoch"] >= TOTAL_EPOCHS:
                print("finished.")
                return
        if not a.follow:
            if h is None:
                print(f"no readable history at {path}")
            return
        time.sleep(60)


if __name__ == "__main__":
    main()