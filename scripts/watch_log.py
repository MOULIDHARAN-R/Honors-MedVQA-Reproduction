"""
python -m scripts.watch_log --log experiments/diagnostic_qa_run1/logs/training.log
python -m scripts.watch_log --log <path to a training log> --follow      # keeps printing new epochs (Ctrl+C to stop)

Parses the per-epoch lines of the training log, e.g.
  Epoch 06/50 [167s] | lr 5.0e-05 | loss 1.5714 | acc 0.5703 (strict 0.5703, peak 0.5703) | gnorm max 17.41 mean 5.25 | ...
and prints a table with the same HEURISTIC flags as watch_run:
  LOSS+   train loss > 0.5 above its best value so far
  ACC-    test accuracy > 0.05 below its best value so far
  GN-MAX  grad-norm max  > 3x the median of earlier epochs' max
  GN-MEAN grad-norm mean > 3x the median of earlier epochs' mean
  >> COLLAPSE? when LOSS+ and ACC- occur together: stop (Ctrl+C), do not resume.
Read-only: it never touches training.
"""
import argparse
import re
import statistics
import time
from pathlib import Path

LINE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\[(\d+)s\]\s*\|\s*(?:lr\s+([0-9.eE+-]+)\s*\|\s*)?loss\s+([0-9.]+)\s*\|\s*"
    r"acc\s+([0-9.]+)\s*\(strict\s+([0-9.]+).*?\)\s*\|\s*gnorm max\s+([0-9.]+)\s+mean\s+([0-9.]+)")


def parse(path):
    rows = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    for m in LINE.finditer(text):
        ep, tot, dur, lr, loss, acc, strict, gmax, gmean = m.groups()
        rows[int(ep)] = {"epoch": int(ep), "total": int(tot), "dur": int(dur),
                         "lr": float(lr) if lr else float("nan"), "loss": float(loss),
                         "acc": float(acc), "strict": float(strict),
                         "gmax": float(gmax), "gmean": float(gmean)}
    return [rows[k] for k in sorted(rows)]      # one row per epoch, even if the log was resumed


def flags_for(h, i):
    r, prev = h[i], h[:i]
    out = []
    if prev:
        if r["loss"] > min(p["loss"] for p in prev) + 0.5:
            out.append("LOSS+")
        if r["acc"] < max(p["acc"] for p in prev) - 0.05:
            out.append("ACC-")
    if len(prev) >= 2:
        if r["gmax"] > 3 * statistics.median(p["gmax"] for p in prev):
            out.append("GN-MAX")
        if r["gmean"] > 3 * statistics.median(p["gmean"] for p in prev):
            out.append("GN-MEAN")
    if "LOSS+" in out and "ACC-" in out:
        out.append(">> COLLAPSE? stop (Ctrl+C), do not resume")
    return out


def show(h, start):
    for i in range(start, len(h)):
        r = h[i]
        print(f"{r['epoch']:5d} {r['lr']:8.1e} {r['loss']:7.3f} {r['acc']:7.4f} {r['strict']:7.4f} "
              f"{r['gmax']:7.2f} {r['gmean']:6.2f}  {' '.join(flags_for(h, i))}")
    if len(h) > start:
        mean_t = sum(r["dur"] for r in h) / len(h)
        left = max(0, h[-1]["total"] - h[-1]["epoch"])
        print(f"      done {h[-1]['epoch']}/{h[-1]['total']} | {mean_t:.0f}s/epoch | about {left * mean_t / 3600:.1f} h left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="experiments/diagnostic_qa_run1/logs/training.log")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=30)
    a = ap.parse_args()

    print("epoch       lr    loss     acc  strict  gn_max gn_mean  flags")
    seen = 0
    while True:
        h = parse(a.log)
        if len(h) > seen:
            show(h, seen)
            seen = len(h)
            if h[-1]["epoch"] >= h[-1]["total"]:
                print("finished.")
                return
        if not a.follow:
            if not h:
                print(f"no epoch lines found in {a.log}")
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()