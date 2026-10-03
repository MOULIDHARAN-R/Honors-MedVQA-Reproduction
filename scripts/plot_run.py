"""
python -m scripts.plot_run --run run2 --split 1             # draw once
python -m scripts.plot_run --run run2 --split 1 --follow    # redraw after every epoch (Ctrl+C to stop)

Reads  checkpoints/<run>/split_<k>/history.json   (written by the trainer after each epoch)
Writes results/reproduction/<run>/split_<k>/plots/training_curves.png
It only reads history.json, so it is safe to run while training.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_ACC = 0.60            # paper's reported overall accuracy (VQA-Med 2019)
NAMES = {"organ": "organ system"}


def load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None         # missing or half-written; caller retries


def draw(h, title, out):
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle(title, fontsize=13)

    ax[0, 0].plot(ep, [r["train_loss"] for r in h], marker="o", ms=3, color="tab:red")
    ax[0, 0].set(title="Training loss", xlabel="epoch", ylabel="cross-entropy")

    ax[0, 1].plot(ep, [r["test_acc"] for r in h], marker="o", ms=3, label="official semantics")
    if all("strict_acc" in r for r in h):
        ax[0, 1].plot(ep, [r["strict_acc"] for r in h], ls="--", label="strict")
    ax[0, 1].axhline(PAPER_ACC, color="gray", ls=":", label=f"paper ({PAPER_ACC:.2f})")
    ax[0, 1].set(title="Test accuracy (overall)", xlabel="epoch", ylabel="accuracy", ylim=(0, 0.8))
    ax[0, 1].legend(loc="lower right")

    for c in sorted(h[-1]["category_acc"].keys()):
        ax[1, 0].plot(ep, [r["category_acc"].get(c, float("nan")) for r in h],
                      marker="o", ms=3, label=NAMES.get(c, c))
    ax[1, 0].set(title="Test accuracy by category", xlabel="epoch", ylabel="accuracy", ylim=(0, 1.15))
    ax[1, 0].legend(loc="upper left", ncol=4, fontsize=8)

    if all("grad_norm_max" in r for r in h):
        ax[1, 1].plot(ep, [r["grad_norm_max"] for r in h], marker="o", ms=3, label="max per epoch")
        ax[1, 1].plot(ep, [r["grad_norm_mean"] for r in h], marker="o", ms=3, label="mean per epoch")
        ax[1, 1].set_yscale("log")
        ax[1, 1].legend()
    ax[1, 1].set(title="Gradient norm (before any clipping)", xlabel="epoch", ylabel="L2 norm")

    for axis in ax.flat:
        axis.grid(alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + "_tmp.png")        # write then replace, so a viewer never sees a half file
    fig.savefig(tmp, dpi=150)
    plt.close(fig)
    tmp.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run2")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--history", default=None, help="explicit path to a history.json")
    ap.add_argument("--follow", action="store_true", help="redraw whenever a new epoch appears")
    ap.add_argument("--interval", type=int, default=30, help="seconds between checks in --follow mode")
    a = ap.parse_args()

    hist_path = Path(a.history) if a.history else \
        PROJECT_ROOT / "checkpoints" / a.run / f"split_{a.split}" / "history.json"
    out = PROJECT_ROOT / "results" / "reproduction" / a.run / f"split_{a.split}" / "plots" / "training_curves.png"

    seen = 0
    while True:
        h = load(hist_path)
        if h and len(h) != seen:
            seen = len(h)
            title = f"Med-VQA reproduction | run {a.run} | split {a.split} | {seen} epochs"
            draw(h, title, out)
            last = h[-1]
            print(f"[{time.strftime('%H:%M:%S')}] epoch {last['epoch']}: acc {last['test_acc']:.4f} "
                  f"loss {last['train_loss']:.4f} -> {out}")
        if not a.follow:
            if h is None:
                print(f"no readable history at {hist_path}")
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()