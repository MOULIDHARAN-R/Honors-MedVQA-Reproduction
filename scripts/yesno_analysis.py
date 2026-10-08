"""
python -m scripts.yesno_analysis --run run4

Reads results/reproduction/<run>/modality_breakdown/predictions_split_k.csv (written by modality_breakdown)
and breaks the modality YES/NO questions down by question wording (whitespace/tab-normalised).
CPU only, no model needed.

Per question template (pooled over the splits; test sets overlap, so n counts row appearances):
  n, accuracy, share of 'yes' in the reference, share of 'yes' in the predictions,
  accuracy when the reference is 'yes' / 'no', and the majority-answer baseline for that template.
Saves results/reproduction/<run>/modality_breakdown/yesno_by_template.csv
"""
import argparse
import csv
import glob
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", str(q)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run4")
    ap.add_argument("--min_n", type=int, default=30)
    a = ap.parse_args()

    d = PROJECT_ROOT / "results" / "reproduction" / a.run / "modality_breakdown"
    files = sorted(glob.glob(str(d / "predictions_split_*.csv")))
    if not files:
        print(f"no predictions_split_*.csv in {d}")
        return
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False).assign(split=Path(f).stem[-1]) for f in files])
    df = df[df["qtype"] == "yes/no"].copy()
    df["tmpl"] = df["question"].map(norm)
    df["gt"] = df["ground_truth"].str.strip().str.lower()
    df["pr"] = df["prediction"].str.strip().str.lower()
    df["ok"] = df["correct"].astype(int)

    print(f"yes/no modality rows pooled over {len(files)} splits: {len(df)}")
    print(f"overall accuracy {df['ok'].mean():.3f} | reference 'yes' share {(df['gt'] == 'yes').mean():.3f} "
          f"| predicted 'yes' share {(df['pr'] == 'yes').mean():.3f} "
          f"| predictions that are neither yes nor no: {(~df['pr'].isin(['yes', 'no'])).sum()}\n")

    rows = []
    for t, g in df.groupby("tmpl"):
        if len(g) < a.min_n:
            continue
        yes, no = g[g["gt"] == "yes"], g[g["gt"] == "no"]
        rows.append({
            "template": t, "n": len(g), "acc": g["ok"].mean(),
            "ref_yes": (g["gt"] == "yes").mean(), "pred_yes": (g["pr"] == "yes").mean(),
            "acc_if_ref_yes": yes["ok"].mean() if len(yes) else float("nan"),
            "acc_if_ref_no": no["ok"].mean() if len(no) else float("nan"),
            "majority": max((g["gt"] == "yes").mean(), (g["gt"] == "no").mean()),
        })
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 48)
    print(out.round(3).to_string(index=False))
    out.to_csv(d / "yesno_by_template.csv", index=False)
    print(f"\n(templates with n < {a.min_n} omitted)  saved to {d / 'yesno_by_template.csv'}")


if __name__ == "__main__":
    main()