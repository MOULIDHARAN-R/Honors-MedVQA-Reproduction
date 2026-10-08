"""
experiments/blind_baseline/run_blind_baseline.py

Runs the language-prior blind baseline experiment.
Evaluates accuracy of a question-only majority baseline without images.

Usage:
    python experiments/blind_baseline/run_blind_baseline.py --run run4
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.answers import canonical_label
from src.config import CFG

CATS = ["modality", "plane", "organ", "abnormality"]
TYPES = ["yes/no", "choice", "open"]


def norm(q):
    return re.sub(r"\s+", " ", str(q)).strip().lower()


def qtype(question, answer):
    if answer.strip().lower() in ("yes", "no"):
        return "yes/no"
    return "choice" if " or " in question.lower() else "open"


def main():
    parser = argparse.ArgumentParser(description="Blind Language Baseline")
    parser.add_argument("--run", default="run4", help="Run name (default: run4)")
    args = parser.parse_args()

    cfg, sep = CFG, CFG.answer_alt_separator
    df = pd.read_csv(cfg.csv_path, dtype=str, keep_default_na=False)
    df["tmpl"] = df["question"].map(norm)
    df["cat"] = df["category"].str.lower()
    df["lab"] = df["answer"].map(lambda x: canonical_label(x, sep))

    res = defaultdict(list)
    for k in range(1, cfg.n_splits + 1):
        with open(cfg.split_file(k), "r", encoding="utf-8") as f:
            sp = json.load(f)
        train, test = df.iloc[sp["train_idx"]], df.iloc[sp["test_idx"]]
        by_t = {t: Counter(g["lab"]).most_common(1)[0][0] for t, g in train.groupby("tmpl")}
        by_c = {c: Counter(g["lab"]).most_common(1)[0][0] for c, g in train.groupby("cat")}
        glob = Counter(train["lab"]).most_common(1)[0][0]

        corr, tot = Counter(), Counter()
        for _, r in test.iterrows():
            pred = by_t.get(r["tmpl"], by_c.get(r["cat"], glob))
            c = r["cat"]
            t = f"{c}: {qtype(r['question'], r['answer'])}"
            hit = (pred == r["lab"])
            for g in ["overall", c, t]:
                corr[g] += hit
                tot[g] += 1
        for g in tot:
            res[g].append(corr[g] / tot[g])

    # Load model results if available
    mod_means = {}
    mod_file = PROJECT_ROOT / "results" / "reproduction" / args.run / "results_table.csv"
    if mod_file.exists():
        with open(mod_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == "mean":
                    mod_means["overall"] = float(row["overall"]) / 100
                    for c in CATS:
                        mod_means[c] = float(row[c]) / 100

    import numpy as np
    rows = []
    print(f"\n=== BLIND LANGUAGE BASELINE (Run: {args.run}) ===")
    print(f"{'Group':<20} | {'Blind Mean +- Std':<22} | {'Model Mean':<12} | {'Model - Blind':<14}")
    print("-" * 75)

    order = ["overall"] + CATS + [f"modality: {t}" for t in TYPES]
    for g in order:
        if g in res:
            m = np.mean(res[g])
            s = np.std(res[g], ddof=1) if len(res[g]) > 1 else 0.0
            mm = mod_means.get(g, None)
            diff_str = f"{(mm - m)*100:+.2f}%" if mm is not None else "-"
            mm_str = f"{mm*100:.2f}%" if mm is not None else "-"
            print(f"{g:<20} | {m*100:>5.2f}% +- {s*100:>4.2f}%         | {mm_str:<12} | {diff_str:<14}")
            rows.append({"group": g, "blind_mean": f"{m:.4f}", "blind_std": f"{s:.4f}", "model_mean": f"{mm:.4f}" if mm else "-"})

    # Save results
    out_dirs = [
        PROJECT_ROOT / "experiments" / "blind_baseline" / "results",
        PROJECT_ROOT / "results" / "reproduction" / args.run,
    ]
    for out_d in out_dirs:
        out_d.mkdir(parents=True, exist_ok=True)
        with open(out_d / "blind_baseline.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "blind_mean", "blind_std", "model_mean"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {out_d / 'blind_baseline.csv'}")


if __name__ == "__main__":
    main()
