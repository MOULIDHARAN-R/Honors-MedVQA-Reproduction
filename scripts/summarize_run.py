"""
scripts/summarize_run.py

Summarizes a 5-split reproduction run:
  - Last-epoch overall and strict accuracies per split
  - Per-category breakdown (Modality, Plane, Organ, Abnormality)
  - Unseen-answer ceilings from data/splits/image/summary.json
  - 5-split mean +- sample standard deviation (ddof=1)
  - Comparison to the paper's Table III (BERT model)

Usage:
    python -m scripts.summarize_run --run run4
"""
import argparse
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_SPLITS = 5
CATEGORIES = ["modality", "plane", "organ", "abnormality"]

PAPER_TABLE3 = {
    "modality": 0.76,
    "plane": 0.77,
    "organ": 0.74,
    "abnormality": 0.08,
    "overall": 0.60,
}


def main():
    parser = argparse.ArgumentParser(description="Summarize a 5-split reproduction run.")
    parser.add_argument("--run", default="run4", help="Run name under checkpoints/ (default: run4)")
    args = parser.parse_args()

    # Load split summary for unseen answer ceilings
    splits_summary_path = PROJECT_ROOT / "data" / "splits" / "image" / "summary.json"
    ceilings = {}
    if splits_summary_path.exists():
        with open(splits_summary_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            for item in meta:
                s = item["split"]
                unseen = item.get("test_rows_with_answer_unseen_in_train", 0.0)
                ceilings[s] = (1.0 - unseen) * 100

    splits_data = []
    for s in range(1, N_SPLITS + 1):
        hist_file = PROJECT_ROOT / "checkpoints" / args.run / f"split_{s}" / "history.json"
        if not hist_file.exists():
            print(f"Error: {hist_file} not found!")
            return
        with open(hist_file, "r", encoding="utf-8") as f:
            h = json.load(f)
        last = h[-1]
        row = {
            "split": s,
            "epoch": last["epoch"],
            "overall": last["test_acc"] * 100,
            "strict": last.get("strict_acc", last["test_acc"]) * 100,
            "ceiling": ceilings.get(s, float("nan")),
        }
        for c in CATEGORIES:
            row[c] = last["category_acc"].get(c, 0.0) * 100
        splits_data.append(row)

    print("\n" + "=" * 100)
    print(f"MED-VQA REPRODUCTION SUMMARY: {args.run.upper()} (5 Image-Level Splits, 50 Epochs each)")
    print("=" * 100)

    # 1. Per-split table
    header = f"{'Split':<7} | {'Overall (%)':<11} | {'Strict (%)':<10} | {'Modality (%)':<12} | {'Plane (%)':<10} | {'Organ (%)':<10} | {'Abnormal (%)':<12} | {'Ceiling (%)':<11}"
    print(header)
    print("-" * len(header))
    for r in splits_data:
        print(f"Split {r['split']:<1} | {r['overall']:<11.2f} | {r['strict']:<10.2f} | {r['modality']:<12.2f} | {r['plane']:<10.2f} | {r['organ']:<10.2f} | {r['abnormality']:<12.2f} | {r['ceiling']:<11.2f}")

    print("-" * len(header))

    # Calculate statistics
    means = {k: np.mean([r[k] for r in splits_data]) for k in ["overall", "strict", "ceiling"] + CATEGORIES}
    stds = {k: np.std([r[k] for r in splits_data], ddof=1) for k in ["overall", "strict", "ceiling"] + CATEGORIES}

    mean_row = f"{'Mean':<7} | {means['overall']:<11.2f} | {means['strict']:<10.2f} | {means['modality']:<12.2f} | {means['plane']:<10.2f} | {means['organ']:<10.2f} | {means['abnormality']:<12.2f} | {means['ceiling']:<11.2f}"
    std_row = f"{'Std':<7} | {stds['overall']:<11.2f} | {stds['strict']:<10.2f} | {stds['modality']:<12.2f} | {stds['plane']:<10.2f} | {stds['organ']:<10.2f} | {stds['abnormality']:<12.2f} | {stds['ceiling']:<11.2f}"
    print(mean_row)
    print(std_row)
    print("=" * len(header))

    # 2. Paper Comparison Table
    print("\n" + "=" * 80)
    print("COMPARISON WITH PAPER (Table III: BERT + VGG-16 Model)")
    print("=" * 80)
    cmp_hdr = f"{'Category / Metric':<20} | {'Reproduction (Mean +- Std)':<28} | {'Paper Target':<15} | {'Gap (Our - Paper)':<18}"
    print(cmp_hdr)
    print("-" * len(cmp_hdr))

    for c in CATEGORIES:
        m, s = means[c], stds[c]
        p_target = PAPER_TABLE3[c] * 100
        gap = m - p_target
        print(f"{c.capitalize():<20} | {m:>5.2f}% +- {s:>4.2f}%              | {p_target:>5.1f}%          | {gap:>+6.2f} points")

    print("-" * len(cmp_hdr))
    m_ov, s_ov = means["overall"], stds["overall"]
    p_ov = PAPER_TABLE3["overall"] * 100
    gap_ov = m_ov - p_ov
    print(f"{'Overall (Official)':<20} | {m_ov:>5.2f}% +- {s_ov:>4.2f}%              | 60.0% +- 1.0%   | {gap_ov:>+6.2f} points")
    print(f"{'Overall (Strict)':<20} | {means['strict']:>5.2f}% +- {stds['strict']:>4.2f}%              |       -         |        -")
    print(f"{'Unseen-Answer Ceiling':<20} | {means['ceiling']:>5.2f}% +- {stds['ceiling']:>4.2f}%              |       -         |        -")
    print("=" * len(cmp_hdr) + "\n")

    # 3. Save to Markdown and CSV in results directory
    out_dirs = [
        PROJECT_ROOT / "results" / "reproduction" / args.run,
    ]
    for out_d in out_dirs:
        out_d.mkdir(parents=True, exist_ok=True)
        # Markdown Table
        md_lines = [
            f"# Reproduction Results Summary ({args.run.upper()})\n",
            "| Split | Overall Acc (%) | Strict Acc (%) | Modality (%) | Plane (%) | Organ (%) | Abnormality (%) | Unseen Ceiling (%) |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in splits_data:
            md_lines.append(f"| Split {r['split']} | {r['overall']:.2f} | {r['strict']:.2f} | {r['modality']:.2f} | {r['plane']:.2f} | {r['organ']:.2f} | {r['abnormality']:.2f} | {r['ceiling']:.2f} |")
        md_lines.append(f"| **Mean** | **{means['overall']:.2f}** | **{means['strict']:.2f}** | **{means['modality']:.2f}** | **{means['plane']:.2f}** | **{means['organ']:.2f}** | **{means['abnormality']:.2f}** | **{means['ceiling']:.2f}** |")
        md_lines.append(f"| **Std** | **± {stds['overall']:.2f}** | **± {stds['strict']:.2f}** | **± {stds['modality']:.2f}** | **± {stds['plane']:.2f}** | **± {stds['organ']:.2f}** | **± {stds['abnormality']:.2f}** | **± {stds['ceiling']:.2f}** |")
        md_lines.append("\n## Comparison with Paper (Table III: BERT + VGG-16)\n")
        md_lines.append("| Category / Metric | Reproduction (Mean ± Std) | Paper Target | Gap (Our - Paper) |")
        md_lines.append("| :--- | :---: | :---: | :---: |")
        for c in CATEGORIES:
            md_lines.append(f"| {c.capitalize()} | {means[c]:.2f}% ± {stds[c]:.2f}% | {PAPER_TABLE3[c]*100:.1f}% | {means[c]-PAPER_TABLE3[c]*100:+.2f} points |")
        md_lines.append(f"| **Overall (Official)** | **{means['overall']:.2f}% ± {stds['overall']:.2f}%** | **60.0% ± 1.0%** | **{means['overall']-60.0:+.2f} points** |")
        md_lines.append(f"| Overall (Strict) | {means['strict']:.2f}% ± {stds['strict']:.2f}% | - | - |")
        md_lines.append(f"| Unseen-Answer Ceiling | {means['ceiling']:.2f}% ± {stds['ceiling']:.2f}% | - | - |")

        with open(out_d / "results_table.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines) + "\n")

        # CSV Table
        import csv
        with open(out_d / "results_table.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "overall", "strict", "modality", "plane", "organ", "abnormality", "ceiling"])
            for r in splits_data:
                writer.writerow([r["split"], f"{r['overall']:.2f}", f"{r['strict']:.2f}", f"{r['modality']:.2f}", f"{r['plane']:.2f}", f"{r['organ']:.2f}", f"{r['abnormality']:.2f}", f"{r['ceiling']:.2f}"])
            writer.writerow(["mean", f"{means['overall']:.2f}", f"{means['strict']:.2f}", f"{means['modality']:.2f}", f"{means['plane']:.2f}", f"{means['organ']:.2f}", f"{means['abnormality']:.2f}", f"{means['ceiling']:.2f}"])
            writer.writerow(["std", f"{stds['overall']:.2f}", f"{stds['strict']:.2f}", f"{stds['modality']:.2f}", f"{stds['plane']:.2f}", f"{stds['organ']:.2f}", f"{stds['abnormality']:.2f}", f"{stds['ceiling']:.2f}"])
        print(f"Saved results table to {out_d / 'results_table.md'} and .csv")


if __name__ == "__main__":
    main()