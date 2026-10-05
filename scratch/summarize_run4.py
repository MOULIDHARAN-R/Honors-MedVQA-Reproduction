import json
from pathlib import Path
import numpy as np

splits_data = []
categories = ['modality', 'plane', 'organ', 'abnormality']

print("--- PER-SPLIT LAST EPOCH BREAKDOWN (RUN 4) ---")
for s in range(1, 6):
    p = Path(f"checkpoints/run4/split_{s}/history.json")
    with open(p, "r", encoding="utf-8") as f:
        h = json.load(f)
    last = h[-1]
    row = {
        "split": s,
        "overall": last["test_acc"],
        "strict": last.get("strict_acc", last["test_acc"]),
    }
    for c in categories:
        row[c] = last["category_acc"].get(c, 0.0)
    splits_data.append(row)
    print(f"Split {s}: Overall={row['overall']*100:.2f}%, Strict={row['strict']*100:.2f}% | "
          f"Modality={row['modality']*100:.2f}%, Plane={row['plane']*100:.2f}%, "
          f"Organ={row['organ']*100:.2f}%, Abnormality={row['abnormality']*100:.2f}%")

print("\n--- 5-SPLIT MEAN +- SAMPLE STD (RUN 4) ---")
overall_vals = [r["overall"] for r in splits_data]
strict_vals = [r["strict"] for r in splits_data]
print(f"Overall Acc: {np.mean(overall_vals)*100:.2f}% +- {np.std(overall_vals, ddof=1)*100:.2f}%")
print(f"Strict Acc:  {np.mean(strict_vals)*100:.2f}% +- {np.std(strict_vals, ddof=1)*100:.2f}%")

paper_table3 = {
    "modality": 0.76,
    "plane": 0.77,
    "organ": 0.74,
    "abnormality": 0.08,
    "overall": 0.60
}

print("\n--- COMPARISON TO PAPER (TABLE III: BERT Model) ---")
print(f"{'Category':<15} | {'Our Mean +- Std':<20} | {'Paper (Table III)':<20} | {'Gap (Our - Paper)':<20}")
print("-" * 80)
for c in categories:
    c_vals = [r[c] for r in splits_data]
    m = np.mean(c_vals)
    s = np.std(c_vals, ddof=1)
    target = paper_table3[c]
    gap = m - target
    print(f"{c.capitalize():<15} | {m*100:.2f}% +- {s*100:.2f}%     | {target*100:.1f}%                | {gap*100:+.2f} points")

m_ov = np.mean(overall_vals)
s_ov = np.std(overall_vals, ddof=1)
gap_ov = m_ov - paper_table3["overall"]
print("-" * 80)
print(f"{'Overall':<15} | {m_ov*100:.2f}% +- {s_ov*100:.2f}%     | 60.0% +- 1.0%        | {gap_ov*100:+.2f} points")
