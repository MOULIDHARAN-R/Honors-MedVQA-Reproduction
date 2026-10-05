import json
import pandas as pd

df = pd.read_csv("data/processed/all_qa_pairs.csv", dtype=str, keep_default_na=False)
train_idx = json.load(open("data/splits/image/split_1.json"))["train_idx"]
train = df.iloc[train_idx]
sel = pd.read_csv("results/reproduction/run4/split_1/gradcam/selection.csv")

print("--- Abnormality Selection Leakage / Frequency Check ---")
for _, r in sel[sel.category == "abnormality"].iterrows():
    matches = train[train["answer"].str.lower().str.strip() == r["ground_truth"].lower().strip()]
    n = len(matches)
    print(f'Correct: {str(r["correct"]):5} | Train matches: {n:3d} | Image: {r["image_id"]} | GT: {r["ground_truth"]}')
    if n > 0 and n <= 5:
        print(f'   Matching train image_ids: {matches["image_id"].tolist()}')
