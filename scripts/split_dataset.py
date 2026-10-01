"""
split_dataset.py

Creates N repeated random 80/20 train/test splits of all_qa_pairs.csv.

PAPER vs. OUR CHOICES
  Paper states: random 80/20 split, repeated five times, same split for every model
  variant. Paper does NOT state: the split unit (image vs QA pair) or the seed values.
  Our implementation choices (not paper facts):
    * Primary reproduction: --split_unit image (all QA pairs of an image stay together,
      so no image appears in both train and test). --split_unit qa is kept as an option.
    * Seed protocol: split i uses seed = base_seed + i, i.e. seeds 0,1,2,3,4.
  The five splits are independent random draws, so test sets may overlap across splits
  (this is repeated random subsampling, not k-fold CV).

DATA HANDLING
  This script never modifies the data. CSV values are read as raw strings and the
  script only SELECTS rows. Saved splits are row indices into the CSV exactly as
  loaded, so do not re-sort or filter the CSV afterwards (csv_md5 detects changes).

Usage (from MedVqa/):
    python scripts/split_dataset.py --split_unit image --seed 0
    python scripts/split_dataset.py --split_unit qa    --seed 0

Output:
    data/splits/<split_unit>/split_1.json ... split_5.json, summary.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLS = ["image_id", "question", "answer", "category", "image_path", "source"]


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(csv_path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: values like "None"/"NA" stay as written.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    for c in ["image_id", "question", "answer"]:
        n_empty = int((df[c] == "").sum())
        if n_empty:
            raise ValueError(f"Column '{c}' has {n_empty} empty values")
    return df


def report_dataset(df, expected_rows, expected_images):
    n_img = df["image_id"].nunique()
    print(f"QA pairs      : {len(df)}  (expected {expected_rows}; paper text says 15,992)")
    print(f"Unique images : {n_img}  (expected {expected_images})")
    if expected_rows is not None and len(df) != expected_rows:
        print(f"  WARNING: row count {len(df)} != expected {expected_rows}")
    if expected_images is not None and n_img != expected_images:
        print(f"  WARNING: image count {n_img} != expected {expected_images}")
    print("\nRows per source:\n" + df["source"].value_counts().to_string())
    print("\nRows per category:\n" + df["category"].value_counts().to_string())
    print("\nQA pairs per image:\n" + df.groupby("image_id").size().describe().to_string())
    print(f"\nUnique answers (raw): {df['answer'].nunique()}")
    # Read-only diagnostics: nothing is modified.
    for c in ["image_id", "question", "answer", "category"]:
        s = df[c]
        n_ws = int((s != s.str.strip()).sum())
        n_case = s.nunique() - s.str.lower().nunique()
        if n_ws or n_case:
            print(f"  note: '{c}' has {n_ws} values with edge whitespace and "
                  f"{n_case} case-only duplicate values (left untouched)")
    print()


def split_by_image(df, rng, test_size):
    images = df["image_id"].unique()
    perm = rng.permutation(len(images))
    n_test = int(round(len(images) * test_size))
    test_imgs = set(images[perm[:n_test]])
    is_test = df["image_id"].isin(test_imgs).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0]


def split_by_qa(df, rng, test_size):
    perm = rng.permutation(len(df))
    n_test = int(round(len(df) * test_size))
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/all_qa_pairs.csv")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--split_unit", choices=["image", "qa"], default="image")
    ap.add_argument("--seed", type=int, default=0, help="base seed; split i uses seed+i")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--expected_rows", type=int, default=15292)
    ap.add_argument("--expected_images", type=int, default=4200)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    df = load_raw(csv_path)
    report_dataset(df, args.expected_rows, args.expected_images)

    out_dir = Path(args.out_dir) / args.split_unit
    out_dir.mkdir(parents=True, exist_ok=True)
    split_fn = split_by_image if args.split_unit == "image" else split_by_qa
    csv_hash = file_md5(csv_path)

    summary = []
    for i in range(args.n_splits):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        train_idx, test_idx = split_fn(df, rng, args.test_size)
        train, test = df.iloc[train_idx], df.iloc[test_idx]

        shared_images = len(set(train["image_id"]) & set(test["image_id"]))
        unseen_frac = float((~test["answer"].isin(set(train["answer"]))).mean())

        info = {
            "split": i + 1,
            "seed": seed,
            "split_unit": args.split_unit,
            "test_size": args.test_size,
            "csv_md5": csv_hash,
            "n_total_rows": int(len(df)),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_train_images": int(train["image_id"].nunique()),
            "n_test_images": int(test["image_id"].nunique()),
            "images_in_both": int(shared_images),
            "test_rows_with_answer_unseen_in_train": round(unseen_frac, 4),
            "note": "split unit and seeds are our implementation choices; "
                    "the paper does not specify them",
        }
        with open(out_dir / f"split_{i + 1}.json", "w") as f:
            json.dump({**info, "train_idx": train_idx.tolist(),
                       "test_idx": test_idx.tolist()}, f)
        summary.append(info)
        print(f"split {i + 1} (seed {seed}): train={info['n_train']} test={info['n_test']} "
              f"| shared images={shared_images} | unseen-answer test rows={unseen_frac:.1%}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()