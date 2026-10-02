"""
preprocessing.py  (place at MedVqa/src/preprocessing.py)

Builds data/processed/all_qa_pairs.csv from the raw VQA-Med 2019 train / validation /
test files.

Design rules
  * question and answer are kept RAW (only the line ending is removed).
  * image_id and category are metadata: surrounding whitespace is stripped, and the
    category label is lower-cased so train/val/test use the same label set.
  * image_path is stored RELATIVE to the project root (portable across machines).
  * Category files only attach metadata; they never add QA pairs.

Run from MedVqa/:   python -m src.preprocessing
"""
from pathlib import Path
import pandas as pd

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DIR = RAW_DIR / "ImageClef-2019-VQA-Med-Training"
VAL_DIR = RAW_DIR / "ImageClef-2019-VQA-Med-Validation"
TEST_DIR = RAW_DIR / "VQAMed2019Test"

OUTPUT_FILE = PROCESSED_DIR / "all_qa_pairs.csv"

EXPECTED_CATEGORIES = {"modality", "plane", "organ", "abnormality"}
EXPECTED_ROWS = 15292      # our recollection of official split sizes; paper text says 15,992
EXPECTED_IMAGES = 4200
ENCODING = "utf-8-sig"     # handles a possible BOM on Windows-created files


def rel_path(image_dir: Path, image_id: str) -> str:
    return (image_dir / f"{image_id}.jpg").relative_to(PROJECT_ROOT).as_posix()


def read_lines(file_path: Path):
    """Yield (line_number, line) with only the line ending removed; skip blank lines."""
    with open(file_path, "r", encoding=ENCODING) as f:
        for n, line in enumerate(f, start=1):
            line = line.rstrip("\r\n")
            if line.strip():
                yield n, line


# ============================================================
# Parsers
# ============================================================
def parse_train_val_file(file_path, image_dir, source):
    """Format: image_id|question|answer"""
    rows = []
    for n, line in read_lines(file_path):
        parts = line.split("|", maxsplit=2)
        if len(parts) != 3:
            print(f"WARNING: could not parse {file_path.name} line {n}: {line}")
            continue
        image_id, question, answer = parts
        image_id = image_id.strip()
        rows.append({
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "category": None,
            "image_path": rel_path(image_dir, image_id),
            "source": source,
        })
    return rows


def parse_test_file(file_path, image_dir):
    """Format: image_id|category|question|reference_answer  ('#' separates alternatives)"""
    rows = []
    for n, line in read_lines(file_path):
        parts = line.split("|", maxsplit=3)
        if len(parts) != 4:
            print(f"WARNING: could not parse {file_path.name} line {n}: {line}")
            continue
        image_id, category, question, answer = parts
        image_id = image_id.strip()
        rows.append({
            "image_id": image_id,
            "question": question,
            "answer": answer,
            "category": category.strip().lower(),
            "image_path": rel_path(image_dir, image_id),
            "source": "test",
        })
    return rows


# ============================================================
# Categories (metadata only)
# ============================================================
def load_categories(file_path, category):
    """Map (image_id, question) -> category from a category-specific file."""
    category_map = {}
    if not file_path.exists():
        print(f"WARNING: category file not found: {file_path}")
        return category_map
    for n, line in read_lines(file_path):
        parts = line.split("|", maxsplit=2)
        if len(parts) != 3:
            print(f"WARNING: could not parse category file {file_path.name} line {n}")
            continue
        image_id, question, _ = parts
        category_map[(image_id.strip(), question)] = category
    return category_map


def attach_categories(rows, category_maps):
    for row in rows:
        key = (row["image_id"], row["question"])
        for cmap in category_maps:
            if key in cmap:
                row["category"] = cmap[key]
                break
    return rows


# ============================================================
# Validation (read-only checks)
# ============================================================
def validate_dataset(df):
    print("\n========== DATASET VALIDATION ==========")
    n_rows, n_img = len(df), df["image_id"].nunique()
    print(f"Total QA pairs: {n_rows}  (expected {EXPECTED_ROWS}; paper text says 15,992)")
    print(f"Unique images:  {n_img}  (expected {EXPECTED_IMAGES})")
    print(f"Unique image paths: {df['image_path'].nunique()}")
    if n_rows != EXPECTED_ROWS:
        print(f"  WARNING: row count differs from {EXPECTED_ROWS}")
    if n_img != EXPECTED_IMAGES:
        print(f"  WARNING: image count differs from {EXPECTED_IMAGES}")

    print("\nQA pairs by source:")
    print(df["source"].value_counts())

    print("\nQA pairs by category:")
    print(df["category"].value_counts(dropna=False))
    n_missing_cat = int(df["category"].isna().sum())
    if n_missing_cat:
        print(f"  WARNING: {n_missing_cat} rows have no category "
              f"(their (image_id, question) key was not found in any category file)")
    unexpected = set(df["category"].dropna().unique()) - EXPECTED_CATEGORIES
    if unexpected:
        print(f"  WARNING: unexpected category labels: {sorted(unexpected)}")

    dup = int(df.duplicated(subset=["image_id", "question", "answer"], keep=False).sum())
    print(f"\nDuplicate QA rows (kept): {dup}")

    n_sources = df.groupby("image_id")["source"].nunique()
    shared = int((n_sources > 1).sum())
    print(f"Images appearing in more than one source: {shared}")
    if shared:
        print("  WARNING: the same image_id occurs in several of train/val/test")

    hash_ans = df[df["answer"].str.contains("#", regex=False)]
    print(f"\nAnswers containing '#': {len(hash_ans)}")
    if len(hash_ans):
        print(hash_ans[["source", "category", "answer"]].head(10).to_string(index=False))

    missing = [p for p in df["image_path"].unique() if not (PROJECT_ROOT / p).exists()]
    print(f"\nMissing image files (unique paths): {len(missing)}")
    for p in missing[:10]:
        print(f"  {p}")
    print("========================================\n")


# ============================================================
# Main
# ============================================================
def main():
    print("========================================")
    print("     VQA-Med Dataset Preparation")
    print("========================================")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_qa = TRAIN_DIR / "All_QA_Pairs_train.txt"
    val_qa = VAL_DIR / "All_QA_Pairs_val.txt"
    test_qa = TEST_DIR / "VQAMed2019_Test_Questions_w_Ref_Answers.txt"
    for path in (train_qa, val_qa, test_qa):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    train_rows = parse_train_val_file(train_qa, TRAIN_DIR / "Train_images", "train")
    print(f"Training QA pairs:   {len(train_rows)}")
    val_rows = parse_train_val_file(val_qa, VAL_DIR / "Val_images", "validation")
    print(f"Validation QA pairs: {len(val_rows)}")
    test_rows = parse_test_file(test_qa, TEST_DIR / "VQAMed2019_Test_Images")
    print(f"Test QA pairs:       {len(test_rows)}")

    category_maps = []
    for base, suffix in ((TRAIN_DIR, "train"), (VAL_DIR, "val")):
        cat_dir = base / "QAPairsByCategory"
        category_maps += [
            load_categories(cat_dir / f"C1_Modality_{suffix}.txt", "modality"),
            load_categories(cat_dir / f"C2_Plane_{suffix}.txt", "plane"),
            load_categories(cat_dir / f"C3_Organ_{suffix}.txt", "organ"),
            load_categories(cat_dir / f"C4_Abnormality_{suffix}.txt", "abnormality"),
        ]
    train_rows = attach_categories(train_rows, category_maps)
    val_rows = attach_categories(val_rows, category_maps)

    df = pd.DataFrame(train_rows + val_rows + test_rows)
    df = df[["image_id", "question", "answer", "category", "image_path", "source"]]

    validate_dataset(df)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"Saved: {OUTPUT_FILE}\nDone.")


if __name__ == "__main__":
    main()