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


# ============================================================
# Helper functions
# ============================================================

def parse_train_val_file(file_path, image_dir, source):
    """
    Parse the VQA-Med training/validation QA files.

    Expected format:
        image_id|question|answer
    """

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split("|", maxsplit=2)

            if len(parts) != 3:
                print(
                    f"WARNING: Could not parse {file_path.name} "
                    f"line {line_number}: {line}"
                )
                continue

            image_id, question, answer = parts

            image_id = image_id.strip()
            question = question.strip()
            answer = answer.strip()

            image_path = image_dir / f"{image_id}.jpg"

            rows.append({
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "category": None,
                "image_path": str(image_path),
                "source": source,
            })

    return rows


def parse_test_file(file_path, image_dir):
    """
    Parse the VQA-Med 2019 test reference-answer file.

    Expected format:
        image_id|category|question|reference_answer

    Multiple acceptable answers are preserved using '#'.
    Optional-answer notation is also preserved exactly.
    """

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split("|", maxsplit=3)

            if len(parts) != 4:
                print(
                    f"WARNING: Could not parse {file_path.name} "
                    f"line {line_number}: {line}"
                )
                continue

            image_id, category, question, answer = parts

            image_id = image_id.strip()
            category = category.strip()
            question = question.strip()
            answer = answer.strip()

            image_path = image_dir / f"{image_id}.jpg"

            rows.append({
                "image_id": image_id,
                "question": question,
                "answer": answer,
                "category": category,
                "image_path": str(image_path),
                "source": "test",
            })

    return rows


# ============================================================
# Category handling
# ============================================================

def load_categories(file_path, category):
    """
    Read a category-specific QA file and create a mapping:

        (image_id, question) -> category

    These files are used only to attach category metadata.
    They are NOT added as additional QA pairs.
    """

    category_map = {}

    if not file_path.exists():
        print(f"WARNING: Category file not found: {file_path}")
        return category_map

    with open(file_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = line.split("|", maxsplit=2)

            if len(parts) != 3:
                print(
                    f"WARNING: Could not parse category file "
                    f"{file_path.name} line {line_number}"
                )
                continue

            image_id, question, _ = parts

            key = (image_id.strip(), question.strip())

            category_map[key] = category

    return category_map


def attach_categories(rows, category_maps):
    """
    Attach category information to train/validation rows.
    """

    for row in rows:

        key = (
            row["image_id"],
            row["question"]
        )

        for category_map in category_maps:

            if key in category_map:
                row["category"] = category_map[key]
                break

    return rows


# ============================================================
# Validation
# ============================================================

def validate_dataset(df):
    """
    Perform basic dataset sanity checks.
    """

    print("\n========== DATASET VALIDATION ==========")

    print(f"Total QA pairs: {len(df)}")
    print(f"Unique images:  {df['image_id'].nunique()}")

    print("\nQA pairs by source:")
    print(df["source"].value_counts())

    print("\nQA pairs by category:")
    print(df["category"].value_counts(dropna=False))

    # Duplicate QA pairs
    duplicate_mask = df.duplicated(
        subset=["image_id", "question", "answer"],
        keep=False
    )

    duplicate_count = duplicate_mask.sum()

    print(f"\nDuplicate QA rows: {duplicate_count}")

    # Missing images
    missing_images = []

    for path in df["image_path"]:
        if not Path(path).exists():
            missing_images.append(path)

    print(f"Missing image files: {len(missing_images)}")

    if missing_images:
        print("\nFirst missing images:")
        for path in missing_images[:10]:
            print(f"  {path}")

    print("========================================\n")


# ============================================================
# Main dataset preparation
# ============================================================

def main():

    print("========================================")
    print("     VQA-Med Dataset Preparation")
    print("========================================")

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check required directories/files
    # --------------------------------------------------------

    train_qa = TRAIN_DIR / "All_QA_Pairs_train.txt"
    val_qa = VAL_DIR / "All_QA_Pairs_val.txt"
    test_qa = TEST_DIR / "VQAMed2019_Test_Questions_w_Ref_Answers.txt"

    required_paths = [
        train_qa,
        val_qa,
        test_qa,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Parse training data
    # --------------------------------------------------------

    print("\nReading training QA pairs...")

    train_rows = parse_train_val_file(
        train_qa,
        TRAIN_DIR / "Train_images",
        "train"
    )

    print(f"Training QA pairs: {len(train_rows)}")

    # --------------------------------------------------------
    # Parse validation data
    # --------------------------------------------------------

    print("\nReading validation QA pairs...")

    val_rows = parse_train_val_file(
        val_qa,
        VAL_DIR / "Val_images",
        "validation"
    )

    print(f"Validation QA pairs: {len(val_rows)}")

    # --------------------------------------------------------
    # Parse test data
    # --------------------------------------------------------

    print("\nReading test QA pairs...")

    test_rows = parse_test_file(
        test_qa,
        TEST_DIR / "VQAMed2019_Test_Images"
    )

    print(f"Test QA pairs: {len(test_rows)}")

    # --------------------------------------------------------
    # Load category information
    # --------------------------------------------------------

    print("\nLoading category information...")

    category_maps = []

    # Training categories
    train_category_dir = TRAIN_DIR / "QAPairsByCategory"

    category_maps.extend([
        load_categories(
            train_category_dir / "C1_Modality_train.txt",
            "modality"
        ),
        load_categories(
            train_category_dir / "C2_Plane_train.txt",
            "plane"
        ),
        load_categories(
            train_category_dir / "C3_Organ_train.txt",
            "organ"
        ),
        load_categories(
            train_category_dir / "C4_Abnormality_train.txt",
            "abnormality"
        ),
    ])

    # Validation categories
    val_category_dir = VAL_DIR / "QAPairsByCategory"

    category_maps.extend([
        load_categories(
            val_category_dir / "C1_Modality_val.txt",
            "modality"
        ),
        load_categories(
            val_category_dir / "C2_Plane_val.txt",
            "plane"
        ),
        load_categories(
            val_category_dir / "C3_Organ_val.txt",
            "organ"
        ),
        load_categories(
            val_category_dir / "C4_Abnormality_val.txt",
            "abnormality"
        ),
    ])

    train_rows = attach_categories(
        train_rows,
        category_maps
    )

    val_rows = attach_categories(
        val_rows,
        category_maps
    )

    # --------------------------------------------------------
    # Combine all QA pairs
    # --------------------------------------------------------

    all_rows = train_rows + val_rows + test_rows

    df = pd.DataFrame(all_rows)

    # Ensure consistent column order
    df = df[
        [
            "image_id",
            "question",
            "answer",
            "category",
            "image_path",
            "source",
        ]
    ]

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"Processed dataset saved to:")

    print(OUTPUT_FILE)

    print("\nDone.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()