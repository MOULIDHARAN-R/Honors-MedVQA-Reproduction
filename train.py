"""
train.py (Root CLI Entry Point)

Usage:
  # Dry-run test (runs 2 batches in ~5 seconds to verify everything)
  python train.py --split 1 --dry_run

  # Full training of Split 1 (50 epochs)
  python train.py --split 1

  # Train all 5 splits sequentially
  python train.py --all_splits
"""
import argparse
import sys
from src.config import CFG
from src.training.trainer import run_training_split


def main():
    parser = argparse.ArgumentParser(description="Med-VQA Reproduction Training")
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Split number to train (1 to 5). Default: 1"
    )
    parser.add_argument(
        "--all_splits",
        action="store_true",
        help="Train all 5 splits sequentially (1 through 5)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 2 batches for 1 epoch to test pipeline end-to-end"
    )

    args = parser.parse_args()

    if args.all_splits:
        splits = [1, 2, 3, 4, 5]
        print(f"Training all 5 splits sequentially: {splits}")
    else:
        splits = [args.split]

    for k in splits:
        run_training_split(split_k=k, cfg=CFG, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
