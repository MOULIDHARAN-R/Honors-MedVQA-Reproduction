"""
train.py (project root CLI)

  python train.py --split 1 --dry_run            # 2 micro-batches, separate folder
  python train.py --split 1                      # full run (run name: run2)
  python train.py --split 1 --resume             # continue from checkpoints/run2/split_1/last.pt
  python train.py --all_splits --run_name run2
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from src.config import CFG
from src.training.trainer import run_training_split


def main():
    p = argparse.ArgumentParser(description="Med-VQA Reproduction Training")
    p.add_argument("--split", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all_splits", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--run_name", default="run2", help="folder name under checkpoints/ and results/")
    a = p.parse_args()

    splits = [1, 2, 3, 4, 5] if a.all_splits else [a.split]
    for k in splits:
        run_training_split(split_k=k, cfg=CFG, dry_run=a.dry_run, resume=a.resume, run_name=a.run_name)


if __name__ == "__main__":
    main()