"""
plot_results.py

Generates publication-quality Matplotlib figures from training history:
  1. training_curves.png: Train Loss & Overall Test Accuracy vs. Epochs (with 60% paper target line).
  2. category_curves.png: Per-category accuracy curves (Modality, Plane, Organ, Abnormality).

Usage:
  python scripts/plot_results.py --split 1
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all platforms
import matplotlib.pyplot as plt


def plot_training_history(split_k: int, history_path: Path = None, output_dir: Path = None, verbose: bool = True):
    history_path = history_path or Path(f"checkpoints/split_{split_k}/history.json")
    output_dir = output_dir or Path(f"results/reproduction/split_{split_k}/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not history_path.exists():
        print(f"Error: {history_path} does not exist yet. Run training first.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        print(f"Warning: {history_path} is empty.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_acc = [h["test_acc"] * 100 for h in history]

    # Style settings
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # =========================================================================
    # 1. Main Training Curves (Loss & Accuracy)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Plot 1: Train Loss
    ax1.plot(epochs, train_loss, color="#d9534f", linewidth=2.2, marker="o", markersize=4, label="Train Loss")
    ax1.set_title(f"Split {split_k}: Training Loss vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Plot 2: Test Accuracy
    ax2.plot(epochs, test_acc, color="#0275d8", linewidth=2.2, marker="s", markersize=4, label="Test Accuracy (20% Split)")
    ax2.axhline(y=60.0, color="#5cb85c", linestyle="--", linewidth=1.8, label="Paper Target (60% ± 1%)")
    best_acc = max(test_acc)
    best_epoch = epochs[test_acc.index(best_acc)]
    ax2.scatter([best_epoch], [best_acc], color="#f0ad4e", s=100, zorder=5, label=f"Peak: {best_acc:.2f}% (Epoch {best_epoch})")

    ax2.set_title(f"Split {split_k}: Test Accuracy vs. Epochs", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim([0, max(75, best_acc + 5)])
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    curve_path = output_dir / "training_curves.png"
    plt.savefig(curve_path, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved: {curve_path}")

    # =========================================================================
    # 2. Category Accuracy Breakdown
    # =========================================================================
    categories = ["modality", "plane", "organ", "abnormality"]
    cat_colors = {
        "modality": "#0275d8",
        "plane": "#5bc0de",
        "organ": "#5cb85c",
        "abnormality": "#d9534f"
    }
    paper_targets = {
        "modality": 76.0,
        "plane": 77.0,
        "organ": 74.0,
        "abnormality": 8.0
    }

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    for cat in categories:
        cat_vals = [h.get("category_acc", {}).get(cat, 0.0) * 100 for h in history]
        label = f"{cat.capitalize()} (Paper: {paper_targets.get(cat, 0)}%)"
        ax.plot(epochs, cat_vals, color=cat_colors.get(cat, "gray"), linewidth=2.0, marker="o", markersize=3.5, label=label)

    ax.set_title(f"Split {split_k}: Category Accuracy Breakdown (Table III)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim([0, 100])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="center right", frameon=True)

    plt.tight_layout()
    cat_path = output_dir / "category_curves.png"
    plt.savefig(cat_path, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved: {cat_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Med-VQA Training Results")
    parser.add_argument("--split", type=int, default=1, choices=[1, 2, 3, 4, 5], help="Split number (1-5)")
    args = parser.parse_args()

    plot_training_history(split_k=args.split)


if __name__ == "__main__":
    main()
