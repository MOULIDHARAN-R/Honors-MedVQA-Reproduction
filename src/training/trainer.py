"""
trainer.py (inside src/training/trainer.py)

Main training runner for a single split.
Paper: "Visual Question Answering in the Medical Domain" (Canepa et al., 2023)
  * 50 epochs, Adam optimizer (lr=1e-4), batch size 64.
  * ImageNet-pretrained VGG-16 + BERT + Concat + 2 FC layers.
  * Checkpoints saved to checkpoints/split_{k}/ (last.pt and best.pt).
"""
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config, CFG
from ..dataset import build_datasets, load_dataframe
from ..models import build_model
from .losses import get_criterion
from .checkpoint import save_checkpoint
from ..evaluation.evaluate import evaluate_model


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_batches: Optional[int] = None
) -> float:
    """Trains the model for one epoch and returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch:02d} [Train]",
        leave=False,
        file=sys.stdout
    )

    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (total_loss / num_batches) if num_batches > 0 else 0.0


def run_training_split(
    split_k: int,
    cfg: Config = CFG,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Executes the training and evaluation loop for split k.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"Starting Training: Split {split_k} on {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: {cfg.epochs} epochs | Batch Size: {cfg.batch_size} | LR: {cfg.lr}")
    if dry_run:
        print("[DRY RUN MODE ENABLED: Running 2 batches for 1 epoch]")
    print("=" * 70)

    # 1. Paths & Directories
    checkpoint_dir = cfg.checkpoint_dir(split_k)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "train.log"
    history_file = checkpoint_dir / "history.json"

    # Setup logger function
    def log(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"--- Training Run Initialized: Split {split_k} ---")

    # 2. Build Datasets & Vocabulary
    df = load_dataframe(cfg)
    train_ds, test_ds, vocab = build_datasets(cfg, split_k)
    num_classes = len(vocab)
    log(f"Split {split_k}: Train samples={len(train_ds)}, Test samples={len(test_ds)}, Classes={num_classes}")

    # 3. DataLoaders
    workers = 0 if sys.platform == "win32" else cfg.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda")
    )

    # 4. Model, Criterion, Optimizer
    seed = cfg.seed_for(split_k)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(num_classes=num_classes, cfg=cfg).to(device)
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    # 5. Training Loop
    best_acc = 0.0
    history = []
    epochs = 1 if dry_run else cfg.epochs
    max_batches = 2 if dry_run else None

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # A. Train
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=max_batches
        )

        # B. Evaluate
        eval_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            df=df,
            device=device,
            max_batches=max_batches
        )

        test_acc = eval_metrics["overall_acc"]
        cat_acc = eval_metrics["category_acc"]
        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        epoch_duration = time.time() - epoch_start

        # C. Format metrics
        cat_str = " | ".join(f"{c.capitalize()}: {acc:.2%}" for c, acc in cat_acc.items())
        msg = (
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_duration:.1f}s] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Test Acc: {test_acc:.4f} (Best: {best_acc:.4f}) "
            f"| {cat_str}"
        )
        log(msg)

        # D. Save history and checkpoint
        epoch_record = {
            "epoch": epoch,
            "duration_sec": epoch_duration,
            "train_loss": train_loss,
            "test_acc": test_acc,
            "best_acc": best_acc,
            "category_acc": cat_acc,
        }
        history.append(epoch_record)

        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        checkpoint_state = {
            "epoch": epoch,
            "split": split_k,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": vocab.idx2ans,
            "metrics": epoch_record,
        }
        save_checkpoint(
            state=checkpoint_state,
            is_best=is_best,
            checkpoint_dir=checkpoint_dir,
            filename="last.pt"
        )

        # Update live Matplotlib plots after every epoch
        try:
            from scripts.plot_results import plot_training_history
            plot_dir = cfg.results_dir / "reproduction" / f"split_{split_k}" / "plots"
            plot_training_history(split_k=split_k, history_path=history_file, output_dir=plot_dir, verbose=False)
        except Exception:
            pass

    total_time = time.time() - start_time
    log("=" * 70)
    log(f"Training Complete for Split {split_k}! Total Time: {total_time/60:.1f} mins")
    log(f"Final Epoch Accuracy (Official): {history[-1]['test_acc']:.4f}")
    log(f"Peak Epoch Accuracy: {best_acc:.4f}")
    log("=" * 70)

    # 6. Save summary in results/
    results_dir = cfg.results_dir / "reproduction" / f"split_{split_k}"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "summary.json"
    summary_data = {
        "split": split_k,
        "epochs": epochs,
        "final_accuracy": history[-1]["test_acc"],
        "best_accuracy": best_acc,
        "final_category_acc": history[-1]["category_acc"],
        "total_time_sec": total_time,
        "classes": num_classes,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 7. Generate Matplotlib visualization plots
    try:
        from scripts.plot_results import plot_training_history
        plot_training_history(split_k=split_k, history_path=history_file, output_dir=results_dir / "plots")
        log(f"Plots saved to: {results_dir / 'plots'}")
    except Exception as e:
        log(f"Plotting skipped: {e}")

    return summary_data
