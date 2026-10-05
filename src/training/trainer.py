"""
trainer.py  (src/training/trainer.py)

Training runner for one split.
  * Effective batch = cfg.batch_size (64) [PAPER], realised as micro-batches of
    cfg.micro_batch_size with gradient accumulation (memory only; same gradient).
  * Adam, 50 epochs [PAPER]. Learning rate = cfg.lr * cfg.lr_gamma ** ((epoch-1) // cfg.lr_step_size),
    a pure function of the epoch number (safe to resume). Gradient clipping at cfg.grad_clip_norm.
    Both the schedule and the clipping are DOCUMENTED DEVIATIONS from the paper.
  * The gradient norm is measured every optimizer step and logged per epoch.
  * Result to report = LAST-epoch accuracy. best.pt / best_acc are diagnostics selected
    on the test set and must not be reported as the result.
  * Each run writes to checkpoints/<run_name>/split_k and results/reproduction/<run_name>/split_k.
    Dry runs use their own 'dry_run' folders.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # OpenMP duplicate-library workaround (Windows)

import json
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config, CFG
from ..dataset import build_datasets, load_dataframe, file_md5
from ..models import build_model
from ..evaluation.evaluate import evaluate_model
from .checkpoint import load_checkpoint, save_checkpoint
from .losses import get_criterion


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, cfg,
                    max_batches: Optional[int] = None):
    """One epoch with gradient accumulation. Returns (mean per-sample loss, list of grad norms)."""
    model.train()
    accum = cfg.batch_size // cfg.micro_batch_size
    n_total = len(dataloader.dataset)
    total_loss, total_n, grad_norms = 0.0, 0, []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch:02d} [Train]", leave=False, file=sys.stdout)
    for i, batch in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        loss = criterion(model(images, input_ids, attention_mask), labels)   # mean over micro-batch

        # weight so the accumulated gradient equals the mean over the whole effective batch
        group = i // accum
        group_size = min(cfg.batch_size, n_total - group * cfg.batch_size)
        (loss * labels.size(0) / group_size).backward()

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)

        is_last = (i + 1 == len(dataloader)) or (max_batches is not None and i + 1 == max_batches)
        if (i + 1) % accum == 0 or is_last:
            max_norm = cfg.grad_clip_norm if cfg.grad_clip_norm else float("inf")  # inf = measure only
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            grad_norms.append(float(gn))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (total_loss / total_n if total_n else 0.0), grad_norms


def run_training_split(split_k: int, cfg: Config = CFG, dry_run: bool = False,
                       resume: bool = False, run_name: str = "run2") -> Optional[Dict[str, Any]]:
    assert cfg.batch_size % cfg.micro_batch_size == 0, "batch_size must be a multiple of micro_batch_size"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_tag = "dry_run" if dry_run else run_name
    checkpoint_dir = cfg.checkpoints_dir / run_tag / f"split_{split_k}"
    results_dir = cfg.results_dir / "reproduction" / run_tag / f"split_{split_k}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "train.log"
    history_file = checkpoint_dir / "history.json"

    def log(msg: str):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"--- Training run '{run_tag}' | split {split_k} ---")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "-"
    log(f"Device: {device} ({gpu}) | torch {torch.__version__}")
    log(f"Epochs {cfg.epochs} | effective batch {cfg.batch_size} "
        f"(micro {cfg.micro_batch_size} x {cfg.batch_size // cfg.micro_batch_size}) | lr {cfg.lr} "
        f"| grad_clip {cfg.grad_clip_norm} | lr x{cfg.lr_gamma} every {cfg.lr_step_size} epochs "
        f"| AMP {cfg.use_amp} | workers {cfg.num_workers}")
    if dry_run:
        log("[DRY RUN: 2 micro-batches, 1 epoch]")

    csv_md5 = file_md5(cfg.csv_path)
    cfg.save(results_dir / "config.json")

    # data
    df = load_dataframe(cfg)
    train_ds, test_ds, vocab = build_datasets(cfg, split_k)
    num_classes = len(vocab)
    log(f"Train={len(train_ds)} Test={len(test_ds)} Classes={num_classes} csv_md5={csv_md5}")

    workers = cfg.num_workers
    common = dict(num_workers=workers, pin_memory=(device.type == "cuda"),
                  persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.micro_batch_size, shuffle=False, **common)

    # model
    seed = cfg.seed_for(split_k)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(num_classes=num_classes, cfg=cfg).to(device)
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    start_epoch, best_acc, history = 1, 0.0, []
    epochs = 1 if dry_run else cfg.epochs
    max_batches = 2 if dry_run else None

    last_ckpt = checkpoint_dir / "last.pt"
    if resume and last_ckpt.exists():
        log(f"Loading {last_ckpt} ...")
        ckpt = load_checkpoint(last_ckpt, model=model, optimizer=optimizer, device=device)
        assert ckpt["vocab"] == vocab.idx2ans, \
            "Vocabulary differs from the checkpoint: CSV, split or answers.py changed"
        start_epoch = ckpt["epoch"] + 1
        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            history = [r for r in history if r["epoch"] <= ckpt["epoch"]]   # drop epochs the checkpoint doesn't have
            best_acc = max((h["test_acc"] for h in history), default=0.0)
        log(f"Resumed. Next epoch {start_epoch}/{epochs}")

    start_time = time.time()
    last_eval: Dict[str, Any] = {}

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr = cfg.lr * (cfg.lr_gamma ** ((epoch - 1) // cfg.lr_step_size))
        for g in optimizer.param_groups:
            g["lr"] = lr
        train_loss, grad_norms = train_one_epoch(model, train_loader, criterion, optimizer,
                                                 device, epoch, cfg, max_batches)
        last_eval = evaluate_model(model, test_loader, df, device, max_batches)

        test_acc = last_eval["overall_acc"]
        is_best = test_acc > best_acc          # diagnostic only (selected on the test set)
        best_acc = max(best_acc, test_acc)
        duration = time.time() - t0
        gn_max = max(grad_norms) if grad_norms else 0.0
        gn_mean = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        cat_str = " | ".join(f"{c.capitalize()}: {a:.2%}" for c, a in last_eval["category_acc"].items())
        log(f"Epoch {epoch:02d}/{epochs:02d} [{duration:.0f}s] | lr {lr:.1e} | loss {train_loss:.4f} "
            f"| acc {test_acc:.4f} (strict {last_eval['strict_acc']:.4f}, peak {best_acc:.4f}) "
            f"| gnorm max {gn_max:.2f} mean {gn_mean:.2f} | {cat_str}")

        record = {
            "epoch": epoch, "lr": lr, "duration_sec": duration, "train_loss": train_loss,
            "test_acc": test_acc, "strict_acc": last_eval["strict_acc"], "best_acc": best_acc,
            "grad_norm_max": gn_max, "grad_norm_mean": gn_mean,
            "category_acc": last_eval["category_acc"],
            "category_strict_acc": last_eval["category_strict_acc"],
        }
        history.append(record)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(
            state={
                "epoch": epoch, "split": split_k, "run": run_tag,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": vocab.idx2ans, "metrics": record, "csv_md5": csv_md5,
                "config": json.loads(json.dumps(asdict(cfg), default=str)),
            },
            is_best=is_best, checkpoint_dir=checkpoint_dir, filename="last.pt")

        try:
            from scripts.plot_results import plot_training_history
            plot_training_history(split_k=split_k, history_path=history_file,
                                  output_dir=results_dir / "plots", verbose=False)
        except Exception:
            pass

    total_time = time.time() - start_time
    if not history:
        log("Nothing to do (already finished?).")
        return None

    final = history[-1]
    log("=" * 70)
    log(f"Split {split_k} done in {total_time / 60:.1f} min | classes {num_classes}")
    log(f"REPORTED (last epoch {final['epoch']}): official {final['test_acc']:.4f} | strict {final['strict_acc']:.4f}")
    log(f"Diagnostic only, selected on test set: peak {best_acc:.4f}")
    log("=" * 70)

    summary = {
        "split": split_k, "run": run_tag, "epochs_run": final["epoch"],
        "final_accuracy": final["test_acc"], "final_strict_accuracy": final["strict_acc"],
        "final_category_acc": final["category_acc"],
        "final_category_strict_acc": final["category_strict_acc"],
        "unseen_total": last_eval.get("unseen_total"),
        "category_unseen": last_eval.get("category_unseen"),
        "category_total": last_eval.get("category_total"),
        "peak_accuracy_test_selected": best_acc,
        "classes": num_classes, "csv_md5": csv_md5, "total_time_sec": total_time,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary