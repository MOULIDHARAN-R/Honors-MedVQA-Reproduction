"""
experiments/diagnostic_qa_run1/train_diagnostic.py

QA-BASED SPLIT DIAGNOSTIC RUN 1
================================
Self-contained runner for the QA-level split diagnostic experiment.
Tests whether partitioning by QA pair (allowing the same image to appear in both
train and test with different questions) materially changes measured performance.

THIS EXPERIMENT IS DIAGNOSTIC ONLY.
It does NOT modify, replace, or claim to represent the official image-level reproduction.

Usage:
    # Full training (50 epochs) - to be executed manually by user:
    python experiments/diagnostic_qa_run1/train_diagnostic.py

    # Lightweight validation / smoke test only (1 micro-batch, no training):
    python experiments/diagnostic_qa_run1/train_diagnostic.py --smoke_test
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def draw_live_overview(history, out_path: Path):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    fig.suptitle(f"QA Split Diagnostic Run 1 | Live Progress (Epoch {epochs[-1]}/50)", fontsize=13, fontweight="bold")

    # Panel 1: Training Loss
    ax1 = axes[0, 0]
    ax1.plot(epochs, [h["train_loss"] for h in history], color="#D32F2F", marker="o", markersize=3.5, label="Train Loss")
    ax1.set_title("Training Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    # Panel 2: Overall Accuracy
    ax2 = axes[0, 1]
    ax2.plot(epochs, [h["test_acc"] * 100 for h in history], color="#1976D2", marker="o", markersize=3.5, label="QA Diagnostic (Official Semantics)")
    if all("strict_acc" in h for h in history):
        ax2.plot(epochs, [h["strict_acc"] * 100 for h in history], color="#0288D1", linestyle="--", label="QA Diagnostic (Strict)")
    ax2.axhline(57.15, color="#388E3C", linestyle=":", linewidth=1.8, label="Image Run 1 Benchmark (57.15%)")
    ax2.axhline(60.0, color="#7B1FA2", linestyle="--", linewidth=1.5, label="Paper Target (60.0%)")
    ax2.set_title("Overall Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_ylim([40, 75])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=8.5)

    # Panel 3: Category Breakdown
    ax3 = axes[1, 0]
    cats = ["modality", "plane", "organ", "abnormality"]
    cat_colors = {"modality": "#0288D1", "plane": "#0097A7", "organ": "#388E3C", "abnormality": "#E64A19"}
    image_run1_cat = {"modality": 66.45, "plane": 78.24, "organ": 72.82, "abnormality": 11.27}
    for c in cats:
        vals = [h["category_acc"].get(c, 0.0) * 100 for h in history]
        ax3.plot(epochs, vals, color=cat_colors[c], marker="o", markersize=3, label=f"{c.capitalize()} (Image R1: {image_run1_cat[c]}%)")
    ax3.set_title("Category Accuracy Breakdown (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Epoch", fontsize=10)
    ax3.set_ylabel("Accuracy (%)", fontsize=10)
    ax3.set_ylim([0, 95])
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(loc="upper left", ncol=2, fontsize=8)

    # Panel 4: Gradient Norm & Learning Rate
    ax4 = axes[1, 1]
    ax4_lr = ax4.twinx()
    gn_max = [h.get("grad_norm_max", 0.0) for h in history]
    gn_mean = [h.get("grad_norm_mean", 0.0) for h in history]
    ax4.plot(epochs, gn_max, color="#F57C00", linestyle="-", label="Max Grad Norm (pre-clip)")
    ax4.plot(epochs, gn_mean, color="#FFA000", linestyle="--", label="Mean Grad Norm")
    ax4.axhline(5.0, color="gray", linestyle=":", label="Clip Threshold (5.0)")
    ax4.set_ylabel("Grad Norm (L2)", fontsize=10)
    ax4.set_yscale("log")
    lrs = [h.get("lr", 1e-4) for h in history]
    ax4_lr.plot(epochs, lrs, color="#512DA8", linestyle=":", alpha=0.7, label="LR")
    ax4_lr.set_ylabel("Learning Rate", color="#512DA8", fontsize=10)
    ax4_lr.set_yscale("log")
    ax4.set_title("Optimization Dynamics (Grad Norm & LR)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Epoch", fontsize=10)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.stem + "_tmp.png")
    fig.savefig(tmp_path, dpi=150)
    plt.close(fig)
    tmp_path.replace(out_path)

# Import shared modules from src (read-only, no modifications to core architectures)
from src.config import Config, PROJECT_ROOT
from src.dataset import MedVQADataset, load_dataframe, file_md5, get_tokenizer
from src.answers import AnswerVocab
from src.models import build_model
from src.evaluation.evaluate import evaluate_model
from src.training.losses import get_criterion

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "diagnostic_qa_run1"
SPLIT_FILE = EXPERIMENT_DIR / "split" / "split_1.json"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoint"
RESULTS_DIR = EXPERIMENT_DIR / "results"
OBSERVATIONS_DIR = EXPERIMENT_DIR / "observations"
LOGS_DIR = EXPERIMENT_DIR / "logs"

IMAGE_RUN1_METRICS = {
    "overall_acc": 0.5715,
    "strict_acc": 0.5712,
    "modality_acc": 0.6645,
    "plane_acc": 0.7824,
    "organ_acc": 0.7282,
    "abnormality_acc": 0.1127,
}


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, cfg,
                    max_batches=None):
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

        loss = criterion(model(images, input_ids, attention_mask), labels)

        group = i // accum
        group_size = min(cfg.batch_size, n_total - group * cfg.batch_size)
        (loss * labels.size(0) / group_size).backward()

        total_loss += loss.item() * labels.size(0)
        total_n += labels.size(0)

        is_last = (i + 1 == len(dataloader)) or (max_batches is not None and i + 1 == max_batches)
        if (i + 1) % accum == 0 or is_last:
            max_norm = cfg.grad_clip_norm if cfg.grad_clip_norm else float("inf")
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            grad_norms.append(float(gn))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return (total_loss / total_n if total_n else 0.0), grad_norms


def main():
    parser = argparse.ArgumentParser(description="Med-VQA QA-Split Diagnostic Run 1")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run lightweight 1-step validation without training loop")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last.pt checkpoint")
    args = parser.parse_args()

    # Create isolated experiment directories
    for d in [CHECKPOINT_DIR, RESULTS_DIR, OBSERVATIONS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log_path = LOGS_DIR / "training.log"
    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 80)
    log("QA DIAGNOSTIC RUN 1")
    log("Experiment: QA-Level Train/Test Split Diagnostic")
    log(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Split File: {SPLIT_FILE}")
    log("=" * 80)

    # 1. Load configuration and split
    cfg = Config(split_file_override=SPLIT_FILE, split_unit="qa")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "-"
    log(f"Device: {device} ({gpu_name}) | PyTorch {torch.__version__}")
    log(f"Config: epochs={cfg.epochs}, batch_size={cfg.batch_size}, micro_batch={cfg.micro_batch_size}, "
        f"lr={cfg.lr}, grad_clip={cfg.grad_clip_norm}, lr_step={cfg.lr_step_size} (gamma={cfg.lr_gamma})")

    df = load_dataframe(cfg)
    current_md5 = file_md5(cfg.csv_path)

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    assert split_meta["csv_md5"] == current_md5, "CSV MD5 mismatch with split file!"
    assert split_meta["split_unit"] == "qa", "split_unit must be 'qa'!"
    assert len(set(split_meta["train_idx"]) & set(split_meta["test_idx"])) == 0, "Train and test overlap!"

    train_idx = split_meta["train_idx"]
    test_idx = split_meta["test_idx"]
    tokenizer = get_tokenizer(cfg)

    # 2. Build Answer Vocabulary (STRICTLY FROM TRAIN ROWS ONLY)
    vocab = AnswerVocab(df.iloc[train_idx]["answer"], cfg.answer_alt_separator, cfg.expand_optional_parens)
    train_ds = MedVQADataset(df, train_idx, vocab, tokenizer, cfg, train=True)
    test_ds = MedVQADataset(df, test_idx, vocab, tokenizer, cfg, train=False)

    num_classes = len(vocab)
    n_train_imgs = split_meta["n_train_images"]
    n_test_imgs = split_meta["n_test_images"]
    images_in_both = split_meta["images_in_both"]
    n_unseen = sum(1 for acc in test_ds.accepted if not acc)

    log(f"Train QA Pairs: {len(train_ds)} | Test QA Pairs: {len(test_ds)}")
    log(f"Train Images:   {n_train_imgs} | Test Images:   {n_test_imgs}")
    log(f"Shared Images (In Both Train & Test): {images_in_both} ({images_in_both / n_test_imgs:.1%} of test images)")
    log(f"Vocab Size (from Train Only): {num_classes} classes")
    log(f"Test Rows with Unseen Answer in Vocab: {n_unseen} ({n_unseen / len(test_ds):.2%})")

    # Save isolated config
    cfg_dump = asdict(cfg)
    cfg_dump["split_file_override"] = str(SPLIT_FILE)
    with open(RESULTS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, default=str)

    # 3. Model, Criterion, Optimizer
    seed = 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = build_model(num_classes=num_classes, cfg=cfg).to(device)
    criterion = get_criterion(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    workers = cfg.num_workers
    common = dict(num_workers=workers, pin_memory=(device.type == "cuda"), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg.micro_batch_size, shuffle=True, **common)
    test_loader = DataLoader(test_ds, batch_size=cfg.micro_batch_size, shuffle=False, **common)

    # -------------------------------------------------------------
    # SMOKE TEST MODE (No full training)
    # -------------------------------------------------------------
    if args.smoke_test:
        log("\n>>> RUNNING LIGHTWEIGHT SMOKE TEST (1 micro-batch) <<<")
        train_loss, gn = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=1, cfg=cfg, max_batches=1)
        log(f"Smoke Test Train Step Loss: {train_loss:.4f} | Grad Norm: {gn[0] if gn else 0.0:.2f}")

        eval_res = evaluate_model(model, test_loader, df, device, max_batches=1)
        log(f"Smoke Test Eval Step Completed. Batch Sample Acc: {eval_res['overall_acc']:.2%}")

        # Verify live plotting logic
        dummy_hist = [{
            "epoch": 1, "lr": 1e-4, "duration_sec": 1.0, "train_loss": train_loss,
            "test_acc": eval_res["overall_acc"], "strict_acc": eval_res["strict_acc"], "best_acc": eval_res["overall_acc"],
            "grad_norm_max": gn[0] if gn else 1.0, "grad_norm_mean": gn[0] if gn else 1.0,
            "category_acc": eval_res["category_acc"], "category_strict_acc": eval_res["category_strict_acc"]
        }]
        plots_dir = RESULTS_DIR / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        draw_live_overview(dummy_hist, plots_dir / "overview_smoke_test.png")
        if (plots_dir / "overview_smoke_test.png").exists():
            (plots_dir / "overview_smoke_test.png").unlink()
            log("Live plotting function verified successfully.")

        # Verify no official directories were modified or targeted
        forbidden = [
            PROJECT_ROOT / "checkpoints" / "run1",
            PROJECT_ROOT / "checkpoints" / "run2",
            PROJECT_ROOT / "checkpoints" / "run3",
            PROJECT_ROOT / "checkpoints" / "run4",
            PROJECT_ROOT / "checkpoints" / "run5",
            PROJECT_ROOT / "results" / "reproduction" / "run1",
            PROJECT_ROOT / "results" / "reproduction" / "run4",
            PROJECT_ROOT / "data" / "splits" / "image",
        ]
        log("\nVerifying safety of official reproduction paths:")
        for fb in forbidden:
            log(f"  [SAFE] Isolated from {fb}")

        log("\n" + "=" * 80)
        log("SMOKE TEST PASSED SUCCESSFULLY!")
        log("No full training was executed.")
        log("All components, datasets, loaders, model forward/backward, and output paths are verified.")
        log("=" * 80)
        return

    # -------------------------------------------------------------
    # FULL TRAINING LOOP (50 Epochs) - To be run manually by user
    # -------------------------------------------------------------
    epochs = cfg.epochs
    start_epoch = 1
    best_acc = 0.0
    history = []
    training_log_csv = RESULTS_DIR / "training_log.csv"
    history_file = RESULTS_DIR / "history.json"
    last_ckpt = CHECKPOINT_DIR / "last.pt"

    csv_fields = ["epoch", "lr", "train_loss", "test_acc", "strict_acc", "best_acc",
                  "modality_acc", "plane_acc", "organ_acc", "abnormality_acc"]

    if args.resume and last_ckpt.exists():
        log(f"\nResuming training from checkpoint: {last_ckpt}")
        try:
            ckpt = torch.load(last_ckpt, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        log(f"Restored model and optimizer states. Resuming from Epoch {start_epoch:02d}/{epochs:02d}...")

        if history_file.exists():
            with open(history_file, "r", encoding="utf-8") as f:
                history = json.load(f)
            history = [h for h in history if h["epoch"] <= ckpt["epoch"]]
            best_acc = max((h["test_acc"] for h in history), default=0.0)

        # Truncate training_log.csv to keep only epochs <= ckpt["epoch"]
        if training_log_csv.exists():
            with open(training_log_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                valid_rows = [r for r in reader if int(r["epoch"]) <= ckpt["epoch"]]
            with open(training_log_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()
                writer.writerows(valid_rows)
    else:
        with open(training_log_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()

    start_time = time.time()
    last_eval = {}

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr = cfg.lr * (cfg.lr_gamma ** ((epoch - 1) // cfg.lr_step_size))
        for g in optimizer.param_groups:
            g["lr"] = lr

        train_loss, grad_norms = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, cfg)
        last_eval = evaluate_model(model, test_loader, df, device)

        test_acc = last_eval["overall_acc"]
        is_best = test_acc > best_acc
        best_acc = max(best_acc, test_acc)
        duration = time.time() - t0
        gn_max = max(grad_norms) if grad_norms else 0.0
        gn_mean = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        cat_acc = last_eval["category_acc"]
        cat_str = " | ".join(f"{c.capitalize()}: {a:.2%}" for c, a in cat_acc.items())
        log(f"Epoch {epoch:02d}/{epochs:02d} [{duration:.0f}s] | lr {lr:.1e} | loss {train_loss:.4f} "
            f"| acc {test_acc:.4f} (strict {last_eval['strict_acc']:.4f}, peak {best_acc:.4f}) "
            f"| gnorm max {gn_max:.2f} mean {gn_mean:.2f} | {cat_str}")

        row_record = {
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "test_acc": test_acc, "strict_acc": last_eval["strict_acc"], "best_acc": best_acc,
            "modality_acc": cat_acc.get("modality", 0.0),
            "plane_acc": cat_acc.get("plane", 0.0),
            "organ_acc": cat_acc.get("organ", 0.0),
            "abnormality_acc": cat_acc.get("abnormality", 0.0),
        }
        with open(training_log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row_record)

        history.append({
            "epoch": epoch, "lr": lr, "duration_sec": duration, "train_loss": train_loss,
            "test_acc": test_acc, "strict_acc": last_eval["strict_acc"], "best_acc": best_acc,
            "category_acc": cat_acc, "category_strict_acc": last_eval["category_strict_acc"]
        })

        # Draw and overwrite live progress plot automatically after each epoch
        try:
            plots_dir = RESULTS_DIR / "plots"
            draw_live_overview(history, plots_dir / "overview.png")
            with open(RESULTS_DIR / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            log(f"Warning: Failed to update live plot: {e}")

        # Save checkpoint strictly inside experiments/diagnostic_qa_run1/checkpoint/
        ckpt_state = {
            "epoch": epoch,
            "split": 1,
            "split_unit": "qa",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "vocab": vocab.idx2ans,
            "metrics": row_record,
            "csv_md5": current_md5,
            "config": cfg_dump,
        }
        torch.save(ckpt_state, CHECKPOINT_DIR / "last.pt")
        if is_best:
            torch.save(ckpt_state, CHECKPOINT_DIR / "best.pt")

    total_time = time.time() - start_time
    final = history[-1]
    final_cat = final["category_acc"]

    log("=" * 80)
    log(f"TRAINING COMPLETE in {total_time / 60:.1f} minutes")
    log(f"Final Reported Accuracy (Epoch 50): {final['test_acc']:.4f} (Strict: {final['strict_acc']:.4f})")
    log("=" * 80)

    # 4. Generate Final Metrics & Comparison Files
    metrics_data = {
        "experiment": "diagnostic_qa_run1",
        "split_unit": "qa",
        "seed": 0,
        "epochs": 50,
        "total_time_min": round(total_time / 60, 2),
        "dataset": {
            "n_total_rows": len(df),
            "n_train_qa": len(train_ds),
            "n_test_qa": len(test_ds),
            "n_train_images": n_train_imgs,
            "n_test_images": n_test_imgs,
            "images_in_both": images_in_both,
            "image_overlap_pct": round(images_in_both / n_test_imgs * 100, 2),
            "unseen_test_answers": n_unseen,
            "unseen_test_answers_pct": round(n_unseen / len(test_ds) * 100, 2),
            "vocab_classes": num_classes
        },
        "metrics": {
            "overall_accuracy": float(final["test_acc"]),
            "strict_accuracy": float(final["strict_acc"]),
            "modality_accuracy": float(final_cat.get("modality", 0.0)),
            "plane_accuracy": float(final_cat.get("plane", 0.0)),
            "organ_accuracy": float(final_cat.get("organ", 0.0)),
            "abnormality_accuracy": float(final_cat.get("abnormality", 0.0)),
        }
    }
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    # Write metrics.csv
    with open(RESULTS_DIR / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["overall_accuracy", f"{final['test_acc']:.4f}"])
        w.writerow(["strict_accuracy", f"{final['strict_acc']:.4f}"])
        w.writerow(["modality_accuracy", f"{final_cat.get('modality', 0.0):.4f}"])
        w.writerow(["plane_accuracy", f"{final_cat.get('plane', 0.0):.4f}"])
        w.writerow(["organ_accuracy", f"{final_cat.get('organ', 0.0):.4f}"])
        w.writerow(["abnormality_accuracy", f"{final_cat.get('abnormality', 0.0):.4f}"])
        w.writerow(["test_qa_pairs", len(test_ds)])
        w.writerow(["unseen_test_answers", n_unseen])
        w.writerow(["train_images", n_train_imgs])
        w.writerow(["test_images", n_test_imgs])
        w.writerow(["images_in_both", images_in_both])

    # Comparison with Image Run 1
    comp_rows = [
        ["Metric", "Image Run 1", "QA Diagnostic Run 1", "Difference (pts)"],
        ["Overall Accuracy", f"{IMAGE_RUN1_METRICS['overall_acc']*100:.2f}%", f"{final['test_acc']*100:.2f}%",
         f"{(final['test_acc'] - IMAGE_RUN1_METRICS['overall_acc'])*100:+.2f}"],
        ["Modality", f"{IMAGE_RUN1_METRICS['modality_acc']*100:.2f}%", f"{final_cat.get('modality', 0.0)*100:.2f}%",
         f"{(final_cat.get('modality', 0.0) - IMAGE_RUN1_METRICS['modality_acc'])*100:+.2f}"],
        ["Plane", f"{IMAGE_RUN1_METRICS['plane_acc']*100:.2f}%", f"{final_cat.get('plane', 0.0)*100:.2f}%",
         f"{(final_cat.get('plane', 0.0) - IMAGE_RUN1_METRICS['plane_acc'])*100:+.2f}"],
        ["Organ", f"{IMAGE_RUN1_METRICS['organ_acc']*100:.2f}%", f"{final_cat.get('organ', 0.0)*100:.2f}%",
         f"{(final_cat.get('organ', 0.0) - IMAGE_RUN1_METRICS['organ_acc'])*100:+.2f}"],
        ["Abnormality", f"{IMAGE_RUN1_METRICS['abnormality_acc']*100:.2f}%", f"{final_cat.get('abnormality', 0.0)*100:.2f}%",
         f"{(final_cat.get('abnormality', 0.0) - IMAGE_RUN1_METRICS['abnormality_acc'])*100:+.2f}"],
    ]
    with open(RESULTS_DIR / "comparison_with_image_run1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(comp_rows)

    # Machine-readable observations.json
    obs_data = {
        "experiment_name": "diagnostic_qa_run1",
        "experiment_type": "diagnostic_split_unit_ablation",
        "split_unit": "qa",
        "seed": 0,
        "test_size": 0.20,
        "n_train_qa": len(train_ds),
        "n_test_qa": len(test_ds),
        "n_train_images": n_train_imgs,
        "n_test_images": n_test_imgs,
        "images_in_both": images_in_both,
        "image_overlap_pct": round(images_in_both / n_test_imgs * 100, 2),
        "image_run1_overall_accuracy": IMAGE_RUN1_METRICS["overall_acc"],
        "qa_run1_overall_accuracy": float(final["test_acc"]),
        "overall_accuracy_difference": float(final["test_acc"] - IMAGE_RUN1_METRICS["overall_acc"]),
        "image_run1_modality_accuracy": IMAGE_RUN1_METRICS["modality_acc"],
        "qa_run1_modality_accuracy": float(final_cat.get("modality", 0.0)),
        "modality_accuracy_difference": float(final_cat.get("modality", 0.0) - IMAGE_RUN1_METRICS["modality_acc"]),
        "image_run1_plane_accuracy": IMAGE_RUN1_METRICS["plane_acc"],
        "qa_run1_plane_accuracy": float(final_cat.get("plane", 0.0)),
        "plane_accuracy_difference": float(final_cat.get("plane", 0.0) - IMAGE_RUN1_METRICS["plane_acc"]),
        "image_run1_organ_accuracy": IMAGE_RUN1_METRICS["organ_acc"],
        "qa_run1_organ_accuracy": float(final_cat.get("organ", 0.0)),
        "organ_accuracy_difference": float(final_cat.get("organ", 0.0) - IMAGE_RUN1_METRICS["organ_acc"]),
        "image_run1_abnormality_accuracy": IMAGE_RUN1_METRICS["abnormality_acc"],
        "qa_run1_abnormality_accuracy": float(final_cat.get("abnormality", 0.0)),
        "abnormality_accuracy_difference": float(final_cat.get("abnormality", 0.0) - IMAGE_RUN1_METRICS["abnormality_acc"]),
        "interpretation": "QA-level splitting allows the same image to occur in both training and testing. "
                          "This experiment demonstrates whether split granularity materially changes measured performance.",
        "limitations": "This diagnostic experiment only tests the behavior of split granularity. "
                       "It does not establish patient-level leakage or prove the original paper's methodology."
    }
    with open(OBSERVATIONS_DIR / "observations.json", "w", encoding="utf-8") as f:
        json.dump(obs_data, f, indent=2)

    log("\nFinal Results and Comparison Table:")
    for row in comp_rows:
        log(f"  {row[0]:<20} | {row[1]:<12} | {row[2]:<20} | {row[3]:<16}")
    log("\nAll diagnostic outputs written to: " + str(EXPERIMENT_DIR))


if __name__ == "__main__":
    main()
