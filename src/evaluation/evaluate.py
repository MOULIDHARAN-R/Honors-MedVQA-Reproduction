"""
evaluate.py

Evaluation engine for Medical VQA.
Computes overall test accuracy and the exact category breakdown reported in
Table III of the paper:
  * Modality
  * Plane
  * Organ System
  * Abnormality
"""
from typing import Dict, Any, Optional
from collections import defaultdict
import torch
import torch.nn as nn
import pandas as pd

from .metrics import evaluate_predictions


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    df: pd.DataFrame,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, Any]:
    """
    Evaluates the model across the dataloader.
    Returns:
        metrics: dict with overall accuracy, total count, and category breakdown.
    """
    model.eval()

    total_correct = 0
    total_samples = 0

    category_correct = defaultdict(int)
    category_total = defaultdict(int)

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        alt_ids = batch["alt_ids"].to(device)
        row_indices = batch["row_idx"].tolist()

        # Forward pass
        logits = model(images, input_ids, attention_mask)
        preds = logits.argmax(dim=-1)

        # Check correctness against accepted IDs (# alternatives and () expansions)
        correct_mask = evaluate_predictions(preds, alt_ids).cpu().tolist()

        for row_idx, is_correct in zip(row_indices, correct_mask):
            cat = str(df.iloc[row_idx].get("category", "unknown")).lower()
            total_samples += 1
            category_total[cat] += 1
            if is_correct:
                total_correct += 1
                category_correct[cat] += 1

    overall_acc = (total_correct / total_samples) if total_samples > 0 else 0.0

    category_acc = {}
    for cat in sorted(category_total.keys()):
        tot = category_total[cat]
        corr = category_correct[cat]
        category_acc[cat] = (corr / tot) if tot > 0 else 0.0

    return {
        "overall_acc": overall_acc,
        "correct": total_correct,
        "total": total_samples,
        "category_acc": category_acc,
        "category_total": dict(category_total),
        "category_correct": dict(category_correct),
    }
