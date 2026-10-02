"""
metrics.py

Metric calculation for Medical VQA evaluation.
Handles VQA-Med answer semantics where:
  * alt_ids contains all acceptable class IDs for a reference answer (# alternatives and () expansions).
  * A prediction is correct if its predicted class ID is present in alt_ids.
"""
from typing import List
import torch


def is_prediction_correct(pred_id: int, accepted_ids: List[int]) -> bool:
    """Check if a predicted class ID is among the acceptable class IDs."""
    return pred_id in accepted_ids


def evaluate_predictions(preds: torch.Tensor, alt_ids: torch.Tensor) -> torch.Tensor:
    """
    Vectorized correctness evaluation for a batch.
    Args:
        preds:   [B] tensor of predicted class IDs
        alt_ids: [B, max_alts] tensor of accepted class IDs (-1 for padding)
    Returns:
        [B] boolean tensor, True if prediction matches any accepted ID
    """
    # preds: [B, 1], alt_ids: [B, max_alts]
    # match when alt_ids == preds and alt_ids >= 0
    valid_alts = alt_ids >= 0
    matches = (alt_ids == preds.unsqueeze(1)) & valid_alts
    return matches.any(dim=1)
