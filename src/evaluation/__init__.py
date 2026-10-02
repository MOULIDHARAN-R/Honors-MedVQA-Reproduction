"""
src/evaluation package
"""
from .metrics import evaluate_predictions, is_prediction_correct
from .evaluate import evaluate_model

__all__ = [
    "evaluate_predictions",
    "is_prediction_correct",
    "evaluate_model",
]
