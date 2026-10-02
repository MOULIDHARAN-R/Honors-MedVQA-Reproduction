"""
losses.py

Loss functions for Medical VQA.
Paper: "Visual Question Answering in the Medical Domain" (Canepa et al., 2023)
  * Uses standard Multi-Class Cross-Entropy Loss over the discrete answer classes.
"""
import torch.nn as nn
from ..config import Config, CFG


def get_criterion(cfg: Config = CFG) -> nn.Module:
    """
    Returns the classification loss criterion.
    """
    return nn.CrossEntropyLoss()
