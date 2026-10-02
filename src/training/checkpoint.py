"""
checkpoint.py

Checkpoint management for Medical VQA models.
Provides atomic saving and safe loading for model weights, optimizer state,
answer vocabulary, and evaluation metrics.
"""
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
import tempfile
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Path,
    filename: str = "last.pt"
) -> Path:
    """
    Atomically saves training state to disk.
    Args:
        state: dictionary containing model_state_dict, optimizer_state_dict, epoch, etc.
        is_best: boolean, if True copies the checkpoint to best.pt
        checkpoint_dir: directory to save the checkpoint in
        filename: default "last.pt"
    Returns:
        Path to the saved checkpoint file
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target_path = checkpoint_dir / filename

    # Atomic write to temporary file first, then replace target
    with tempfile.NamedTemporaryFile(dir=checkpoint_dir, delete=False, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        torch.save(state, tmp_path)

    shutil.move(str(tmp_path), str(target_path))

    if is_best:
        best_path = checkpoint_dir / "best.pt"
        shutil.copyfile(str(target_path), str(best_path))

    return target_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Loads saved weights into model and optimizer.
    Args:
        checkpoint_path: path to .pt checkpoint file
        model: PyTorch model instance
        optimizer: optional optimizer instance to restore momentum/velocity
        device: device to map tensors onto
    Returns:
        The loaded checkpoint dictionary (epoch, metrics, vocab, config)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
