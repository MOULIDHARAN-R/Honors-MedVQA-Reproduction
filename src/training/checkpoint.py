"""
checkpoint.py  (src/training/checkpoint.py)

Atomic saving (os.replace is atomic on Windows and Linux) and loading of training state.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import shutil
import torch
import torch.nn as nn


def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: Path,
                    filename: str = "last.pt") -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for stale in checkpoint_dir.glob("*.tmp"):          # leftovers from interrupted saves
        try:
            stale.unlink()
        except OSError:
            pass

    target = checkpoint_dir / filename
    tmp = checkpoint_dir / (filename + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, target)

    if is_best:
        best = checkpoint_dir / "best.pt"
        best_tmp = checkpoint_dir / "best.pt.tmp"
        shutil.copyfile(target, best_tmp)
        os.replace(best_tmp, best)
    return target


def load_checkpoint(checkpoint_path: Path, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: Optional[torch.device] = None) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    map_loc = device or "cpu"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_loc, weights_only=True)
    except Exception:
        # our own checkpoints only (they contain plain Python objects as well as tensors)
        checkpoint = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint