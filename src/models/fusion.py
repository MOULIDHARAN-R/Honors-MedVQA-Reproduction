"""
fusion.py

Multimodal Feature Fusion for Medical VQA.
Paper: "Visual Question Answering in the Medical Domain" (Canepa et al., 2023)
  * Baseline & Best Model: Simple concatenation of visual and textual feature vectors.
"""
import torch
import torch.nn as nn

try:
    from ..config import Config, CFG
except (ImportError, ValueError):
    from src.config import Config, CFG


class ConcatFusion(nn.Module):
    """
    Concatenates 1D image feature vector and 1D question feature vector.
    Input dims: image_dim (4096) + question_dim (768) -> 4864
    """
    def __init__(self, cfg: Config = CFG):
        super().__init__()
        self.image_dim = cfg.image_feature_dim
        self.question_dim = cfg.question_feature_dim
        self.out_dim = self.image_dim + self.question_dim  # 4864

    def forward(self, img_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_feat: [B, image_dim]
            q_feat:   [B, question_dim]
        Returns:
            fused:    [B, image_dim + question_dim]
        """
        return torch.cat([img_feat, q_feat], dim=-1)
