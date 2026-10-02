"""
medvqa_model.py

Complete VGG-16 + BERT + Concatenation Med-VQA Architecture.
Paper: "Visual Question Answering in the Medical Domain" (Canepa et al., 2023)
  * Best-performing architecture (60% accuracy on VQA-Med 2019).
  * VGG-16 image encoder (4096-d) + BERT question encoder (768-d).
  * Vector concatenation fusion (4864-d).
  * Two fully connected layers mapping to answer classes.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn

try:
    from ..config import Config, CFG
    from .image_encoder import VGG16ImageEncoder
    from .question_encoder import BERTQuestionEncoder
    from .fusion import ConcatFusion
except (ImportError, ValueError):
    from src.config import Config, CFG
    from src.models.image_encoder import VGG16ImageEncoder
    from src.models.question_encoder import BERTQuestionEncoder
    from src.models.fusion import ConcatFusion


class MedVQAModel(nn.Module):
    """
    End-to-End Medical Visual Question Answering Model.
    Architecture:
        Image (224x224x3)  ──► VGG-16 (fc7)       ──► [B, 4096] ────┐
                                                                     ├──► Concat [B, 4864] ──► 2 FC Layers ──► Logits [B, num_classes]
        Question (tokens)  ──► BERT (pooled [CLS]) ──► [B, 768]  ────┘
    """
    def __init__(self, num_classes: int, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # 1. Encoders
        self.image_encoder = VGG16ImageEncoder(cfg)
        self.question_encoder = BERTQuestionEncoder(cfg)

        # 2. Multimodal Fusion (Concatenation)
        self.fusion = ConcatFusion(cfg)
        fused_dim = self.fusion.out_dim  # 4096 + 768 = 4864

        # 3. Two Fully Connected Classification Layers
        # FC 1: Linear(4864, 1024) -> ReLU -> Dropout(0.3)
        # FC 2: Linear(1024, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, cfg.fc_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=cfg.dropout),
            nn.Linear(cfg.fc_hidden_dim, num_classes)
        )

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Standard forward pass returning class prediction logits.
        Args:
            image:          [B, 3, 224, 224] normalized image tensor
            input_ids:      [B, max_len] token IDs
            attention_mask: [B, max_len] attention mask
        Returns:
            logits:         [B, num_classes] unnormalized class scores
        """
        img_feat = self.image_encoder(image)
        q_feat = self.question_encoder(input_ids, attention_mask)
        fused = self.fusion(img_feat, q_feat)
        logits = self.classifier(fused)
        return logits

    def forward_features(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass exposing intermediate features for analysis and Grad-CAM explainability.
        Returns:
            (logits, img_feat, q_feat, fused)
        """
        img_feat = self.image_encoder(image)
        q_feat = self.question_encoder(input_ids, attention_mask)
        fused = self.fusion(img_feat, q_feat)
        logits = self.classifier(fused)
        return logits, img_feat, q_feat, fused


def build_model(num_classes: int, cfg: Config = CFG) -> MedVQAModel:
    """Factory helper to construct the MedVQAModel."""
    return MedVQAModel(num_classes=num_classes, cfg=cfg)
