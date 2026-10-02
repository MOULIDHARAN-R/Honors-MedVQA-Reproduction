"""
question_encoder.py

BERT Question Encoder for Medical VQA.
Paper: "Visual Question Answering in the Medical Domain" (Canepa et al., 2023)
  * Uses BERT pretrained on English Wikipedia and BookCorpus.
  * Encodes tokenized questions into a 768-dimensional sentence representation.
"""
import torch
import torch.nn as nn
from transformers import AutoModel

try:
    from ..config import Config, CFG
except (ImportError, ValueError):
    from src.config import Config, CFG


class BERTQuestionEncoder(nn.Module):
    """
    Question encoder using BERT (default: bert-base-uncased).
    Extracts the 768-dimensional pooled representation.
    """
    def __init__(self, cfg: Config = CFG):
        super().__init__()
        self.cfg = cfg
        self.bert = AutoModel.from_pretrained(cfg.bert_name)
        self.pooling = cfg.question_pooling  # "pooler"
        self.out_dim = cfg.question_feature_dim  # 768

        if cfg.freeze_bert:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the BERT question encoder."""
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the BERT question encoder."""
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, max_len] token IDs
            attention_mask: [B, max_len] attention mask (1 for real tokens, 0 for padding)
        Returns:
            [B, 768] question representation vector
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "pooler" and getattr(outputs, "pooler_output", None) is not None:
            # Standard BERT pooler (dense + tanh on [CLS] token)
            return outputs.pooler_output
        else:
            # Fallback to raw [CLS] token representation if pooler_output is unavailable
            return outputs.last_hidden_state[:, 0, :]
