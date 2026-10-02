"""
smoke_test_model.py

Smoke test for the VGG-16 + BERT + Concatenation model.
Verifies:
  1. Model instantiation with dynamic num_classes.
  2. Forward pass with real input shapes.
  3. Backward pass and gradient flow through VGG-16, BERT, and FC classification head.
  4. Integration with real batch from src.dataset.
"""
import sys
import torch
import torch.nn as nn

from src.config import CFG
from src.models import build_model, MedVQAModel
from src.dataset import build_datasets


def test_model_forward_backward():
    print("=" * 60)
    print("1. Testing Model Construction and Forward/Backward Pass")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    num_classes = 1523  # Split 1 class count
    batch_size = 4
    max_len = CFG.max_question_len

    model = build_model(num_classes=num_classes, cfg=CFG).to(device)
    model.train()

    # Synthetic batch
    images = torch.randn(batch_size, 3, CFG.image_size, CFG.image_size, device=device)
    input_ids = torch.randint(0, 1000, (batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.ones(batch_size, max_len, dtype=torch.long, device=device)
    labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long, device=device)

    print(f"Synthetic Input Shapes:")
    print(f"  Images:         {images.shape}")
    print(f"  Input IDs:      {input_ids.shape}")
    print(f"  Attention Mask: {attention_mask.shape}")
    print(f"  Labels:         {labels.shape}")

    # Forward pass
    logits = model(images, input_ids, attention_mask)
    print(f"Output Logits Shape: {logits.shape} (Expected: [{batch_size}, {num_classes}])")
    assert logits.shape == (batch_size, num_classes), f"Shape mismatch: {logits.shape}"

    # Backward pass
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"CrossEntropyLoss: {loss.item():.4f}")

    loss.backward()

    # Check gradients
    vgg_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.image_encoder.parameters())
    bert_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.question_encoder.parameters())
    fc_grad = any(p.grad is not None and p.grad.norm() > 0 for p in model.classifier.parameters())

    print(f"Gradient Check:")
    print(f"  VGG-16 parameters have gradients: {vgg_grad}")
    print(f"  BERT parameters have gradients:   {bert_grad}")
    print(f"  Classifier layers have gradients: {fc_grad}")

    assert vgg_grad and bert_grad and fc_grad, "Gradients failed to flow through all components!"
    print("Synthetic forward + backward test PASSED successfully!\n")


def test_real_dataset_batch():
    print("=" * 60)
    print("2. Testing with Real Mini-Batch from src.dataset (Split 1)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, vocab = build_datasets(CFG, k=1)

    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        collate_fn=None
    )

    batch = next(iter(loader))
    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    model = build_model(num_classes=len(vocab), cfg=CFG).to(device)
    model.eval()

    with torch.no_grad():
        logits, img_f, q_f, fused = model.forward_features(images, input_ids, attention_mask)

    print(f"Real Batch Shapes:")
    print(f"  Image features:    {img_f.shape} (Expected: [4, 4096])")
    print(f"  Question features: {q_f.shape} (Expected: [4, 768])")
    print(f"  Fused features:    {fused.shape} (Expected: [4, 4864])")
    print(f"  Logits:            {logits.shape} (Expected: [4, {len(vocab)}])")

    assert img_f.shape == (4, 4096)
    assert q_f.shape == (4, 768)
    assert fused.shape == (4, 4864)
    assert logits.shape == (4, len(vocab))

    preds = torch.argmax(logits, dim=-1)
    print(f"Predicted class IDs: {preds.tolist()}")
    print(f"Target class IDs:    {labels.tolist()}")
    print("Real dataset mini-batch test PASSED successfully!\n")


if __name__ == "__main__":
    test_model_forward_backward()
    test_real_dataset_batch()
    print("ALL MODEL SMOKE TESTS PASSED!")
