"""python -m scripts.check_accumulation
Compares the gradient of one batch of 16 against accumulated micro-batches, in default precision
and in strict fp32 (TF32 off, deterministic). Eval mode = dropout off, so runs are comparable."""
import torch
from src.config import CFG
from src.dataset import build_datasets
from src.models.medvqa_model import build_model

train_ds, _, vocab = build_datasets(CFG, 1)
dev = "cuda"
model = build_model(len(vocab), CFG).to(dev).eval()
items = [train_ds[i] for i in range(16)]
batch = {k: torch.stack([it[k] for it in items]).to(dev)
         for k in ("image", "input_ids", "attention_mask", "label")}


def grads(micro):
    model.zero_grad(set_to_none=True)
    n = batch["label"].size(0)
    for s in range(0, n, micro):
        b = {k: v[s:s + micro] for k, v in batch.items()}
        loss = torch.nn.functional.cross_entropy(
            model(b["image"], b["input_ids"], b["attention_mask"]), b["label"])
        (loss * b["label"].size(0) / n).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def rel(a, b):
    return ((a - b).norm() / a.norm()).item()


def report(tag):
    g16, g16b, g8, g4 = grads(16), grads(16), grads(8), grads(4)
    print(f"[{tag}] repeat of same config (noise floor): {rel(g16, g16b):.2e}")
    print(f"[{tag}] 16 vs 2x8: {rel(g16, g8):.2e}   16 vs 4x4: {rel(g16, g4):.2e}")


report("default precision")

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("highest")
report("strict fp32     ")