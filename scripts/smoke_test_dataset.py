from src.config import CFG
from src.dataset import build_datasets

train_ds, test_ds, vocab = build_datasets(CFG, 1)
item = train_ds[0]
for key, value in item.items():
    shape = tuple(value.shape) if hasattr(value, "shape") else value
    print(f"{key:15s} {shape}")
print("first 5 classes:", vocab.idx2ans[:5])