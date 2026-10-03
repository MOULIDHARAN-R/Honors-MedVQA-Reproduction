"""python -m scripts.profile_speed <num_workers> <micro_batch_size>   e.g.  4 16"""
import sys, time, torch
from torch.utils.data import DataLoader
from src.config import CFG
from src.dataset import build_datasets
from src.models.medvqa_model import build_model


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    bs = int(sys.argv[2]) if len(sys.argv) > 2 else CFG.micro_batch_size
    train_ds, _, vocab = build_datasets(CFG, 1)
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, {props.total_memory / 1e9:.1f} GB | micro-batch {bs} | workers {workers}")

    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers, pin_memory=True)
    it = iter(loader); next(it)
    t = time.time()
    for _ in range(5): batch = next(it)
    print(f"data loading: {(time.time() - t) / 5:.2f} s/micro-batch")

    model = build_model(len(vocab), CFG).cuda().train()
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    b = {k: v.cuda() for k, v in batch.items()}

    def step():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(
            model(b["image"], b["input_ids"], b["attention_mask"]), b["label"])
        loss.backward(); opt.step()

    step(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(5): step()
    torch.cuda.synchronize()
    per = (time.time() - t) / 5
    print(f"compute: {per:.2f} s/micro-batch  (~{per * (CFG.batch_size / bs):.2f} s per effective step of {CFG.batch_size})")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()