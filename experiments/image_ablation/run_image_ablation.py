"""
experiments/image_ablation/run_image_ablation.py

Re-scores the FINISHED models (checkpoints/<run>/split_k/last.pt) on each split's TEST rows under three conditions:
  real      the true image
  shuffled  every test image is replaced by a DIFFERENT test image (fixed seed = split number)
  blank     an all-zero (= dataset-mean) image
Same question, same model, same scoring (official '#' / '()' rules). No training.

If accuracy barely drops when the image is shuffled, the model is not using the image for that group of questions.
Per category and per modality question type (yes/no, choice, open). Saves image_ablation.csv

Usage:
    python experiments/image_ablation/run_image_ablation.py --run run4
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import copy
import csv
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.dataset import build_datasets, get_tokenizer, load_dataframe
from src.models.medvqa_model import build_model

CATS = ["modality", "plane", "organ", "abnormality"]
TYPES = ["yes/no", "choice", "open"]
CONDS = ["real", "shuffled", "blank"]


def qtype(question, answer):
    if answer.strip().lower() in ("yes", "no"):
        return "yes/no"
    return "choice" if " or " in question.lower() else "open"


def derange_paths(paths, seed):
    """Map every distinct image to a different distinct image (a cyclic shift of a shuffled list)."""
    uniq = sorted(set(paths))
    order = uniq[:]
    random.Random(seed).shuffle(order)
    m = {order[j]: order[(j + 1) % len(order)] for j in range(len(order))}
    return [m[p] for p in paths]


@torch.no_grad()
def predict(model, ds, device, blank=False, bs=16):
    out = []
    for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        img = torch.zeros_like(b["image"]) if blank else b["image"]
        logits = model(img.to(device), b["input_ids"].to(device), b["attention_mask"].to(device))
        out.append(logits.argmax(1).cpu())
    return torch.cat(out).tolist()


def main():
    ap = argparse.ArgumentParser(description="Image Perturbation Ablation")
    ap.add_argument("--run", default="run4", help="Run name (default: run4)")
    a = ap.parse_args()
    cfg = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataframe(cfg)
    tok = get_tokenizer(cfg)

    res = {c: defaultdict(list) for c in CONDS}
    for k in range(1, cfg.n_splits + 1):
        _, test_ds, vocab = build_datasets(cfg, k, tok)
        ckpt = torch.load(cfg.checkpoints_dir / a.run / f"split_{k}" / "last.pt",
                          map_location=device, weights_only=False)
        assert ckpt["vocab"] == vocab.idx2ans, f"split {k}: vocabulary differs from the checkpoint"
        model = build_model(num_classes=len(vocab), cfg=cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        shuf_ds = copy.copy(test_ds)
        shuf_ds.image_paths = derange_paths(test_ds.image_paths, seed=k)
        assert all(p != q for p, q in zip(test_ds.image_paths, shuf_ds.image_paths))

        rows = df.iloc[test_ds.indices]
        cat = [str(c).lower() for c in rows["category"]]
        qt = [qtype(q, an) if c == "modality" else None
              for q, an, c in zip(rows["question"], rows["answer"], cat)]

        for cond, ds, blank in (("real", test_ds, False), ("shuffled", shuf_ds, False), ("blank", test_ds, True)):
            preds = predict(model, ds, device, blank)
            ok, tot = defaultdict(int), defaultdict(int)
            for i, (p, acc) in enumerate(zip(preds, test_ds.accepted)):
                keys = ["overall", cat[i]] + ([f"modality: {qt[i]}"] if qt[i] else [])
                for key in keys:
                    tot[key] += 1
                    ok[key] += int(p in acc)
            for key in tot:
                res[cond][key].append(ok[key] / tot[key])
            print(f"split {k} {cond:8s}: overall {ok['overall'] / tot['overall']:.4f}")
        del model, ckpt
        torch.cuda.empty_cache()

    order = ["overall"] + CATS + [f"modality: {t}" for t in TYPES]
    print(f"\n{'':20s} {'real':>8s} {'shuffled':>9s} {'blank':>8s} {'real-shuffled':>14s}")
    table = [["group", "real", "shuffled", "blank"]]
    for key in order:
        m = {c: statistics.mean(res[c][key]) for c in CONDS if res[c].get(key)}
        if len(m) < 3:
            continue
        print(f"{key:20s} {m['real']:8.3f} {m['shuffled']:9.3f} {m['blank']:8.3f} "
              f"{(m['real'] - m['shuffled']) * 100:+11.1f} pts")
        table.append([key] + [f"{m[c]:.4f}" for c in CONDS])

    out_dirs = [
        PROJECT_ROOT / "results" / "reproduction" / a.run,
        PROJECT_ROOT / "experiments" / "image_ablation" / "results",
    ]
    for out_d in out_dirs:
        out_d.mkdir(parents=True, exist_ok=True)
        out = out_d / "image_ablation.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(table)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
