"""
python -m scripts.modality_breakdown --run run4

For each split k = 1..5, loads checkpoints/<run>/split_k/last.pt, predicts the TEST rows, and reports
MODALITY accuracy by question type, plus the most frequent modality confusions. No training.

Question types (our rule, applied to modality rows only):
  yes/no   reference answer is 'yes' or 'no'
  choice   the question contains ' or '            e.g. "is this a contrast or noncontrast mri?"
  open     everything else                         e.g. "what imaging modality was used?"
Accuracy uses the official '#' / '()' semantics (prediction in accepted ids). Rows whose answer has no
class in the training vocabulary stay in the denominator. 'majority' = share of the most common reference
answer within that type, a reference baseline.

Output: results/reproduction/<run>/modality_breakdown/{by_type.csv, confusions.csv, predictions_split_k.csv}
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import csv
import statistics
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader

from src.answers import canonical_label
from src.config import CFG
from src.dataset import build_datasets, get_tokenizer, load_dataframe
from src.models.medvqa_model import build_model

TYPES = ["yes/no", "choice", "open"]


def qtype(question: str, answer: str) -> str:
    if answer.strip().lower() in ("yes", "no"):
        return "yes/no"
    if " or " in question.lower():
        return "choice"
    return "open"


@torch.no_grad()
def predict_all(model, ds, device, bs=16):
    out = []
    for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        logits = model(b["image"].to(device), b["input_ids"].to(device), b["attention_mask"].to(device))
        out.append(logits.argmax(1).cpu())
    return torch.cat(out).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run4")
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    cfg = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataframe(cfg)
    tok = get_tokenizer(cfg)
    out_dir = cfg.results_dir / "reproduction" / a.run / "modality_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split = {}                       # k -> {type: dict}
    confusions = Counter()
    for k in range(1, cfg.n_splits + 1):
        _, test_ds, vocab = build_datasets(cfg, k, tok)
        ckpt = torch.load(cfg.checkpoints_dir / a.run / f"split_{k}" / "last.pt",
                          map_location=device, weights_only=False)
        assert ckpt["vocab"] == vocab.idx2ans, f"split {k}: vocabulary differs from the checkpoint"
        model = build_model(num_classes=len(vocab), cfg=cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        preds = predict_all(model, test_ds, device)

        rows = df.iloc[test_ds.indices]
        n_all = len(preds)
        ok_all = sum(p in acc for p, acc in zip(preds, test_ds.accepted))
        stats = {t: {"n": 0, "ok": 0, "unseen": 0, "refs": Counter()} for t in TYPES}
        mod_n = mod_ok = 0
        with open(out_dir / f"predictions_split_{k}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["row_idx", "image_id", "qtype", "question", "ground_truth", "prediction", "correct"])
            for i, (p, acc) in enumerate(zip(preds, test_ds.accepted)):
                r = rows.iloc[i]
                if str(r["category"]).lower() != "modality":
                    continue
                t = qtype(r["question"], r["answer"])
                good = p in acc
                s = stats[t]
                s["n"] += 1; s["ok"] += int(good); s["unseen"] += int(not acc)
                s["refs"][canonical_label(r["answer"])] += 1
                mod_n += 1; mod_ok += int(good)
                if not good and acc:
                    confusions[(canonical_label(r["answer"]), vocab.idx2ans[p])] += 1
                w.writerow([test_ds.indices[i], r["image_id"], t, r["question"], r["answer"],
                            vocab.idx2ans[p], int(good)])
        per_split[k] = stats
        print(f"split {k}: overall {ok_all / n_all:.4f} | modality {mod_ok / mod_n:.4f} "
              f"(compare with your summary table) | " +
              " | ".join(f"{t} {stats[t]['ok']}/{stats[t]['n']}" for t in TYPES))
        del model, ckpt
        torch.cuda.empty_cache()

    # ---- table over splits ----
    print("\nMODALITY ACCURACY BY QUESTION TYPE (mean +- sample std over 5 splits; test sets overlap)")
    print(f"{'type':8s} {'rows/split':>10s} {'accuracy':>16s} {'unseen':>8s} {'majority':>9s}")
    table = [["type", "mean_rows", "acc_mean", "acc_std", "unseen_mean", "majority_share_mean"]]
    for t in TYPES:
        ns = [per_split[k][t]["n"] for k in per_split]
        accs = [per_split[k][t]["ok"] / per_split[k][t]["n"] for k in per_split if per_split[k][t]["n"]]
        uns = [per_split[k][t]["unseen"] for k in per_split]
        maj = [max(per_split[k][t]["refs"].values()) / per_split[k][t]["n"]
               for k in per_split if per_split[k][t]["n"]]
        sd = statistics.stdev(accs) if len(accs) > 1 else float("nan")
        print(f"{t:8s} {statistics.mean(ns):10.0f} {statistics.mean(accs):9.3f} +- {sd:.3f} "
              f"{statistics.mean(uns):8.1f} {statistics.mean(maj):9.3f}")
        table.append([t, f"{statistics.mean(ns):.1f}", f"{statistics.mean(accs):.4f}", f"{sd:.4f}",
                      f"{statistics.mean(uns):.1f}", f"{statistics.mean(maj):.4f}"])
    with open(out_dir / "by_type.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(table)

    print(f"\nTOP {a.top} MODALITY CONFUSIONS (reference -> prediction), counts summed over 5 overlapping splits")
    rows_out = [["reference", "prediction", "count"]]
    for (g, p), c in confusions.most_common(a.top):
        print(f"{c:5d}  {g[:38]:38s} -> {p[:38]}")
    for (g, p), c in confusions.most_common():
        rows_out.append([g, p, c])
    with open(out_dir / "confusions.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows_out)
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()