"""
evaluate.py  (src/evaluation/evaluate.py)

Evaluation engine for Medical VQA.
  * overall_acc / category_acc        : official VQA-Med semantics ('#' alternatives and
                                        optional '(...)' text; correct if the predicted class is in alt_ids)
  * strict_acc / category_strict_acc  : prediction must equal the reference's canonical
                                        (first-alternative, literal) label
  * unseen counts                     : test rows whose answer has no class in the vocabulary
                                        (they can never be correct and stay in the denominator)
"""
from collections import defaultdict
from typing import Any, Dict, Optional

import pandas as pd
import torch
import torch.nn as nn

from .metrics import evaluate_predictions


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    df: pd.DataFrame,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    model.eval()
    cats = df["category"].fillna("unknown").str.lower().to_numpy()
    total, correct, strict, unseen = (defaultdict(int) for _ in range(4))

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        logits = model(batch["image"].to(device), batch["input_ids"].to(device),
                       batch["attention_mask"].to(device))
        preds = logits.argmax(dim=-1).cpu()
        alt_ids, labels = batch["alt_ids"], batch["label"]

        ok = evaluate_predictions(preds, alt_ids).tolist()               # official semantics
        st = ((preds == labels) & (labels >= 0)).tolist()                # strict
        un = (~(alt_ids >= 0).any(dim=1)).tolist()                       # no accepted class in vocab

        for r, o, s, u in zip(batch["row_idx"].tolist(), ok, st, un):
            c = cats[r]
            total[c] += 1
            correct[c] += int(o)
            strict[c] += int(s)
            unseen[c] += int(u)

    n = sum(total.values())
    acc = lambda d, t: {c: d[c] / t[c] for c in sorted(t)}
    return {
        "overall_acc": sum(correct.values()) / n if n else 0.0,
        "strict_acc": sum(strict.values()) / n if n else 0.0,
        "correct": sum(correct.values()),
        "total": n,
        "unseen_total": sum(unseen.values()),
        "category_acc": acc(correct, total),
        "category_strict_acc": acc(strict, total),
        "category_total": dict(total),
        "category_correct": dict(correct),
        "category_unseen": dict(unseen),
    }