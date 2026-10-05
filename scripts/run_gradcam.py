"""
python -m scripts.run_gradcam --run run4 --split 1 --per_cat 2 --seed 0
python -m scripts.run_gradcam --run run4 --split 1 --image_ids synpic12345 synpic67890
python -m scripts.run_gradcam --run run4 --split 1 --target gt --verify_package

Loads checkpoints/<run>/split_<k>/last.pt (the final-epoch model that is reported), runs the TEST rows of
that split, and draws Grad-CAM heat maps over the (resized) input images.

Default selection (unbiased, reproducible): per category, `per_cat` random correctly-answered and `per_cat`
random wrongly-answered test rows, seeded by --seed. State this selection rule when you show the figure.

Output: results/reproduction/<run>/split_<k>/gradcam/
    gradcam_grid.png   one row per category: correct examples first, wrong examples after
    selection.csv      row_idx, image_id, category, question, ground truth, prediction, correct, cam_raw_max
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # same OpenMP workaround the trainer uses (Windows)

import argparse
import csv
import random
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CFG
from src.dataset import build_datasets, load_dataframe
from src.explainability.gradcam import GradCAM, last_conv_layer, prepare_model_for_gradcam
from src.models.medvqa_model import build_model

CATS = ["modality", "plane", "organ", "abnormality"]
NAMES = {"organ": "organ system"}


def denorm(img, mean, std):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (img.cpu() * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def predict_all(model, ds, device, bs=16):
    preds = []
    for b in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        logits = model(b["image"].to(device), b["input_ids"].to(device), b["attention_mask"].to(device))
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds).tolist()


def wrap(text, width=30, lines=3):
    out = textwrap.wrap(str(text), width)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][: max(0, width - 3)] + "..."
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run4")
    ap.add_argument("--split", type=int, default=1)
    ap.add_argument("--per_cat", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", choices=["pred", "gt"], default="pred",
                    help="class whose score is explained: predicted answer (default) or first accepted true answer")
    ap.add_argument("--image_ids", nargs="*", default=None, help="explain these test images instead")
    ap.add_argument("--verify_package", action="store_true",
                    help="cross-check one sample against the pytorch-grad-cam package (pip install grad-cam)")
    a = ap.parse_args()

    cfg = CFG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_dataframe(cfg)
    _, test_ds, vocab = build_datasets(cfg, a.split)

    ckpt_path = cfg.checkpoints_dir / a.run / f"split_{a.split}" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    assert ckpt["vocab"] == vocab.idx2ans, "vocabulary differs from the checkpoint (wrong split/run?)"
    model = build_model(num_classes=len(vocab), cfg=cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    prepare_model_for_gradcam(model)
    print(f"loaded {ckpt_path} (epoch {ckpt['epoch']})")

    preds = predict_all(model, test_ds, device)
    correct = [p in acc for p, acc in zip(preds, test_ds.accepted)]
    rows = [df.iloc[r] for r in test_ds.indices]
    cat_of = [str(r["category"]).lower() for r in rows]
    print(f"test accuracy recomputed from the checkpoint: {sum(correct) / len(correct):.4f}")

    # ---- selection ----
    rng = random.Random(a.seed)
    columns = []                                    # list of (category, [positions])
    if a.image_ids:
        pos = [i for i, r in enumerate(rows) if r["image_id"] in set(a.image_ids)]
        columns.append(("selected", pos))
    else:
        for c in CATS:
            ok = [i for i in range(len(rows)) if cat_of[i] == c and correct[i]]
            bad = [i for i in range(len(rows)) if cat_of[i] == c and not correct[i]]
            chosen = rng.sample(ok, min(a.per_cat, len(ok))) + rng.sample(bad, min(a.per_cat, len(bad)))
            columns.append((c, chosen))

    # ---- Grad-CAM ----
    cam_engine = GradCAM(model, last_conv_layer(model))
    records, panels = [], []
    for cat, positions in columns:
        row_panels = []
        for p in positions:
            item = test_ds[p]
            img = item["image"].unsqueeze(0).to(device)
            ids = item["input_ids"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)

            target = None
            if a.target == "gt" and test_ds.accepted[p]:
                target = torch.tensor([test_ds.accepted[p][0]], device=device)
            cam, raw_max, logits, tcls = cam_engine(img, ids, mask, target)
            pred = int(logits.argmax(1))

            r = rows[p]
            rec = {"row_idx": test_ds.indices[p], "image_id": r["image_id"], "category": cat_of[p],
                   "question": r["question"], "ground_truth": r["answer"], "prediction": vocab.idx2ans[pred],
                   "correct": bool(pred in test_ds.accepted[p]), "target_class": vocab.idx2ans[int(tcls)],
                   "cam_raw_max": float(raw_max)}
            records.append(rec)
            row_panels.append((denorm(item["image"], cfg.imagenet_mean, cfg.imagenet_std),
                               cam[0].cpu().numpy(), rec))
        panels.append((cat, row_panels))

    # ---- optional cross-check against the package used in the paper ----
    if a.verify_package and records:
        try:
            from pytorch_grad_cam import GradCAM as PkgGradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

            class ImageOnly(torch.nn.Module):
                def __init__(self, m, ids, mask):
                    super().__init__(); self.m, self.ids, self.mask = m, ids, mask
                def forward(self, x):
                    return self.m(x, self.ids.expand(x.size(0), -1), self.mask.expand(x.size(0), -1))

            p0 = columns[0][1][0]
            item = test_ds[p0]
            img = item["image"].unsqueeze(0).to(device)
            ids = item["input_ids"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            mine = cam_engine(img, ids, mask, None)
            pkg = PkgGradCAM(model=ImageOnly(model, ids, mask), target_layers=[last_conv_layer(model)])
            theirs = pkg(input_tensor=img, targets=[ClassifierOutputTarget(int(mine[3]))])[0]
            corr = np.corrcoef(mine[0][0].cpu().numpy().ravel(), theirs.ravel())[0, 1]
            print(f"cross-check vs pytorch-grad-cam: Pearson correlation {corr:.3f} (close to 1 expected)")
        except Exception as e:                        # optional check, never block the figure
            print(f"package cross-check skipped: {type(e).__name__}: {e}")
    cam_engine.remove()

    # ---- figure ----
    n_rows = len(panels)
    n_cols = max(len(rp) for _, rp in panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 3.9 * n_rows), squeeze=False)
    for ri, (cat, rp) in enumerate(panels):
        for ci in range(n_cols):
            ax = axes[ri][ci]
            ax.axis("off")
            if ci >= len(rp):
                continue
            image, cam, rec = rp[ci]
            ax.imshow(image)
            ax.imshow(cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
            colour = "green" if rec["correct"] else "red"
            ax.set_title(f"[{NAMES.get(cat, cat)}]\nQ: {wrap(rec['question'])}\nGT: {wrap(rec['ground_truth'], 30, 2)}\n"
                         f"Pred: {wrap(rec['prediction'], 30, 2)}", fontsize=6.5, color=colour, loc="left")
    fig.suptitle(f"Grad-CAM | run {a.run} | split {a.split} | last epoch | target = {a.target} class "
                 f"| layer = VGG-16 conv5_3 (features[28])", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir = cfg.results_dir / "reproduction" / a.run / f"split_{a.split}" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "gradcam_grid.png", dpi=200)
    with open(out_dir / "selection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    print(f"saved: {out_dir / 'gradcam_grid.png'}\nsaved: {out_dir / 'selection.csv'}")


if __name__ == "__main__":
    main()